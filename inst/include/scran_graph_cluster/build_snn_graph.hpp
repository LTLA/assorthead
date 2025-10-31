#ifndef SCRAN_GRAPH_CLUSTER_BUILD_SNN_GRAPH_HPP
#define SCRAN_GRAPH_CLUSTER_BUILD_SNN_GRAPH_HPP

#include <vector>
#include <algorithm>
#include <memory>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

#if __has_include("igraph.h")
#include "raiigraph/raiigraph.hpp"
#include "edges_to_graph.hpp"
#endif

/**
 * @file build_snn_graph.hpp
 * @brief Build a shared nearest-neighbor graph on the cells.
 */

namespace scran_graph_cluster {

/** 
 * Choice of edge weighting schemes during graph construction in `build_snn_graph()`.
 *
 * Let \f$k\f$ be the number of nearest neighbors for each node, not including the node itself.
 * 
 * - `RANKED` defines the weight of the edge between two nodes as \f$k - r/2\f$ where \f$r\f$ is the smallest sum of ranks for any shared neighboring node (Xu and Su, 2015).
 *   More shared neighbors, or shared neighbors that are close to both observations, will generally yield larger weights.
 *   For the purposes of this ranking, each node has a rank of zero in its own nearest-neighbor set. 
 *   If only the furthest neighbor is shared between nodes (i.e., \f$r = 2f\f$, the weight is set to 1e-6 to distinguish this edge from pairs of cells with no shared neighbors.
 * - `NUMBER` defines the weight of the edge between two nodes as the number of shared nearest neighbors between them. 
 *   The weight can range from zero to \f$k + 1\f$, as we include the node itself. 
 *   This is a simpler scheme that is also slightly faster but does not account for the ranking of neighbors within each set.
 * - `JACCARD` defines the weight of the edge between two nodes as the Jaccard index of their neighbor sets,
 *   motivated by the algorithm used by the [**Seurat** R package](https://cran.r-project.org/package=seurat).
 *   This weight can range from zero to 1, and is a monotonic transformation of the weight used by `NUMBER`.
 *
 * @see
 * Xu C and Su Z (2015).
 * Identification of cell types from single-cell transcriptomes using a novel clustering method.
 * _Bioinformatics_ 31, 1974-80
 */
enum class SnnWeightScheme : char { RANKED, NUMBER, JACCARD };

/**
 * @brief Options for SNN graph construction.
 */
struct BuildSnnGraphOptions {
    /**
     * Number of nearest neighbors to use for graph construction.
     * Larger values increase the connectivity of the graph and reduce the granularity of subsequent community detection steps, at the cost of speed.
     * Only relevant for the `build_snn_graph()` overloads without pre-computed neighbors.
     */
    int num_neighbors = 10;

    /**
     * Weighting scheme for each edge, based on the number of shared nearest neighbors of the two nodes.
     */
    SnnWeightScheme weighting_scheme = SnnWeightScheme::RANKED;

    /**
     * Number of threads to use.
     * The parallelization scheme is defined by `knncolle::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Results of SNN graph construction.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 */
template<typename Node_, typename Weight_>
struct BuildSnnGraphResults {
    /**
     * Number of cells in the dataset.
     */
    Node_ num_cells;

    /**
     * Vector of paired indices defining the edges between cells.
     * The number of edges is half the length of `edges`, where `edges[2*i]` and `edges[2*i+1]` define the vertices for edge `i`.
     */
    std::vector<Node_> edges;

    /**
     * Vector of weights for the edges.
     * This is of length equal to the number of edges, where each `weights[i]` corresponds to an edge `i` in `edges`. 
     */
    std::vector<Weight_> weights;
};

#if __has_include("igraph.h")
typedef igraph_int_t DefaultNode;
typedef igraph_real_t DefaultWeight;
#else
/**
 * Default type of the node indices.
 * Set to `igraph_int_t` if **igraph** is available.
 */
typedef int DefaultNode;

/**
 * Default type of the edge weights.
 * Set to `igraph_real_t` if **igraph** is available.
 */
typedef double DefaultWeight;
#endif

/**
 * @cond
 */
template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(Input_ x) {
    return x;
}
/**
 * @endcond
 */

/**
 * In a shared nearest-neighbor graph, two cells are connected to each other by an edge if they share any of their nearest neighbors.
 * The weight of this edge is determined from the number or ranking of their shared nearest neighbors.
 * If two cells are close together but have distinct sets of neighbors, the corresponding edge is downweighted as the two cells are unlikely to be part of the same neighborhood.
 * In this manner, strongly weighted edges will only form within highly interconnected neighborhoods where many cells share the same neighbors.
 * This provides a more sophisticated definition of similarity between cells compared to a simpler (unweighted) nearest neighbor graph that just focuses on immediate proximity.
 * Community detection algorithms (e.g., `cluster_multilevel()`) can then be applied to the graph to identify clusters of cells.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 * @tparam Index_ Integer type of the observation index.
 * @tparam GetNeighbors_ Function that accepts an `Index_` cell index and returns a (const reference to) a container-like object.
 * The container should be iterable in a range-based for loop, support the `[]` operator, and have a `size()` method.
 * @tparam GetIndex_ Function that accepts an element of the container type returned by `GetNeighbors_` and returns an `Index_` containing its observation index.
 *
 * @param num_cells Number of cells in the dataset.
 * @param get_neighbors Function that accepts an integer cell index in `[0, num_cells)` and returns a container of that cell's neighbors.
 * Each element of the container corresponds to a neighbor, and neighbors should be sorted by increasing distance from the cell.
 * The same number of neighbors should be identified for each cell.
 * @param get_index Function to return the index of each neighbor, given an element of the container returned by `get_neighbors`.
 * In trivial cases, this is the identity function but it can be more complex depending on the contents of the inner container.
 * @param options Further options for graph construction.
 * Note that `BuildSnnGraphOptions::num_neighbors` is ignored here.
 * @param[out] output On output, the edges and weights of the SNN graph.
 * The input value is ignored so this can be re-used across multiple calls to `build_snn_graph()`.
 */
template<typename Node_ = DefaultNode, typename Weight_ = DefaultWeight, typename Index_, class GetNeighbors_, class GetIndex_>
void build_snn_graph(const Index_ num_cells, const GetNeighbors_ get_neighbors, const GetIndex_ get_index, const BuildSnnGraphOptions& options, BuildSnnGraphResults<Node_, Weight_>& output) {
    // Reverse mapping is not parallel-frendly, so we don't construct this with the neighbor search.
    std::vector<std::vector<Node_> > simple_hosts;
    std::vector<std::vector<std::pair<Node_, Weight_> > > ranked_hosts;

    // Check that all implicit casts from Index_ to Node_ will be safe.
    sanisizer::cast<Node_>(num_cells);

    if (options.weighting_scheme == SnnWeightScheme::RANKED) {
        sanisizer::resize(ranked_hosts, num_cells);
        for (Index_ i = 0; i < num_cells; ++i) {
            ranked_hosts[i].emplace_back(i, 0); // each point is its own 0-th nearest neighbor
            const auto& current = get_neighbors(i);
            Weight_ rank = 1;
            for (const auto& x : current) {
                ranked_hosts[get_index(x)].emplace_back(i, rank);
                ++rank;
            }
        }
    } else {
        sanisizer::resize(simple_hosts, num_cells);
        for (Index_ i = 0; i < num_cells; ++i) {
            simple_hosts[i].emplace_back(i); // each point is its own 0-th nearest neighbor
            const auto& current = get_neighbors(i);
            for (const auto& x : current) {
                simple_hosts[get_index(x)].emplace_back(i);
            }
        }
    }

    // Constructing the shared neighbor graph.
    auto edge_stores = sanisizer::create<std::vector<std::vector<Node_> > >(num_cells);
    auto weight_stores = sanisizer::create<std::vector<std::vector<Weight_> > >(num_cells);

    knncolle::parallelize(options.num_threads, num_cells, [&](const int, const Index_ start, const Index_ length) -> void {
        auto current_score = sanisizer::create<std::vector<Weight_> >(num_cells);
        std::vector<Node_> current_added;
        current_added.reserve(num_cells);

        for (Index_ j = start, end = start + length; j < end; ++j) {
            const auto& current_neighbors = get_neighbors(j);
            const auto nneighbors = current_neighbors.size();

            for (decltype(I(nneighbors)) i = 0; i <= nneighbors; ++i) {
                // First iteration treats 'j' as the zero-th neighbor.
                // Remaining iterations go through the neighbors of 'j'.
                const Node_ cur_neighbor = (i == 0 ? j : get_index(current_neighbors[i-1]));

                // Going through all observations 'h' for which 'cur_neighbor'
                // is a nearest neighbor, a.k.a., 'cur_neighbor' is a shared
                // neighbor of both 'h' and 'j'.
                if (options.weighting_scheme == SnnWeightScheme::RANKED) {
                    for (const auto& h : ranked_hosts[cur_neighbor]) {
                        const auto othernode = h.first;
                        if (othernode < static_cast<Node_>(j)) { // avoid duplicates from symmetry in the SNN calculations.
                            auto& existing_other = current_score[othernode];

                            // Recording the lowest combined rank per neighbor. 
                            const Weight_ currank = h.second + static_cast<Weight_>(i);
                            if (existing_other == 0) { 
                                existing_other = currank;
                                current_added.push_back(othernode);
                            } else if (existing_other > currank) {
                                existing_other = currank;
                            }
                        }
                    }

                } else {
                    for (const auto& othernode : simple_hosts[cur_neighbor]) {
                        if (othernode < static_cast<Node_>(j)) { // avoid duplicates from symmetry in the SNN calculations.
                            auto& existing_other = current_score[othernode];

                            // Recording the number of shared neighbors.
                            if (existing_other == 0) { 
                                current_added.push_back(othernode);
                            } 
                            ++existing_other;
                        }
                    }
                }
            }
           
            // Converting to edges.
            auto& current_edges = edge_stores[j];
            current_edges.reserve(current_added.size() * 2);
            auto& current_weights = weight_stores[j];
            current_weights.reserve(current_added.size());

            for (const auto othernode : current_added) {
                current_edges.push_back(j);
                current_edges.push_back(othernode);

                Weight_& otherscore = current_score[othernode];
                current_weights.push_back([&]{
                    if (options.weighting_scheme == SnnWeightScheme::RANKED) {
                        const Weight_ preliminary = static_cast<Weight_>(nneighbors) - otherscore / 2;
                        return std::max(preliminary, static_cast<Weight_>(1e-6)); // Ensuring that an edge with a positive weight is always reported.
                    } else if (options.weighting_scheme == SnnWeightScheme::JACCARD) {
                        return otherscore / (2 * (static_cast<Weight_>(nneighbors) + 1) - otherscore);
                    } else {
                        return otherscore;
                    }
                }());

                // Resetting all those added to zero.
                otherscore = 0;
            }
            current_added.clear();
        }
    });

    // Collating the total number of edges.
    std::size_t nedges = 0;
    for (const auto& w : weight_stores) {
        nedges = sanisizer::sum<std::size_t>(nedges, w.size());
    }

    output.num_cells = num_cells;
    output.weights.clear();
    output.weights.reserve(nedges);
    for (const auto& w : weight_stores) {
        output.weights.insert(output.weights.end(), w.begin(), w.end());
    }
    weight_stores.clear();
    weight_stores.shrink_to_fit(); // forcibly release memory so that we have some more space for edges.

    output.edges.clear();
    output.edges.reserve(nedges * 2);
    for (const auto& e : edge_stores) {
        output.edges.insert(output.edges.end(), e.begin(), e.end());
    }

    return;
}

/**
 * Overload of `build_snn_graph()` to enable convenient usage with pre-computed neighbors from **knncolle**.
 * Distances are ignored here; only the ordering of neighbor indices is used.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Distance_ Floating-point type of the distances.
 *
 * @param neighbors Vector of nearest-neighbor search results for each cell.
 * Each entry is a pair containing a vector of neighbor indices and a vector of distances to those neighbors.
 * Neighbors should be sorted by increasing distance.
 * The same number of neighbors should be present for each cell.
 * @param options Further options for graph construction.
 * Note that `BuildSnnGraphOptions::num_neighbors` is ignored here.
 *
 * @return The edges and weights of the SNN graph.
 */
template<typename Node_ = DefaultNode, typename Weight_ = DefaultWeight, typename Index_, typename Distance_>
BuildSnnGraphResults<Node_, Weight_> build_snn_graph(const knncolle::NeighborList<Index_, Distance_>& neighbors, const BuildSnnGraphOptions& options) {
    BuildSnnGraphResults<Node_, Weight_> output;
    build_snn_graph(
        sanisizer::cast<Index_>(neighbors.size()), 
        [&](const Index_ i) -> const std::vector<std::pair<Index_, Distance_> >& { return neighbors[i]; }, 
        [](const std::pair<Index_, Distance_>& x) -> Index_ { return x.first; }, 
        options,
        output
    );
    return output;
}

/**
 * Overload of `build_snn_graph()` to enable convenient usage with pre-computed neighbors from **knncolle**.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 * @tparam Index_ Integer type of the neighbor indices.
 *
 * @param neighbors Vector of vectors of indices for the neighbors for each cell, sorted by increasing distance.
 * It is generally expected (though not strictly required) that the same number of neighbors are present for each cell.
 * @param options Further options for graph construction.
 * Note that `BuildSnnGraphOptions::num_neighbors` is ignored here.
 *
 * @return The edges and weights of the SNN graph.
 */
template<typename Node_ = int, typename Weight_ = double, typename Index_>
BuildSnnGraphResults<Node_, Weight_> build_snn_graph(const std::vector<std::vector<Index_> >& neighbors, const BuildSnnGraphOptions& options) {
    BuildSnnGraphResults<Node_, Weight_> output;
    build_snn_graph(
        sanisizer::cast<Index_>(neighbors.size()),
        [&](const Index_ i) -> const std::vector<Index_>& { return neighbors[i]; }, 
        [](const Index_ x) -> Index_ { return x; }, 
        options,
        output
    );
    return output;
}

/**
 * Overload of `build_snn_graph()` to enable convenient usage with a prebuilt nearest-neighbor search index from **knncolle**.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 * @tparam Index_ Integer type of the cell index.
 * @tparam Input_ Numeric type of the input data used to build the search index.
 * This is only required to define the `knncolle::Prebuilt` class and is otherwise ignored.
 * @tparam Distance_ Floating-point type of the distances.
 *
 * @param[in] prebuilt A prebuilt nearest-neighbor search index on the cells of interest.
 * @param options Further options for graph construction.
 *
 * @return The edges and weights of the SNN graph.
 */
template<typename Node_ = DefaultNode, typename Weight_ = DefaultWeight, typename Index_, typename Input_, typename Distance_>
BuildSnnGraphResults<Node_, Weight_> build_snn_graph(const knncolle::Prebuilt<Index_, Input_, Distance_>& prebuilt, const BuildSnnGraphOptions& options) {
    const auto neighbors = knncolle::find_nearest_neighbors_index_only(prebuilt, options.num_neighbors, options.num_threads);
    return build_snn_graph<Node_, Weight_>(neighbors, options);
}

/**
 * Overload of `build_snn_graph()` to enable convenient usage with a column-major array of cell coordinates.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 * @tparam Index_ Integer type of the cell index.
 * @tparam Input_ Numeric type of the input data.
 * @tparam Distance_ Floating-point type of the distances.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 *
 * @param num_dims Number of dimensions for the cell coordinates.
 * @param num_cells Number of cells in the dataset.
 * @param[in] data Pointer to a `num_dims`-by-`num_cells` column-major array of cell coordinates where rows are dimensions and columns are cells.
 * @param knn_method Specification of the nearest-neighbor search algorithm, e.g., `knncolle::VptreeBuilder`. 
 * @param options Further options for graph construction.
 *
 * @return The edges and weights of the SNN graph.
 */
template<typename Node_ = DefaultNode, typename Weight_ = DefaultWeight, typename Index_, typename Input_, typename Distance_, class Matrix_ = knncolle::Matrix<Index_, Input_> >
BuildSnnGraphResults<Node_, Weight_> build_snn_graph(
    const std::size_t num_dims, 
    const Index_ num_cells, 
    const Input_* data, 
    const knncolle::Builder<Index_, Input_, Distance_, Matrix_>& knn_method,
    const BuildSnnGraphOptions& options) 
{
    const auto prebuilt = knn_method.build_unique(knncolle::SimpleMatrix(num_dims, num_cells, data));
    return build_snn_graph<Node_, Weight_>(*prebuilt, options);
}

#if __has_include("igraph.h")
/**
 * Convert the edges in `BuildSnnGraphResults` to a **igraph** graph object for use in **igraph** functions.
 * See also `edges_to_graph()`.
 *
 * @tparam Node_ Integer type of the node indices.
 * @tparam Weight_ Floating-point type of the edge weights.
 *
 * @param result Result of `build_snn_graph()`, containing the edges of the SNN graph.
 *
 * @return The SNN graph as an **igraph**-compatible object.
 */
template<typename Node_ = DefaultNode, typename Weight_ = DefaultWeight>
raiigraph::Graph convert_to_graph(const BuildSnnGraphResults<Node_, Weight_>& result) {
    return edges_to_graph(result.edges.size(), result.edges.data(), result.num_cells, IGRAPH_UNDIRECTED);
}
#endif

}

#endif
