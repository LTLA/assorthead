#ifndef SCRAN_GRAPH_CLUSTER_CLUSTER_LEIDEN_HPP
#define SCRAN_GRAPH_CLUSTER_CLUSTER_LEIDEN_HPP

#include <vector>
#include <algorithm>

#include "raiigraph/raiigraph.hpp"
#include "sanisizer/sanisizer.hpp"

#include "igraph.h"

/**
 * @file cluster_leiden.hpp
 * @brief Wrapper around **igraph**'s Leiden community detection algorithm.
 */

namespace scran_graph_cluster {

/**
 * @brief Options for `cluster_leiden()`.
 */
struct ClusterLeidenOptions {
    /**
     * Resolution of the clustering.
     * Larger values result in more fine-grained communities.
     * The default is based on `?cluster_leiden` in the **igraph** R package.
     */
    igraph_real_t resolution = 1;

    /**
     * Level of randomness used during refinement.
     * The default is based on `?cluster_leiden` in the **igraph** R package.
     */
    igraph_real_t beta = 0.01;

    /**
     * Number of iterations of the Leiden algorithm.
     * More iterations can improve separation at the cost of computational time.
     * If negative, the algorithm will iterate until convergence.
     * The default is based on `?cluster_leiden` in the **igraph** R package.
     */
    igraph_int_t iterations = 2;

    /**
     * Objective function to optimize.
     * This should be one of `IGRAPH_LEIDEN_OBJECTIVE_MODULARITY`, `IGRAPH_LEIDEN_OBJECTIVE_CPM` or `IGRAPH_LEIDEN_OBJECTIVE_ER`.
     * CPM typically yields more fine-grained clusters at the same choice of `ClusterLeidenOptions::resolution`.
     * The default is based on `?cluster_leiden` in the **igraph** R package.
     */
    igraph_leiden_objective_t objective = IGRAPH_LEIDEN_OBJECTIVE_CPM;

    /**
     * Whether to report the quality of the clustering in `Results::quality`.
     */
    bool report_quality = true;

    /**
     * Seed for the **igraph** random number generator.
     */
    igraph_uint_t seed = 42;
};

/**
 * @brief Result of `cluster_leiden()`.
 */
struct ClusterLeidenResults {
    /**
     * Vector of length equal to the number of cells, containing 0-indexed cluster identities.
     */
    raiigraph::IntegerVector membership;

    /**
     * Quality of the clustering, closely related to the modularity.
     * This is only defined if `ClusterLeidenOptions::report_quality = true`.
     */
    igraph_real_t quality = 0;
};

/**
 * Run the Leiden community detection algorithm on a pre-constructed graph to obtain communities of highly inter-connected nodes.
 * See [here](https://igraph.org/c/doc/igraph-Community.html#igraph_community_leiden) for more details on the Leiden algorithm.
 *
 * It is assumed that `igraph_setup()` or `raiigraph::initialize()` has already been called before running this function.
 *
 * @param graph A graph.
 * Typically, the nodes are cells and edges are formed between similar cells.
 * @param weights Pointer to an array of weights of length equal to the number of edges in `graph`. 
 * This should be in the same order as the edge list in `graph`.
 * Alternatively `NULL`, if the graph is unweighted.
 * @param options Further options.
 * @param[out] output On output, this is filtered with the clustering results.
 * The input value is ignored, so this object can be re-used across multiple calls to `cluster_leiden()`.
 */
inline void cluster_leiden(const igraph_t* graph, const igraph_vector_t* weights, const ClusterLeidenOptions& options, ClusterLeidenResults& output) {
    const auto membership = output.membership.get();
    const auto quality = (options.report_quality ? &(output.quality) : static_cast<igraph_real_t*>(NULL));

    const raiigraph::RNGScope rngs(options.seed);

    const auto status = igraph_community_leiden_simple(
        graph, 
        weights, 
        options.objective,
        options.resolution, 
        options.beta, 
        false, 
        options.iterations, 
        membership, 
        NULL, 
        quality
    );

    raiigraph::check_code(status);
}

/**
 * Overload of `cluster_leiden()` that accepts C++ containers instead of the raw **igraph** pointers.
 *
 * It is assumed that `igraph_setup()` or `raiigraph::initialize()` has already been called before running this function.
 *
 * @param graph A graph. 
 * Typically, the nodes are cells and edges are formed between similar cells.
 * @param weights Vector of weights of length equal to the number of edges in `graph`. 
 * This should be in the same order as the edge list in `graph`.
 * @param options Further options.
 *
 * @return Clustering results for the nodes of the graph.
 */
inline ClusterLeidenResults cluster_leiden(const raiigraph::Graph& graph, const std::vector<igraph_real_t>& weights, const ClusterLeidenOptions& options) {
    // No need to free this, as it's just a view.
    const auto weight_view = igraph_vector_view(weights.data(), sanisizer::cast<igraph_int_t>(weights.size()));

    ClusterLeidenResults output;
    cluster_leiden(graph.get(), &weight_view, options, output);
    return output;
}

}

#endif
