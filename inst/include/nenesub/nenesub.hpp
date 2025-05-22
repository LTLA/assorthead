#ifndef NENESUB_HPP
#define NENESUB_HPP

#include <vector>
#include <queue>
#include <cstddef>
#include <algorithm>

#include "knncolle/knncolle.hpp"

/**
 * @file nenesub.hpp
 * @brief Nearest-neighbors subsampling.
 */

/**
 * @namespace nenesub
 * @brief Nearest-neighbors subsampling.
 */
namespace nenesub {

/**
 * @brief Options for `compute()`.
 */
struct Options {
    /**
     * The number of nearest neighbors to use, i.e., \f$k\f$. 
     * Only relevant for the `compute()` overloads without pre-computed neighbors.
     */
    int num_neighbors = 20;

    /**
     * The minimum number of remaining neighbors that an observation must have in order to be selected, i.e., \f$m\f$.
     * This should be less than or equal to `Options::num_neighbors`.
     */
    int min_remaining = 10;

    /**
     * The number of threads to use.
     * This uses the parallelization scheme defined by `knncolle::parallelize()`.
     * Only relevant for the `compute()` overloads that perform a neighbor search.
     */
    int num_threads = 10;
};

/**
 * This function generates a deterministic subsampling of a dataset based on nearest neighbors.
 * We first identify the \f$k\f$-nearest neighbors of each observation and use that to define its local neighborhood.
 * We select an observation for subsampling if it:
 *
 * - Does not belong in the local neighborhood of any previously selected observation.
 * - Has the most neighbors that are not selected or in the local neighborhoods of previously selected observations.
 *   Ties are broken using the smallest distance to each observation's \f$k\f$-th neighbor (i.e., the densest region of space).
 * - Has at least \f$m\f$ neighbors that are not selected or in the local neighborhoods of any other selected observation.
 *
 * We repeat this process until there are no more observations that satisfy these requirements. 
 *
 * Each selected observation effectively serves as a representative for up to \f$k\f$ of its nearest neighbors.
 * As such, the rate of subsampling is roughly proportional to the choice of \f$k\f$.
 * A non-zero \f$m\f$ ensures that there are enough neighbors to warrant the selection of an observation,
 * to protect against overrepresentation of outlier points that are not in any observation's neighborhood.
 * Some testing suggests that the dataset is subsampled by a factor of \f$k\f$, though this can increase or decrease for smaller or larger \f$m\f$, respectively.
 *
 * The **nenesub** approach ensures that the subsampled points are well-distributed across the dataset.
 * Low-frequency subpopulations will always have at least a few representatives if they are sufficiently distant from other subpopulations.
 * In contrast, random sampling does not provide strong guarantees for capture of a rare subpopulation.
 * We also preserve the relative density across the dataset as more representatives will be generated from high-density regions. 
 * This simplifies the interpretation of analysis results generated from the subsetted dataset.
 * 
 * @tparam Index_ Integer type for the observation indices.
 * @tparam GetNeighbors_ Function that accepts an `Index_` index and returns a (const reference to a) container-like object.
 * The container should be support the `[]` operator and have a `size()` method.
 * @tparam GetIndex_ Function that accepts a (const reference to a) container of the type returned by `GetNeighbors_` and an `Index_` into that container, and returns `Index_`.
 * @tparam GetNeighbors_ Function that accepts an `Index_` index and returns a distance value, typically floating-point.
 *
 * @param num_obs Number of observations in the dataset.
 * @param get_neighbors Function that accepts an integer observation index in `[0, num_obs)` and returns a container of that observation's neighbors.
 * Each element of the container specifies the index of a neighboring observation.
 * It is generally expected that the returned containers have the same size for all indices.
 * @param get_index Function to return the index of each neighbor, given the container returned by `get_neighbors` and an index into that container.
 * @param get_max_distance Function that accepts an integer observation index in `[0, num_obs)` and returns the distance from that observation to its furthest neighbor.
 * @param options Further options. 
 * Note that `Options::num_neighbors` and `Options::num_threads` are ignored here.
 * @param[out] selected On output, the indices of the observations that were subsampled.
 * These are sorted in ascending order.
 */
template<typename Index_, class GetNeighbors_, class GetIndex_, class GetMaxDistance_>
void compute(Index_ num_obs, GetNeighbors_ get_neighbors, GetIndex_ get_index, GetMaxDistance_ get_max_distance, const Options& options, std::vector<Index_>& selected) {
    typedef decltype(get_max_distance(0)) Distance;
    struct Payload {
        Payload(Index_ identity, Index_ remaining, Distance max_distance) : remaining(remaining), identity(identity), max_distance(max_distance) {}
        Index_ remaining;
        Index_ identity;
        Distance max_distance;
    };

    auto cmp = [](const Payload& left, const Payload& right) -> bool {
        if (left.remaining == right.remaining) {
            if (left.max_distance == right.max_distance) {
                return left.identity > right.identity; // smallest identities show up first.
            }
            return left.max_distance > right.max_distance; // smallest distances show up first.
        }
        return left.remaining < right.remaining; // largest remaining show up first.
    };
    std::priority_queue<Payload, std::vector<Payload>, decltype(cmp)> store(
        cmp,
        [&]{
            std::vector<Payload> container;
            container.reserve(num_obs);
            return container;
        }()
    );

    std::vector<std::vector<Index_> > reverse_map(num_obs);
    std::vector<Index_> remaining(num_obs);
    for (Index_ c = 0; c < num_obs; ++c) {
        const auto& neighbors = get_neighbors(c);
        Index_ nneighbors = neighbors.size();

        if (nneighbors) { // protect get_max_distance just in case there are no neighbors.
            store.emplace(c, nneighbors, get_max_distance(c));
            for (Index_ n = 0; n < nneighbors; ++n) {
                reverse_map[get_index(neighbors, n)].push_back(c);
            }
            remaining[c] = nneighbors;
        }
    }

    selected.clear();
    std::vector<unsigned char> tainted(num_obs);
    Index_ min_remaining = options.min_remaining;
    while (!store.empty()) {
        auto payload = store.top();
        store.pop();
        if (tainted[payload.identity]) {
            continue;
        }

        const auto& neighbors = get_neighbors(payload.identity);
        Index_ new_remaining = remaining[payload.identity];

        if (new_remaining >= min_remaining) {
            payload.remaining = new_remaining;
            if (!store.empty() && cmp(payload, store.top())) {
                store.push(payload);
            } else {
                selected.push_back(payload.identity);
                tainted[payload.identity] = 1;
                for (auto x : reverse_map[payload.identity]) {
                    --remaining[x];
                }

                Index_ nneighbors = neighbors.size();
                for (Index_ n = 0; n < nneighbors; ++n) {
                    auto current = get_index(neighbors, n);
                    tainted[current] = 1;
                    for (auto x : reverse_map[current]) {
                        --remaining[x];
                    }
                }
            }
        }
    }

    std::sort(selected.begin(), selected.end());
}

/**
 * Overload to enable convenient usage with pre-computed neighbors from **knncolle**.
 *
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Distance_ Floating-point type for the distances.
 *
 * @param neighbors Vector of nearest-neighbor search results for each observation.
 * Each entry is a pair containing a vector of neighbor indices and a vector of distances to those neighbors.
 * Neighbors should be sorted by increasing distance.
 * The same number of neighbors should be present for each observation.
 * @param options Further options. 
 * Note that `Options::num_neighbors` and `Options::num_threads` are ignored here.
 *
 * @return A sorted vector of the indices of the subsampled observations.
 */
template<typename Index_, typename Distance_>
std::vector<Index_> compute(const knncolle::NeighborList<Index_, Distance_>& neighbors, const Options& options) {
    std::vector<Index_> output;
    compute(
        static_cast<Index_>(neighbors.size()),
        [&](Index_ i) -> const auto& { return neighbors[i]; }, 
        [](const std::vector<std::pair<Index_, Distance_> >& x, Index_ n) -> Index_ { return x[n].first; }, 
        [&](Index_ i) -> Distance_ { return neighbors[i].back().second; }, 
        options,
        output
    );
    return output;
}

/**
 * Overload to enable convenient usage with a prebuilt nearest-neighbor search index from **knncolle**.
 *
 * @tparam Dim_ Integer type for the dimension index.
 * @tparam Index_ Integer type for the observation index.
 * @tparam Input_ Numeric type for the input data used to build the search index.
 * This is only required to define the `knncolle::Prebuilt` class and is otherwise ignored.
 * @tparam Distance_ Floating-point type for the distances.
 *
 * @param[in] prebuilt A prebuilt nearest-neighbor search index on the observations of interest.
 * @param options Further options.
 *
 * @return A sorted vector of the indices of the subsampled observations.
 */
template<typename Index_, typename Input_, typename Distance_>
std::vector<Index_> compute(const knncolle::Prebuilt<Index_, Input_, Distance_>& prebuilt, const Options& options) {
    int k = options.num_neighbors;
    if (k < options.min_remaining) {
        throw std::runtime_error("number of neighbors is less than 'min_remaining'");
    }

    Index_ nobs = prebuilt.num_observations();
    auto capped_k = knncolle::cap_k(k, nobs);
    std::vector<std::vector<Index_> > nn_indices(nobs);
    std::vector<Distance_> max_distance(nobs);

    knncolle::parallelize(options.num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto sptr = prebuilt.initialize();
        std::vector<Distance_> nn_distances;
        for (Index_ i = start, end = start + length; i < end; ++i) {
            sptr->search(i, capped_k, &(nn_indices[i]), &nn_distances);
            max_distance[i] = (capped_k ? 0 : nn_distances.back());
        }
    });

    std::vector<Index_> output;
    compute(
        nobs,
        [&](Index_ i) -> const std::vector<Index_>& { return nn_indices[i]; }, 
        [](const std::vector<Index_>& x, Index_ n) -> Index_ { return x[n]; }, 
        [&](Index_ i) -> Distance_ { return max_distance[i]; },
        options,
        output
    );
    return output;
}

/**
 * Overload to enable convenient usage with a column-major array of coordinates for each observation.
 *
 * @tparam Dim_ Integer type for the dimension index.
 * @tparam Index_ Integer type for the observation index.
 * @tparam Input_ Numeric type for the input data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 *
 * @param num_dims Number of dimensions for the observation coordinates.
 * @param num_obs Number of observations in the dataset.
 * @param[in] data Pointer to a `num_dims`-by-`num_observations` column-major array of observation coordinates where rows are dimensions and columns are observations.
 * @param knn_method Specification of the nearest-neighbor search algorithm, e.g., `knncolle::VptreeBuilder`, `knncolle::KmknnBuilder`.
 * @param options Further options.
 *
 * @return A sorted vector of the indices of the subsampled observations.
 */
template<typename Index_, typename Input_, typename Distance_, class Matrix_ = knncolle::Matrix<Index_, Input_> >
std::vector<Index_> compute(
    std::size_t num_dims, 
    Index_ num_obs, 
    const Input_* data, 
    const knncolle::Builder<Index_, Input_, Distance_, Matrix_>& knn_method,
    const Options& options) 
{
    auto prebuilt = knn_method.build_unique(knncolle::SimpleMatrix<Index_, Input_>(num_dims, num_obs, data));
    return compute(*prebuilt, options);
}

}

#endif
