#ifndef MUMOSA_SIMPLE_HPP
#define MUMOSA_SIMPLE_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <type_traits>

#include "knncolle/knncolle.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file simple.hpp
 * @brief Compute distances to nearest neighbors.
 */

namespace mumosa {

/**
 * @brief Options for `compute_distance()`.
 */
struct Options {
    /**
     * Number of neighbors for the nearest neighbor search.
     * Larger values improve stability at the risk of including biological heterogeneity into the distance.
     * `num_neighbors + 1` can also be interpreted as the expected minimum size of each subpopulation.
     */
    int num_neighbors = 20;

    /**
     * Number of threads to use.
     * The parallelization mechanism is determined by `knncolle::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @tparam Index_ Integer type of the number of cells.
 * @tparam Distance_ Floating-point type of the distances.
 *
 * @param num_cells Number of cells.
 * @param[in, out] distances Pointer to an array of length `num_cells`, 
 * containing the distances from each cell to its \f$k\f$-nearest neighbor.
 * It is expected that the same \f$k\f$ was used for each cell.
 * On output, the order of values may be arbitrarily altered during the median calculation;
 * if this is undesirable, users should pass in a copy of the array.
 *
 * @return Pair containing the median distance to the nearest neighbor (first)
 * and the root-mean-squared distance across all cells (second).
 * These values can be used in `compute_scale()`.
 */
template<typename Index_, typename Distance_>
std::pair<Distance_, Distance_> compute_distance(const Index_ num_cells, Distance_* const distances) {
    const Distance_ med = tatami_stats::medians::direct(distances, num_cells, /* skip_nan = */ false);
    Distance_ rmsd = 0;
    for (Index_ i = 0; i < num_cells; ++i) {
        const auto d = distances[i];
        rmsd += d * d;
    }
    rmsd = std::sqrt(rmsd);
    return std::make_pair(med, rmsd);
}

/**
 * Overload of `compute_distance()` that accepts a pre-built neighbor search index.
 *
 * @tparam Index_ Integer type of the number of cells.
 * @tparam Input_ Numeric type of the input data used to build the search index.
 * This is only required to define the `knncolle::Prebuilt` class and is otherwise ignored.
 * @tparam Distance_ Floating-point type of the distances.
 *
 * @param prebuilt A prebuilt neighbor search index for a modality-specifi embedding.
 * @param[out] distances Pointer to an array of length `prebuilt.num_observations()`,
 * containing the distances from each cell to its \f$k\f$-nearest neighbor.
 * This may not be ordered on output.
 * @param options Further options.
 *
 * @return Pair containing the median distance to the `Options::num_neighbors`-th nearest neighbor (first)
 * and the root-mean-squared distance across all cells (second).
 * These values can be used in `compute_scale()`.
 */
template<typename Index_, typename Input_, typename Distance_>
std::pair<Distance_, Distance_> compute_distance(
    const knncolle::Prebuilt<Index_, Input_, Distance_>& prebuilt,
    Distance_* const distances,
    const Options& options
) {
    const Index_ nobs = prebuilt.num_observations();
    const auto capped_k = knncolle::cap_k(options.num_neighbors, nobs);

    knncolle::parallelize(options.num_threads, nobs, [&](const int, const Index_ start, const Index_ length) -> void {
        const auto searcher = prebuilt.initialize();
        std::vector<Distance_> cur_distances;
        for (Index_ i = start, end = start + length; i < end; ++i) {
            searcher->search(i, capped_k, NULL, &cur_distances);
            if (cur_distances.size()) {
                distances[i] = cur_distances.back();
            }
        }
    });

    return compute_distance(nobs, distances);
}

/**
 * Overload of `compute_distance()` that accepts an embedding matrix.
 *
 * @tparam Index_ Integer type of the number of cells.
 * @tparam Input_ Numeric type of the input data.
 * @tparam Distance_ Floating-point type of the distances.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 *
 * @param num_dim Number of dimensions in the embedding.
 * @param num_cells Number of cells in the embedding.
 * @param[in] data Pointer to an array containing the embedding matrix for a modality.
 * This should be stored in column-major layout where each row is a dimension and each column is a cell.
 * @param builder Algorithm to use for the neighbor search.
 * @param options Further options.
 *
 * @return Pair containing the median distance to the `Options::num_neighbors`-th nearest neighbor (first)
 * and the root-mean-squared distance across all cells (second).
 * These values can be used in `compute_scale()`.
 */
template<typename Index_, typename Input_, typename Distance_, class Matrix_ = knncolle::Matrix<Index_, Input_> >
std::pair<Distance_, Distance_> compute_distance(
    const std::size_t num_dim,
    const Index_ num_cells,
    const Input_* const data,
    const knncolle::Builder<Index_, Input_, Distance_, Matrix_>& builder,
    const Options& options
) {
    auto dist = sanisizer::create<std::vector<Distance_> >(num_cells);
    const auto prebuilt = builder.build_unique(knncolle::SimpleMatrix(num_dim, num_cells, data));
    return compute_distance(*prebuilt, dist.data(), options);
}

}

#endif
