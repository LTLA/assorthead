#ifndef MUMOSA_HPP
#define MUMOSA_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "tatami_stats/tatami_stats.hpp"

/**
 * @file mumosa.hpp
 * @brief Scale multi-modal embeddings based on their relative variance.
 */

/**
 * @namespace mumosa
 * @brief Scale multi-modal embeddings to adjust for differences in variance.
 */
namespace mumosa {

/**
 * @brief Options for `compute_distance()`.
 */
struct Options {
    /**
     * Number of neighbors for the nearest neighbor search.
     * This can be interpreted as the minimum size of each subpopulation.
     */
    int num_neighbors = 20;

    /**
     * Number of threads to use.
     * The parallelization mechanism is determined by `knncolle::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @tparam Index_ Integer type for the number of cells.
 * @tparam Distance_ Floating-point type for the distances.
 *
 * @param num_cells Number of cells.
 * @param[in, out] distances Pointer to an array containing the distances from each cell to its \f$k\f$-nearest neighbor.
 * It is expected that the same \f$k\f$ was used for each cell.
 * On output, the order of values may be arbitrarily altered during the median calculation;
 * if this is undesirable, users should pass in a copy of the array.
 *
 * @return Pair containing the median distance to the nearest neighbor (first)
 * and the root-mean-squared distance across all cells (second).
 * These values can be used in `compute_scale()`.
 */
template<typename Index_, typename Distance_>
std::pair<Distance_, Distance_> compute_distance(Index_ num_cells, Distance_* distances) {
    Distance_ med = tatami_stats::medians::direct(distances, num_cells, /* skip_nan = */ false);
    Distance_ rmsd = 0;
    for (Index_ i = 0; i < num_cells; ++i) {
        auto d = distances[i];
        rmsd += d * d;
    }
    rmsd = std::sqrt(rmsd);
    return std::make_pair(med, rmsd);
}

/**
 * @tparam Index_ Integer type for the number of cells.
 * @tparam Input_ Numeric type for the input data used to build the search index.
 * This is only required to define the `knncolle::Prebuilt` class and is otherwise ignored.
 * @tparam Distance_ Floating-point type for the distances.
 *
 * @param prebuilt A prebuilt neighbor search index for a modality-specifi embedding.
 * @param options Further options.
 *
 * @return Pair containing the median distance to the `Options::num_neighbors`-th nearest neighbor (first)
 * and the root-mean-squared distance across all cells (second).
 * These values can be used in `compute_scale()`.
 */
template<typename Index_, typename Input_, typename Distance_>
std::pair<Distance_, Distance_> compute_distance(const knncolle::Prebuilt<Index_, Input_, Distance_>& prebuilt, const Options& options) {
    Index_ nobs = prebuilt.num_observations();
    auto capped_k = knncolle::cap_k(options.num_neighbors, nobs);
    std::vector<double> dist(nobs);

    knncolle::parallelize(options.num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto searcher = prebuilt.initialize();
        std::vector<Distance_> distances;
        for (Index_ i = start, end = start + length; i < end; ++i) {
            searcher->search(i, capped_k, NULL, &distances);
            if (distances.size()) {
                dist[i] = distances.back();
            }
        }
    });

    return compute_distance(nobs, dist.data());
}

/**
 * @tparam Index_ Integer type for the number of cells.
 * @tparam Input_ Numeric type for the input data.
 * @tparam Distance_ Floating-point type for the distances.
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
    std::size_t num_dim,
    Index_ num_cells,
    const Input_* data,
    const knncolle::Builder<Index_, Input_, Distance_, Matrix_>& builder,
    const Options& options)
{
    auto prebuilt = builder.build_unique(knncolle::SimpleMatrix(num_dim, num_cells, data));
    return compute_distance(*prebuilt, options);
}

/**
 * Compute the scaling factor to be applied to an embedding of a "target" modality relative to a reference modality.
 * This aims to scale the target so that the within-population variance is equal to that of the reference.
 *
 * Advanced users may want to scale the target so that its variance is some \f$S\f$-fold of the reference, e.g., to give more weight to more important modalities.
 * This can be achieved by multiplying the scaling factor by \f$\sqrt{S}\f$. 
 *
 * @tparam Distance_ Floating-point type for the distances.
 *
 * @param ref Output of `compute_distance()` for the embedding of the reference modality.
 * The first value contains the median distance while the second value contains the root-mean squared distance (RMSD).
 * @param target Output of `compute_distance()` for the embedding of the target modality.
 *
 * @return A scaling factor to apply to the embedding of the target modality, defined as the ratio of the median distances.
 * If either of the median distances is zero, this function instead returns the ratio of the RMSDs.
 * If the reference RMSD is zero, this function will return zero;
 * if the target RMSD is zero, this function will return positive infinity.
 */
template<typename Distance_>
Distance_ compute_scale(const std::pair<Distance_, Distance_>& ref, const std::pair<Distance_, Distance_>& target) {
    if (target.first == 0 || ref.first == 0) {
        if (target.second == 0) {
            return std::numeric_limits<Distance_>::infinity();
        } else if (ref.second == 0) {
            return 0;
        } else {
            return ref.second / target.second; 
        }
    } else {
        return ref.first / target.first;
    }
}

/**
 * Compute the scaling factors for a group of embeddings, given the neighbor distances computed by `compute_distance()`.
 * This aims to scale each embedding so that the within-population variances are equal across embeddings.
 * The "reference" modality is defined as the first embedding with a non-zero RMSD; 
 * other than this requirement, the exact choice of reference has no actual impact on the relative values of the scaling factors.
 *
 * @tparam Distance_ Floating-point type for the distances.
 *
 * @param distances Vector of distances for embeddings, as computed by `compute_distance()` on each embedding.
 *
 * @return Vector of scaling factors of length equal to that of `distances`, to be applied to each embedding.
 * This is equivalent to running `compute_scale()` on each entry of `distances` against the chosen reference.
 */
template<typename Distance_>
std::vector<Distance_> compute_scale(const std::vector<std::pair<Distance_, Distance_> >& distances) {
    std::vector<Distance_> output(distances.size());

    // Use the first entry with a non-zero RMSD as the reference.
    bool found_ref = false;
    auto ndist = distances.size();
    decltype(ndist) ref = 0;
    for (decltype(ndist) e = 0; e < ndist; ++e) {
        if (distances[e].second) {
            found_ref = true;
            ref = e;
            break;
        }
    }

    // If all of them have a zero RMSD, then all scalings are zero, because it doesn't matter.
    if (found_ref) {
        const auto& dref = distances[ref];
        for (decltype(ndist) e = 0; e < ndist; ++e) {
            output[e] = (e == ref ? 1 : compute_scale(dref, distances[e]));
        }
    }

    return output;
}

/**
 * Combine multiple embeddings for different modalities into a single embedding matrix, possibly after scaling each embedding.
 * This is done row-wise, i.e., the coordinates are concatenated across embeddings for each column.
 * 
 * @tparam Index_ Integer type for the number of cells.
 * @tparam Input_ Floating-point type for the input data.
 * @tparam Scale_ Floating-point type for the scaling factor.
 * @tparam Output_ Floating-point type for the output data.
 * 
 * @param num_dims Vector containing the number of dimensions in each embedding.
 * @param num_cells Number of cells in each embedding.
 * @param embeddings Vector of pointers of length equal to that of `num_dims`.
 * Each pointer refers to an array containing an embedding matrix for a single modality, which should be in column-major format with dimensions in rows and cells in columns.
 * The number of rows of the `i`-th matrix should be equal to `num_dims[i]` and the number of columns should be equal to `num_cells`.
 * @param scaling Scaling to apply to each embedding, usually from `compute_scale()`.
 * This should be of length equal to that of `num_dims`.
 * @param[out] output Pointer to the output array.
 * This should be of length equal to the product of `num_cells` and the sum of `num_dims`.
 * On completion, `output` is filled with the combined embeddings in column-major format.
 * Each row corresponds to a dimension while each column corresponds to a cell.
 */
template<typename Index_, typename Input_, typename Scale_, typename Output_>
void combine_scaled_embeddings(const std::vector<std::size_t>& num_dims, Index_ num_cells, const std::vector<Input_*>& embeddings, const std::vector<Scale_>& scaling, Output_* output) {
    auto nembed = num_dims.size();
    if (embeddings.size() != nembed || scaling.size() != nembed) {
        throw std::runtime_error("'num_dims', 'embeddings' and 'scale' should have the same length");
    }

    std::size_t ntotal = std::accumulate(num_dims.begin(), num_dims.end(), static_cast<std::size_t>(0));
    std::size_t offset = 0;

    for (decltype(nembed) e = 0; e < nembed; ++e) {
        auto curdim = num_dims[e];
        auto inptr = embeddings[e];
        auto s = scaling[e];

        // We use offsets to avoid forming invalid pointers with strided pointers.
        std::size_t in_position = 0;
        std::size_t out_position = offset;

        if (std::isinf(s)) {
            // If the scaling factor is infinite, it implies that the current
            // embedding is all-zero, so we just fill with zeros, and move on.
            for (Index_ c = 0; c < num_cells; ++c, in_position += curdim, out_position += ntotal) {
                std::fill_n(output + out_position, curdim, 0);
            }
        } else {
            for (Index_ c = 0; c < num_cells; ++c, in_position += curdim, out_position += ntotal) {
                for (std::size_t d = 0; d < curdim; ++d) {
                    output[out_position + d] = inptr[in_position + d] * s;
                }
            }
        }

        offset += curdim;
    }
}

}

#endif
