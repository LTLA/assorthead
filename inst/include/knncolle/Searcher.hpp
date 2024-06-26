#ifndef KNNCOLLE_SEARCHER_HPP
#define KNNCOLLE_SEARCHER_HPP

#include <vector>

/**
 * @file Searcher.hpp
 *
 * @brief Interface for searching nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface for searching nearest-neighbor search indices.
 *
 * Instances of `Searcher` subclasses are typically constructed with `Prebuilt::initialize()`.
 * This is intended to allow re-use of data allocations across different calls to `search()`.
 * Users should ensure that a `Searcher` instance does not outlive the `Prebuilt` object used to generate it;
 * this allows developers of the former to hold references to the latter.
 *
 * @tparam Index_ Integer type for the indices.
 * For the output of `Builder::build`, this is set to `MockMatrix::index_type`.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<typename Index_, typename Float_>
class Searcher {
public:
    /**
     * @cond
     */
    virtual ~Searcher() = default;
    /**
     * @endcond
     */

    /** 
     * Find the nearest neighbors of the `i`-th observation in the dataset.
     *
     * @param i The index of the observation of interest.
     * This should be non-negative and less than the total number of observations in `Prebuilt::num_observations()`.
     * @param k The number of neighbors to identify.
     * @param[out] output_indices Pointer to a vector, to be filled with the identities of the nearest neighbors in order of increasing distance.
     * Length of the vector on output is no more than `k`, but may be shorter if the total number of observations is less than `k + 1`.
     * This vector is guaranteed to not contain `i` itself.
     * Optionally NULL, in which case no indices are returned.
     * @param[out] output_indices Pointer to a vector, to be filled with the distances of the nearest neighbors. 
     * This corresponds to the indices reported in `output_indices`.
     * Optionally NULL, in which case no distances are returned.
     */
    virtual void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `Prebuilt::num_dimensions()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     * @param[out] output_indices Pointer to a vector, to be filled with the identities of the nearest neighbors in order of increasing distance.
     * Length of the vector on output is no more than `k`, but may be shorter if the total number of observations is less than `k + 1`.
     * Optionally NULL, in which case no indices are returned.
     * @param[out] output_indices Pointer to a vector, to be filled with the distances of the nearest neighbors. 
     * This corresponds to the indices reported in `output_indices`.
     * Optionally NULL, in which case no distances are returned.
     */
    virtual void search(const Float_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) = 0;
};

}

#endif
