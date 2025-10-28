#ifndef MUMOSA_COMPUTE_SCALE_HPP
#define MUMOSA_COMPUTE_SCALE_HPP

#include <limits>
#include <utility>
#include <vector>

#include "sanisizer/sanisizer.hpp"

/**
 * @file compute_scale.hpp
 * @brief Compute scaling factors for embeddings.
 */

namespace mumosa {

/**
 * Compute the scaling factor to be applied to an embedding of a "target" modality relative to a "reference" modality.
 * The aim is to scale the target so that the average variance in the local neighborhood is equal to that of the reference,
 * to ensure that high noise in one modality does not drown out interesting biology in another modality in downstream analyses.
 *
 * This approach assumes that the median distance to the `Options::num_neighbors`-th nearest neighbor is proportional to the neighborhood variance.
 * The scaling factor is defined as the ratio of the median distances in the reference to the target.
 * If either of the median distances is zero, this function instead returns the ratio of the RMSDs as a fallback.
 *
 * Advanced users may want to scale the target so that its variance is some \f$S\f$-fold of the reference, e.g., to give more weight to more important modalities.
 * This can be achieved by multiplying the returned factor by \f$\sqrt{S}\f$ prior to the actual scaling.
 *
 * @tparam Distance_ Floating-point type of the distances.
 *
 * @param ref Results of `compute_distance()` for the embedding of the reference modality.
 * The first value contains the median distance while the second value contains the root-mean squared distance (RMSD).
 * @param target Results of `compute_distance()` for the embedding of the target modality.
 *
 * @return A scaling factor to multiply the embedding coordinates of the target modality.
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
 * This aims to scale each embedding so that the neighborhood variances are equal across embeddings as described in `compute_scale()`.
 * The "reference" modality is defined as the first embedding with a non-zero RMSD to ensure that the scaling is well-defined for every sample; 
 * other than this requirement, the exact choice of reference has no actual impact on the relative values of the scaling factors.
 *
 * @tparam Distance_ Floating-point type of the distances.
 *
 * @param distances Vector of distances for embeddings, as computed by `compute_distance()` on each embedding.
 *
 * @return Vector of scaling factors of length equal to that of `distances`, to be applied to each embedding.
 * This is equivalent to running `compute_scale()` on each entry of `distances` against the chosen reference.
 */
template<typename Distance_>
std::vector<Distance_> compute_scale(const std::vector<std::pair<Distance_, Distance_> >& distances) {
    const auto ndist = distances.size();
    auto output = sanisizer::create<std::vector<Distance_> >(ndist);

    // Use the first entry with a non-zero RMSD as the reference.
    bool found_ref = false;
    I<decltype(ndist)> ref = 0;
    for (I<decltype(ndist)> e = 0; e < ndist; ++e) {
        if (distances[e].second) {
            found_ref = true;
            ref = e;
            break;
        }
    }

    // If all of them have a zero RMSD, then all scalings are zero, because it doesn't matter.
    if (found_ref) {
        const auto& dref = distances[ref];
        for (I<decltype(ndist)> e = 0; e < ndist; ++e) {
            output[e] = (e == ref ? static_cast<Distance_>(1) : compute_scale(dref, distances[e]));
        }
    }

    return output;
}

}

#endif
