#ifndef SCRAN_MARKERS_COHENS_D_HPP
#define SCRAN_MARKERS_COHENS_D_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <type_traits>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "PrecomputedPairwiseWeights.hpp"
#include "utils.hpp"

namespace scran_markers {

namespace internal {

template<typename Stat_>
Stat_ compute_cohens_d(const Stat_ m1, const Stat_ m2, const Stat_ sd, const Stat_ threshold) {
    if (std::isnan(sd)) {
        return std::numeric_limits<Stat_>::quiet_NaN();
    } 
    
    const Stat_ delta = m1 - m2 - threshold;
    if (sd == 0 && delta == 0) {
        return 0;
    } else if (sd == 0) {
        if (delta > 0) {
            return std::numeric_limits<Stat_>::infinity();
        } else {
            return -std::numeric_limits<Stat_>::infinity();
        }
    } else {
        return delta / sd;
    }
}

template<typename Stat_>
Stat_ cohen_denominator(const Stat_ left_var, const Stat_ right_var) {
    if (std::isnan(left_var) && std::isnan(right_var)) {
        return std::numeric_limits<Stat_>::quiet_NaN();
    } else if (std::isnan(left_var)) {
        return std::sqrt(right_var);
    } else if (std::isnan(right_var)) {
        return std::sqrt(left_var);
    } else {
        // Technically, we should use the pooled variance, but this introduces some unintuitive asymmetry in the behavior of the groups.
        // You wouldn't get the same (expected) Cohen's d when you change the sizes of the groups with different variances.
        // For example, if the larger group has low variance (e.g., because it's all zero), the variance of the smaller group is effectively ignored,
        // unfairly favoring genes with highly variable expression in the smaller group. 
        // So we take a simple average instead.
        return std::sqrt(left_var + (right_var - left_var)/2); // reduce risk of overflow.
    }
}

// 'means' and 'vars' are expected to be 'ngroups * nblocks' arrays
// where groups are the faster-changing dimension and the blocks are slower.
template<typename Stat_, typename Weight_>
std::pair<Stat_, Stat_> compute_pairwise_cohens_d_two_sided(
    const std::size_t g1,
    const std::size_t g2,
    const Stat_* const means,
    const Stat_* const vars,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights,
    const Stat_ threshold
) {
    std::pair<Stat_, Stat_> output(0, 0);

    const auto winfo = preweights.get(g1, g2);
    auto total_weight = winfo.second;
    if (total_weight != 0) {
        total_weight = 0; // need to calculate it more dynamically, in case there are NaN variances.

        for (decltype(I(nblocks)) b = 0; b < nblocks; ++b) {
            const auto weight = winfo.first[b];
            if (weight) {
                const auto offset1 = sanisizer::nd_offset<std::size_t>(g1, ngroups, b); // remember, 'groups' is the faster-changing dimension.
                const auto offset2 = sanisizer::nd_offset<std::size_t>(g2, ngroups, b);
                const auto left_var = vars[offset1];
                const auto right_var = vars[offset2];
                const Stat_ denom = cohen_denominator(left_var, right_var);

                if (!std::isnan(denom)) {
                    total_weight += weight;
                    const auto left_mean = means[offset1];
                    const auto right_mean = means[offset2]; 
                    const Stat_ extra = compute_cohens_d(left_mean, right_mean, denom, threshold) * weight;

                    output.first += extra;
                    if (threshold) {
                        output.second += compute_cohens_d(right_mean, left_mean, denom, threshold) * weight;
                    }
                }
            }
        }
    }

    if (total_weight) {
        output.first /= total_weight;
        if (threshold) {
            output.second /= total_weight;
        } else {
            output.second = -output.first;
        }
    } else {
        output.first = std::numeric_limits<Stat_>::quiet_NaN();
        output.second = std::numeric_limits<Stat_>::quiet_NaN();
    }

    return output;
}

template<typename Stat_, typename Weight_>
void compute_pairwise_cohens_d(
    const Stat_* const means,
    const Stat_* const vars,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights,
    const Stat_ threshold,
    Stat_* const output)
{
    for (decltype(I(ngroups)) g1 = 0; g1 < ngroups; ++g1) {
        for (decltype(I(g1)) g2 = 0; g2 < g1; ++g2) {
            auto tmp = compute_pairwise_cohens_d_two_sided(g1, g2, means, vars, ngroups, nblocks, preweights, threshold);
            output[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)] = tmp.first;
            output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = tmp.second;
        }
        output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g1)] = 0; // zero the diagonals for consistency.
    }
}

}

}

#endif
