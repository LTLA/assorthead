#ifndef SCRAN_MARKERS_COHENS_D_HPP
#define SCRAN_MARKERS_COHENS_D_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <type_traits>
#include <cstddef>

#include "PrecomputedPairwiseWeights.hpp"

#include "sanisizer/sanisizer.hpp"

namespace scran_markers {

namespace internal {

template<typename Stat_>
Stat_ compute_cohens_d(Stat_ m1, Stat_ m2, Stat_ sd, Stat_ threshold) {
    if (std::isnan(sd)) {
        return std::numeric_limits<Stat_>::quiet_NaN();
    } 
    
    Stat_ delta = m1 - m2 - threshold;
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
Stat_ cohen_denominator(Stat_ left_var, Stat_ right_var) {
    if (std::isnan(left_var) && std::isnan(right_var)) {
        return std::numeric_limits<Stat_>::quiet_NaN();
    } else if (std::isnan(left_var)) {
        return std::sqrt(right_var);
    } else if (std::isnan(right_var)) {
        return std::sqrt(left_var);
    } else {
        return std::sqrt((left_var + right_var)/2);
    }
}

// 'means' and 'vars' are expected to be 'ngroups * nblocks' arrays
// where groups are the faster-changing dimension and the blocks are slower.
template<typename Stat_, typename Weight_, class Output_>
void compute_pairwise_cohens_d_internal(
    std::size_t g1,
    std::size_t g2,
    const Stat_* means,
    const Stat_* vars,
    std::size_t ngroups,
    std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights,
    Stat_ threshold,
    Output_& output)
{
    constexpr bool do_both_sides = !std::is_same<Stat_, Output_>::value;

    auto winfo = preweights.get(g1, g2);
    auto total_weight = winfo.second;
    if (total_weight != 0) {
        total_weight = 0; // need to calculate it more dynamically if there are NaN variances.

        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            auto weight = winfo.first[b];
            if (weight) {
                auto offset1 = sanisizer::nd_offset<std::size_t>(g1, ngroups, b); // remember, 'groups' is the faster-changing dimension.
                auto offset2 = sanisizer::nd_offset<std::size_t>(g2, ngroups, b);
                auto left_var = vars[offset1];
                auto right_var = vars[offset2];
                Stat_ denom = cohen_denominator(left_var, right_var);

                if (!std::isnan(denom)) {
                    total_weight += weight;
                    auto left_mean = means[offset1];
                    auto right_mean = means[offset2]; 
                    Stat_ extra = compute_cohens_d(left_mean, right_mean, denom, threshold) * weight;

                    if constexpr(do_both_sides) {
                        output.first += extra;
                        if (threshold) {
                            output.second += compute_cohens_d(right_mean, left_mean, denom, threshold) * weight;
                        }
                    } else {
                        output += extra;
                    }
                }
            }
        }
    }

    if constexpr(do_both_sides) {
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
    } else {
        if (total_weight) {
            output /= total_weight;
        } else {
            output = std::numeric_limits<Stat_>::quiet_NaN();
        }
        return;
    }
}

template<typename Stat_, typename Weight_>
Stat_ compute_pairwise_cohens_d_one_sided(
    std::size_t g1,
    std::size_t g2,
    const Stat_* means,
    const Stat_* vars,
    std::size_t ngroups,
    std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights,
    Stat_ threshold)
{
    Stat_ output = 0;
    compute_pairwise_cohens_d_internal(g1, g2, means, vars, ngroups, nblocks, preweights, threshold, output);
    return output;
}

template<typename Stat_, typename Weight_>
std::pair<Stat_, Stat_> compute_pairwise_cohens_d_two_sided(
    std::size_t g1,
    std::size_t g2,
    const Stat_* means,
    const Stat_* vars,
    std::size_t ngroups,
    std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights,
    Stat_ threshold)
{
    std::pair<Stat_, Stat_> output(0, 0);
    compute_pairwise_cohens_d_internal(g1, g2, means, vars, ngroups, nblocks, preweights, threshold, output);
    return output;
}

template<typename Stat_, typename Weight_>
void compute_pairwise_cohens_d(
    const Stat_* means,
    const Stat_* vars,
    std::size_t ngroups,
    std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights,
    Stat_ threshold,
    Stat_* output)
{
    for (decltype(ngroups) g1 = 0; g1 < ngroups; ++g1) {
        for (decltype(g1) g2 = 0; g2 < g1; ++g2) {
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
