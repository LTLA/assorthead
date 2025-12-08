#ifndef SCRAN_MARKERS_COHENS_D_HPP
#define SCRAN_MARKERS_COHENS_D_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <type_traits>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"
#include "scran_blocks/scran_blocks.hpp"

#include "block_averages.hpp"
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
template<typename Stat_>
std::pair<Stat_, Stat_> compute_cohens_d_blockmean(
    const std::size_t g1,
    const std::size_t g2,
    const Stat_* const means,
    const Stat_* const vars,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const Stat_ threshold,
    const PrecomputedPairwiseWeights<Stat_>& preweights 
) {
    const auto winfo = preweights.get(g1, g2);
    constexpr auto nan = std::numeric_limits<Stat_>::quiet_NaN();
    if (winfo.second == 0) {
        return std::make_pair(nan, nan);
    }

    std::pair<Stat_, Stat_> output(0, 0);
    Stat_ total_weight = 0; // need to calculate it dynamically, in case there are NaN variances for non-zero weights (e.g., one observation per block).
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto weight = winfo.first[b];
        if (weight == 0) {
            continue;
        }

        const auto offset1 = sanisizer::nd_offset<std::size_t>(g1, ngroups, b); // remember, 'groups' is the faster-changing dimension.
        const auto offset2 = sanisizer::nd_offset<std::size_t>(g2, ngroups, b);
        const auto left_var = vars[offset1];
        const auto right_var = vars[offset2];
        const Stat_ denom = cohen_denominator(left_var, right_var);
        if (std::isnan(denom)) {
            continue;
        }

        const auto left_mean = means[offset1];
        const auto right_mean = means[offset2]; 
        output.first += compute_cohens_d(left_mean, right_mean, denom, threshold) * weight;
        if (threshold) {
            output.second += compute_cohens_d(right_mean, left_mean, denom, threshold) * weight;
        }
        total_weight += weight;
    }

    if (total_weight) {
        output.first /= total_weight;
        if (threshold) {
            output.second /= total_weight;
        } else {
            output.second = -output.first;
        }
    } else {
        output.first = nan;
        output.second = nan;
    }

    return output;
}

template<typename Stat_>
void compute_pairwise_cohens_d_blockmean(
    const Stat_* const means,
    const Stat_* const vars,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const Stat_ threshold,
    const PrecomputedPairwiseWeights<Stat_>& preweights,
    Stat_* const output)
{
    for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
        for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
            const auto tmp = compute_cohens_d_blockmean(g1, g2, means, vars, ngroups, nblocks, threshold, preweights);
            output[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)] = tmp.first;
            output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = tmp.second;
        }
        output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g1)] = 0; // zero the diagonals for consistency.
    }
}

template<typename Stat_>
std::pair<Stat_, Stat_> compute_cohens_d_blockquantile(
    const std::size_t g1,
    const std::size_t g2,
    const Stat_* const means,
    const Stat_* const vars,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const Stat_ threshold,
    std::vector<Stat_>& buffer,
    std::vector<Stat_>& rev_buffer,
    scran_blocks::SingleQuantileVariable<Stat_, typename std::vector<Stat_>::iterator>& qcalc
) {
    buffer.clear();
    rev_buffer.clear();

    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto offset1 = sanisizer::nd_offset<std::size_t>(g1, ngroups, b); // remember, 'groups' is the faster-changing dimension.
        const auto offset2 = sanisizer::nd_offset<std::size_t>(g2, ngroups, b);
        const auto left_var = vars[offset1];
        const auto right_var = vars[offset2];
        const Stat_ denom = cohen_denominator(left_var, right_var);
        if (std::isnan(denom)) {
            continue;
        }

        const auto left_mean = means[offset1];
        const auto right_mean = means[offset2]; 
        const auto effect = compute_cohens_d(left_mean, right_mean, denom, threshold);
        if (std::isnan(effect)) {
            continue;
        }

        buffer.push_back(effect);
        if (threshold) {
            rev_buffer.push_back(compute_cohens_d(right_mean, left_mean, denom, threshold));
        }
    }

    std::pair<Stat_, Stat_> output;
    output.first = qcalc(buffer.size(), buffer.begin(), buffer.end());
    if (threshold) {
        output.second = qcalc(rev_buffer.size(), rev_buffer.begin(), rev_buffer.end());
    } else {
        for (auto& x : buffer) {
            x *= -1;
        }
        output.second = qcalc(buffer.size(), buffer.begin(), buffer.end());
    }

    return output;
}

template<typename Stat_>
void compute_pairwise_cohens_d_blockquantile(
    const Stat_* const means,
    const Stat_* const vars,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const Stat_ threshold,
    std::vector<Stat_>& buffer,
    std::vector<Stat_>& rev_buffer,
    scran_blocks::SingleQuantileVariable<Stat_, typename std::vector<Stat_>::iterator>& qcalc,
    Stat_* const output
) {
    for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
        for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
            const auto tmp = compute_cohens_d_blockquantile(g1, g2, means, vars, ngroups, nblocks, threshold, buffer, rev_buffer, qcalc);
            output[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)] = tmp.first;
            output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = tmp.second;
        }
        output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g1)] = 0; // zero the diagonals for consistency.
    }
}

}

}

#endif
