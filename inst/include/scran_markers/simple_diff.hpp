#ifndef SCRAN_MARKERS_SIMPLE_DIFF_HPP
#define SCRAN_MARKERS_SIMPLE_DIFF_HPP

#include <limits>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"
#include "scran_blocks/scran_blocks.hpp"

#include "block_averages.hpp"
#include "utils.hpp"

namespace scran_markers {

namespace internal {

// 'values' is expected to be an 'ngroups * nblocks' array where groups are the
// faster-changing dimension and the blocks are slower.
template<typename Stat_>
Stat_ compute_simple_diff_blockmean(
    const std::size_t g1,
    const std::size_t g2,
    const Stat_* const values,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const PrecomputedPairwiseWeights<Stat_>& preweights
) {
    const auto winfo = preweights.get(g1, g2);
    if (winfo.second == 0) {
        return std::numeric_limits<Stat_>::quiet_NaN();
    }

    Stat_ output = 0;
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto weight = winfo.first[b];
        if (weight == 0) {
            continue;
        }

        const auto left = values[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)]; 
        const auto right = values[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
        output += (left - right) * weight;
    }

    return output / winfo.second;
}

template<typename Stat_>
void compute_pairwise_simple_diff_blockmean(
    const Stat_* const values,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const PrecomputedPairwiseWeights<Stat_>& preweights,
    Stat_* const output
) {
    for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
        for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
            const auto d = compute_simple_diff_blockmean(g1, g2, values, ngroups, nblocks, preweights);
            output[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)] = d;
            output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = -d;
        }
        output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g1)] = 0; // zero the diagonals for consistency.
    }
}

template<typename Stat_>
std::pair<Stat_, Stat_> compute_simple_diff_blockquantile(
    const std::size_t g1,
    const std::size_t g2,
    const Stat_* const values,
    const std::size_t ngroups,
    const std::size_t nblocks,
    std::vector<Stat_>& buffer,
    scran_blocks::SingleQuantileVariable<Stat_, typename std::vector<Stat_>::iterator>& qcalc
) {
    buffer.clear();
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto left = values[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)]; 
        const auto right = values[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
        const auto effect = left - right;
        if (!std::isnan(effect)) {
            buffer.push_back(left - right);
        }
    }

    std::pair<Stat_, Stat_> output;
    output.first = qcalc(buffer.size(), buffer.begin(), buffer.end());
    for (auto& x : buffer) {
        x *= -1;
    }
    output.second = qcalc(buffer.size(), buffer.begin(), buffer.end());

    return output;
}

template<typename Stat_>
void compute_pairwise_simple_diff_blockquantile(
    const Stat_* const values,
    const std::size_t ngroups,
    const std::size_t nblocks,
    std::vector<Stat_>& buffer,
    scran_blocks::SingleQuantileVariable<Stat_, typename std::vector<Stat_>::iterator>& qcalc,
    Stat_* const output
) {
    for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
        for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
            const auto d = compute_simple_diff_blockquantile(g1, g2, values, ngroups, nblocks, buffer, qcalc);
            output[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)] = d.first;
            output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = d.second;
        }
        output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g1)] = 0; // zero the diagonals for consistency.
    }
}

}

}

#endif
