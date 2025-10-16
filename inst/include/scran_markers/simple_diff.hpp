#ifndef SCRAN_MARKERS_SIMPLE_DIFF_HPP
#define SCRAN_MARKERS_SIMPLE_DIFF_HPP

#include <limits>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "PrecomputedPairwiseWeights.hpp"
#include "utils.hpp"

namespace scran_markers {

namespace internal {

// 'values' is expected to be an 'ngroups * nblocks' array where groups are the
// faster-changing dimension and the blocks are slower.
template<typename Stat_, typename Weight_>
Stat_ compute_pairwise_simple_diff(
    const std::size_t g1,
    const std::size_t g2,
    const Stat_* const values,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights)
{
    const auto winfo = preweights.get(g1, g2);
    const auto total_weight = winfo.second;
    if (total_weight == 0) {
        return std::numeric_limits<Stat_>::quiet_NaN();
    }

    Stat_ output = 0;
    for (decltype(I(nblocks)) b = 0; b < nblocks; ++b) {
        const auto weight = winfo.first[b];
        if (weight) {
            const auto left = values[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)]; 
            const auto right = values[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
            output += (left - right) * weight;
        }
    }

    return output / total_weight;
}

template<typename Stat_, typename Weight_>
void compute_pairwise_simple_diff(
    const Stat_* const values,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const PrecomputedPairwiseWeights<Weight_>& preweights, Stat_* output)
{
    for (decltype(I(ngroups)) g1 = 0; g1 < ngroups; ++g1) {
        for (decltype(I(g1)) g2 = 0; g2 < g1; ++g2) {
            const auto d = compute_pairwise_simple_diff(g1, g2, values, ngroups, nblocks, preweights);
            output[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)] = d;
            output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = -d;
        }
        output[sanisizer::nd_offset<std::size_t>(g1, ngroups, g1)] = 0; // zero the diagonals for consistency.
    }
}

}

}

#endif
