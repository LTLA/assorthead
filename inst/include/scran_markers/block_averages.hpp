#ifndef SCRAN_MARKERS_BLOCK_AVERAGES_HPP
#define SCRAN_MARKERS_BLOCK_AVERAGES_HPP

#include <vector>
#include <cstddef>
#include <optional>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file block_averages.hpp
 * @brief Averaging statistics over blocks.
 */

namespace scran_markers {

/**
 * Policy for averaging statistics across blocks.
 *
 * - `MEAN`: computes a weighted mean of per-block statistics.
 *   Weights are based on the size of the block.
 * - `QUANTILE`: computes a quantile of the per-block statistics.
 *   This can be used to enforce a minimum effect size across some percentage of blocks.
 */
enum class BlockAveragePolicy : unsigned char { MEAN, QUANTILE };

/**
 * @cond
 */
namespace internal {

template<typename Stat_>
class PrecomputedPairwiseWeights {
public:
    // 'combo_weights' are expected to be 'ngroups * nblocks' arrays where
    // groups are the faster-changing dimension and the blocks are slower.
    PrecomputedPairwiseWeights(const std::size_t ngroups, const std::size_t nblocks, const Stat_* const combo_weights) :
        my_total(sanisizer::product<I<decltype(my_total.size())> >(ngroups, ngroups)),
        my_by_block(sanisizer::product<I<decltype(my_by_block.size())> >(my_total.size(), nblocks)),
        my_ngroups(ngroups),
        my_nblocks(nblocks)
    {
        auto blocks_in_use = sanisizer::create<std::vector<std::size_t> >(my_total.size());
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            for (I<decltype(ngroups)> g1 = 1; g1 < ngroups; ++g1) {
                const auto w1 = combo_weights[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)];
                if (w1 == 0) {
                    continue;
                }

                for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
                    const auto w2 = combo_weights[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
                    if (w2 == 0) {
                        continue;
                    }

                    // Storing it as a 3D array where the blocks are the fastest changing, 
                    // and then the two groups are the next fastest changing.
                    const Stat_ combined = w1 * w2;
                    const auto out_offset1 = sanisizer::nd_offset<std::size_t>(g2, ngroups, g1);
                    my_by_block[sanisizer::nd_offset<std::size_t>(b, nblocks, out_offset1)] = combined;
                    my_by_block[sanisizer::nd_offset<std::size_t>(b, nblocks, g1, ngroups, g2)] = combined;
                    my_total[out_offset1] += combined;
                    blocks_in_use[out_offset1] += (combined > 0);
                }
            }
        }

        // If we have exactly one block that contributes to the weighted mean, the magnitude of the weight doesn't matter.
        // So, we set the weight to 1 to ensure that the weighted mean calculation is a no-op,
        // i.e., there won't be any introduction of floating-point errors from a mult/div by the weight. 
        // Zero weights do need to be preserved, though, as mult/div by zero gives NaN.
        for (I<decltype(ngroups)> g1 = 1; g1 < ngroups; ++g1) {
            for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
                const auto out_offset1 = sanisizer::nd_offset<std::size_t>(g2, ngroups, g1);
                if (blocks_in_use[out_offset1] != 1) {
                    continue;
                }

                my_total[out_offset1] = 1;
                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    auto& curweight = my_by_block[sanisizer::nd_offset<std::size_t>(b, nblocks, out_offset1)];
                    curweight = (curweight > 0);
                    my_by_block[sanisizer::nd_offset<std::size_t>(b, nblocks, g1, ngroups, g2)] = curweight;
                }
            }
        }

        // Filling the other side of my_totals, for completeness.
        for (I<decltype(ngroups)> g1 = 1; g1 < ngroups; ++g1) {
            for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
                my_total[sanisizer::nd_offset<std::size_t>(g1, ngroups, g2)] = my_total[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)];
            }
        }
    }

public:
    std::pair<const Stat_*, Stat_> get(const std::size_t g1, const std::size_t g2) const {
        const auto offset = sanisizer::nd_offset<std::size_t>(g2, my_ngroups, g1);
        return std::make_pair(
            my_by_block.data() + offset * my_nblocks,
            my_total[offset]
        );
    }

private:
    std::vector<Stat_> my_total;
    std::vector<Stat_> my_by_block;
    std::size_t my_ngroups, my_nblocks;
};

template<typename Stat_>
class BlockAverageInfo {
public:
    BlockAverageInfo() = default;
    BlockAverageInfo(std::vector<Stat_> combo_weights) : my_combo_weights(std::move(combo_weights)) {}
    BlockAverageInfo(const double quantile) : my_quantile(quantile) {}

private:
    std::optional<std::vector<Stat_> > my_combo_weights;
    double my_quantile = 0;

public:
    bool use_mean() const {
        return my_combo_weights.has_value();
    }

    const std::vector<Stat_>& combo_weights() const {
        return *my_combo_weights;
    }

    double quantile() const {
        return my_quantile;
    }
};

}
/**
 * @endcond
 */

}

#endif
