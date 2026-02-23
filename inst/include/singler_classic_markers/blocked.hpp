#ifndef MEDIAN_MARKERS_BLOCKED_HPP
#define MEDIAN_MARKERS_BLOCKED_HPP

#include <cstddef>
#include <optional>
#include <vector>
#include <limits>
#include <cmath>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

#include "queue.hpp"
#include "scan.hpp"
#include "number.hpp"

namespace singler_classic_markers {

/**
 * @brief Options for `choose_blocked()`.
 */
struct ChooseBlockedOptions {
    /**
     * Number of top genes to use as the marker set in each pairwise comparison.
     * If not set, this is automatically determined from the number of labels, see `default_markers()`.
     */
    std::optional<std::size_t> number;

    /**
     * Whether to report ties at the `number`-th gene for each pairwise comparisons.
     * If `true`, all genes with tied differences are reported in the output.
     * Otherwise, no more than `number` genes will be reported, with ties broken by row index (i.e., earlier genes are preferentially retained).
     */
    bool keep_ties = false;

    /**
     * Whether to report the minimum difference across blocks.
     * This focuses on genes that exhibit consistent differences in the same direction in every block.
     * If `false`, the mean difference across blocks is reported instead, which is more lenient to variations in the per-block differences. 
     */
    bool use_minimum = false;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<bool include_stat_, typename Stat_, typename Value_, typename Index_, typename Label_, typename Block_>
Markers<include_stat_, Index_, Stat_> choose_blocked_raw(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Label_* label,
    const Block_* block,
    const ChooseBlockedOptions& options
) {
    const auto NC = matrix.ncol();
    const std::size_t ngroups = tatami_stats::total_groups/*<std::size_t>*/(label, NC);
    const std::size_t nblocks = tatami_stats::total_groups/*<std::size_t>*/(block, NC);

    const auto num_keep = get_num_keep<Index_>(ngroups, options.number);
    if (num_keep == 0 || NC == 0 || matrix.nrow() == 0) {
        return report_empty_markers<include_stat_, Index_, Stat_>(ngroups);
    }

    auto pqueues = sanisizer::create<std::vector<PairwiseTopQueues<Stat_, Index_> > >(options.num_threads);

    // Creating the combinations between block and not.
    const auto ncombos = sanisizer::product<std::size_t>(ngroups, nblocks); // check that all producs below are safe.
    auto combinations = sanisizer::create<std::vector<std::size_t> >(NC);
    for (I<decltype(NC)> c = 0; c < NC; ++c) {
        combinations[c] = sanisizer::nd_offset<std::size_t>(label[c], ngroups, block[c]); // group is the faster changing dimension.
    }
    auto combo_sizes = tatami_stats::tabulate_groups(combinations.data(), NC);

    scan_matrix<Stat_>(
        matrix,
        sanisizer::cast<std::size_t>(ncombos),
        combinations.data(),
        combo_sizes,

        /* setup = */ [&](const int t) -> bool {
            allocate_pairwise_queues(pqueues[t], num_keep, ngroups, options.keep_ties, /* check_nan = */ false); // we'll check it ourselves.
            return false;
        },

        /* fun = */ [&](const int t, const Index_ r, const std::vector<Stat_>& medians, bool) -> void {
            auto& curqueues = pqueues[t];
            for (I<decltype(ngroups)> g1 = 1; g1 < ngroups; ++g1) {
                for (I<decltype(ngroups)> g2 = 0; g2 < g1; ++g2) {

                    if (options.use_minimum) {
                        Stat_ xval = std::numeric_limits<Stat_>::infinity();
                        Stat_ yval = std::numeric_limits<Stat_>::infinity();
                        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                            const auto delta = medians[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)] - medians[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
                            if (!std::isnan(delta)) {
                                xval = std::min(xval, delta);
                                yval = std::min(yval, -delta);
                            }
                        }

                        if (std::isfinite(xval)) {
                            curqueues[g1][g2].emplace(xval, r); 
                            curqueues[g2][g1].emplace(yval, r); 
                        }

                    } else {
                        Stat_ val = 0;
                        std::size_t denom = 0;
                        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                            const auto delta = medians[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)] - medians[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
                            if (!std::isnan(delta)) {
                                ++denom;
                                val += delta; 
                            }
                        }

                        if (denom) {
                            val /= denom;
                            curqueues[g1][g2].emplace(val, r); 
                            curqueues[g2][g1].emplace(-val, r); 
                        }
                    }

                }
            }
        },

        options.num_threads
    );

    Markers<include_stat_, Index_, Stat_> output;
    report_best_top_queues<include_stat_>(pqueues, ngroups, output);
    return output;
}
/**
 * @endcond
 */

/**
 * Variant of `choose()` that handles multiple blocks (e.g., batch effects) in the reference dataset.
 * Differences between medians are computed within each block and then combined across blocks to obtain a single statistic per gene in each pairwise comparison.
 * The default method is to compute the mean of the per-block differences, but we can also compute the minimum for greater stringency.
 * 
 * @tparam Stat_ Floating-point type of the differences between medians.
 * @tparam Value_ Numeric type of matrix values.
 * @tparam Index_ Integer type of matrix row/column indices.
 * @tparam Label_ Integer type of the label identity.
 * @tparam Block_ Integer type of the block assignment.
 *
 * @param matrix Matrix containing a reference dataset.
 * Each column should correspond to a sample while each row should represent a gene.
 * @param label Pointer to an array of length equal to the number of columns in `matrix`.
 * Each value of the array should specify the label for the corresponding column. 
 * Values should lie in \f$[0, L)\f$ for \f$L\f$ unique labels. 
 * @param block Pointer to an array of length equal to the number of columns in `matrix`.
 * Each value of the array should specify the block for the corresponding column. 
 * Values should lie in \f$[0, B)\f$ for \f$B\f$ unique blocks. 
 * @param options Further options.
 * 
 * @return Top markers for each pairwise comparison between labels.
 * This is equivalent in structure to the return value of `choose()`, 
 * except that the combined difference between medians is reported for each marker.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Label_, typename Block_>
std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > > choose_blocked(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Label_* label,
    const Block_* block,
    const ChooseBlockedOptions& options
) {
    return choose_blocked_raw<true, Stat_>(matrix, label, block, options);
}

/**
 * Variant of `choose_blocked()` that only reports the indices of the top markers for each pairwise comparison.
 * This can be used directly in **singlepp** functions.
 *
 * @tparam Stat_ Floating-point type of the differences between medians.
 * @tparam Value_ Numeric type of matrix values.
 * @tparam Index_ Integer type of matrix row/column indices.
 * @tparam Label_ Integer type of the label identity.
 *
 * @param matrix Matrix containing a reference dataset.
 * Each column should correspond to a sample while each row should represent a gene.
 * @param label Pointer to an array of length equal to the number of columns in `matrix`.
 * Each value of the array should specify the label for the corresponding column. 
 * Values should lie in \f$[0, L)\f$ for \f$L\f$ unique labels. 
 * @param block Pointer to an array of length equal to the number of columns in `matrix`.
 * Each value of the array should specify the block for the corresponding column. 
 * Values should lie in \f$[0, B)\f$ for \f$B\f$ unique blocks. 
 * @param options Further options.
 * 
 * @return Top markers for each pairwise comparison between labels.
 * This is the same as the output for `choose_blocked()` except that only the row index is reported in the innermost vector.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Label_, typename Block_>
std::vector<std::vector<std::vector<Index_> > > choose_blocked_index(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Label_* label,
    const Block_* block,
    const ChooseBlockedOptions& options
) {
    return choose_blocked_raw<false, Stat_>(matrix, label, block, options);
}

}

#endif
