#ifndef SINGLER_CLASSIC_MARKERS_CHOOSE_HPP
#define SINGLER_CLASSIC_MARKERS_CHOOSE_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

#include "queue.hpp"
#include "scan.hpp"
#include "number.hpp"

/**
 * @file choose.hpp
 * @brief Choose markers with the classic **SingleR** algorithm.
 */

namespace singler_classic_markers {

/**
 * @brief Options for `choose()`.
 */
struct ChooseOptions {
    /**
     * Number of top genes to use as the marker set in each pairwise comparison.
     * If not set, this is automatically determined from the number of labels, see `default_number()`.
     */
    std::optional<std::size_t> number;

    /**
     * Whether to report ties at the `number`-th gene for each pairwise comparisons.
     * If `true`, all genes with tied differences are reported in the output.
     * Otherwise, no more than `number` genes will be reported, with ties broken by row index (i.e., earlier genes are preferentially retained).
     */
    bool keep_ties = false;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<bool include_stat_, typename Stat_, typename Value_, typename Index_, typename Label_>
Markers<include_stat_, Index_, Stat_> choose_raw(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Label_* label,
    const ChooseOptions& options
) {
    const auto NC = matrix.ncol();
    auto group_sizes = tatami_stats::tabulate_groups(label, NC);
    const auto ngroups = group_sizes.size();

    const auto num_keep = get_num_keep<Index_>(ngroups, options.number);
    auto pqueues = sanisizer::create<std::vector<std::optional<PairwiseTopQueues<Stat_, Index_> > > >(options.num_threads);

    const auto num_used = scan_matrix<Stat_>(
        matrix,
        sanisizer::cast<std::size_t>(ngroups),
        label,
        group_sizes,

        /* setup = */ [&]() -> PairwiseTopQueues<Stat_, Index_> {
            PairwiseTopQueues<Stat_, Index_> output;
            allocate_pairwise_queues(output, num_keep, ngroups, options.keep_ties, /* check_nan = */ true);
            return output;
        },

        /* fun = */ [&](const Index_ r, const std::vector<Stat_>& medians, PairwiseTopQueues<Stat_, Index_>& curqueues) -> void {
            for (I<decltype(ngroups)> g1 = 1; g1 < ngroups; ++g1) {
                for (I<decltype(ngroups)> g2 = 0; g2 < g1; ++g2) {
                    const auto delta = medians[g1] - medians[g2];
                    curqueues[g1][g2].emplace(delta, r); 
                    curqueues[g2][g1].emplace(-delta, r); 
                }
            }
        },

        /* finalize = */ [&](const int t, PairwiseTopQueues<Stat_, Index_>& curqueues) -> void {
            pqueues[t] = std::move(curqueues);
        },

        options.num_threads
    );

    pqueues.resize(num_used); 
    Markers<include_stat_, Index_, Stat_> output;
    report_best_top_queues<include_stat_>(pqueues, ngroups, output);
    return output;
}
/**
 * @endcond
 */

/**
 * Implements the classic **SingleR** method for choosing markers from (typically bulk) reference datasets.
 * We assume that we have a matrix of representative expression profiles for each label, typically computed by averaging across all reference profiles for that label.
 * For the comparison between labels \f$A\f$ and \f$B\f$, we define the marker set as the top genes with the largest positive differences in \f$A\f$'s profile over \f$B\f$.
 * This difference can be interpreted as the log-fold change if the input matrix contains log-expression values.
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
 * @param options Further options.
 *
 * @return Top markers for each pairwise comparison between labels.
 * Given the `output`, the vector at `output[i][j]` contains the top markers for label `i` over label `j`.
 * Each marker is represented by a pair containing the row index in `matrix` and the difference between medians.
 * Each innermost vector is sorted by the differences between medians.
 * All differences are guaranteed to be positive.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Label_>
std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > > choose(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Label_* label,
    const ChooseOptions& options
) {
    return choose_raw<true, Stat_>(matrix, label, options);
}

/**
 * Variant of `choose()` that only reports the indices of the top markers for each pairwise comparison.
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
 * @param options Further options.
 *
 * @return Top markers for each pairwise comparison between labels.
 * This is the same as the output for `choose()` except that only the row index is reported in the innermost vector.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Label_>
std::vector<std::vector<std::vector<Index_> > > choose_index(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Label_* label,
    const ChooseOptions& options
) {
    return choose_raw<false, Stat_>(matrix, label, options);
}

}

#endif
