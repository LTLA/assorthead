#ifndef SCRAN_QC_CHOOSE_FILTER_THRESHOLDS_HPP
#define SCRAN_QC_CHOOSE_FILTER_THRESHOLDS_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cassert>

#include "sanisizer/sanisizer.hpp"
#include "quickstats/quickstats.hpp"

#include "utils.hpp"

/**
 * @file choose_filter_thresholds.hpp
 * @brief Define QC filter thresholds using a MAD-based approach.
 */

namespace scran_qc {

/**
 * @brief Options for `choose_filter_thresholds()`.
 */
struct ChooseFilterThresholdsOptions {
    /**
     * Should low values be considered as potential outliers?
     * If `false`, no lower threshold is applied when defining outliers.
     */
    bool lower = true;

    /**
     * Should high values be considered as potential outliers?
     * If `false`, no upper threshold is applied when defining outliers.
     */
    bool upper = true;

    /**
     * Number of MADs to use to define outliers.
     * Larger values result in more relaxed thresholds.
     * By default, we require 3 MADs, which is motivated by the low probability (less than 1%) of obtaining such a value for normally distributed data.
     */
    double num_mads = 3;

    /**
     * Minimum difference from the median to define outliers.
     * This enforces a more relaxed threshold in cases where the MAD may be too small.
     * If `ChooseFilterThresholdsOptions::log = true`, this difference is interpreted as a unit on the natural log-scale.
     */
    double min_diff = 0;

    /**
     * Whether to compute the median and MAD on the log-scale.
     * If `true`, the threshold is calculated in log-space, and the log-transformation is reversed before the function returns;
     * this ensures that the reported thresholds are on the original scale of the metrics and can be directly compared to the per-cell values of the metrics.
     *
     * Using a log-transformation instructs the outlier definition to focus on the fold-change from the median. 
     * This has several benefits for right-skewed distributions of (mostly) positive values,
     * where the log-transformation symmetrizes the distribution and makes it more normal-like.
     * This improves the relevance of the interpretation of `ChooseFilterThresholdsOptions::num_mads`. 
     * When defining a lower threshold, the log-transformation also ensures that the defined threshold is always positive.
     *
     * Some caution is required for distributions close to zero, e.g., proportions.
     * The conversion of near-zero values to large negative log-values can unexpectedly inflate the MAD.
     * This could be mitigated by adding a pseudo-count prior to log-transformation,
     * but a large pseudo-count would cause the log-transformation to converge to a linear transformation,
     * rendering this option meaningless for distributions consisting of small values.
     */
    bool log = false;
};

/**
 * @brief Results of `compute_adt_qc_metrics()`.
 * @tparam Float_ Floating-point type of the thresholds.
 */
template<typename Float_>
struct ChooseFilterThresholdsResults {
    /**
     * Lower threshold.
     * Cells where the relevant QC metric is below this threshold are considered to be low quality.
     * This is set to negative infinity if `ChooseFilterThresholdsOptions::lower = false`.
     */
    Float_ lower = 0;

    /**
     * Upper threshold.
     * Cells where the relevant QC metric is above this threshold are considered to be low quality.
     * This is set to positive infinity if `ChooseFilterThresholdsOptions::upper = false`.
     */
    Float_ upper = 0;
};

/**
 * @cond
 */
template<typename Float_>
Float_ unlog_threshold(const Float_ val, const bool was_logged) {
    if (was_logged) {
        if (std::isinf(val)) {
            if (val < 0) {
                return 0;
            }
        } else {
            return std::exp(val);
        }
    }
    return val;
}

template<typename Float_>
ChooseFilterThresholdsResults<Float_> choose_filter_thresholds_internal(
    std::size_t num_cells,
    Float_* metrics,
    const ChooseFilterThresholdsOptions& options
) {
    static_assert(std::is_floating_point<Float_>::value);

    // Rotate all the NaNs to the front of the buffer and ignore them.
    I<decltype(num_cells)> lost = 0;
    for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
        if (std::isnan(metrics[i])) {
            std::swap(metrics[i], metrics[lost]);
            ++lost;
        }
    }
    metrics += lost;
    num_cells -= lost;

    // Maybe we do some log-transformation, if that's requested.
    if (options.log) {
        for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
            auto& val = metrics[i];
            if (val > 0) {
                val = std::log(val);
            } else if (val == 0) {
                val = -std::numeric_limits<double>::infinity();
            } else {
                throw std::runtime_error("cannot log-transform negative values");
            }
        }
    }

    ChooseFilterThresholdsResults<Float_> output;
    Float_& lthresh = output.lower;
    Float_& uthresh = output.upper;
    lthresh = -std::numeric_limits<Float_>::infinity();
    uthresh = std::numeric_limits<Float_>::infinity();

    quickstats::MedianOptions<Float_> medopt;
    medopt.placeholder = std::numeric_limits<Float_>::quiet_NaN();
    const auto median = quickstats::median<Float_>(num_cells, metrics, medopt);

    quickstats::MadOptions<Float_> madopt;
    madopt.placeholder = std::numeric_limits<Float_>::quiet_NaN();
    madopt.difference_between_infinities_is_zero = true; // for sane handling of log(0).
    const auto mad = quickstats::scale_mad_to_sd(quickstats::mad<Float_>(num_cells, metrics, median, madopt));

    if (!std::isnan(mad)) {
        const auto delta = std::max(static_cast<Float_>(options.min_diff), static_cast<Float_>(options.num_mads * mad));
        if (options.lower) {
            const auto threshold = median - delta;
            if (!std::isnan(threshold)) {
                lthresh = unlog_threshold(threshold, options.log);
            }
        }
        if (options.upper) {
            const auto threshold = median + delta;
            if (!std::isnan(threshold)) {
                uthresh = unlog_threshold(threshold, options.log);
            }
        }
    }

    return output;
}
/**
 * @endcond
 */

/**
 * We define filter thresholds on the QC metrics by assuming that most cells in the experiment are of high (or at least acceptable) quality.
 * Any outlier values are indicative of low-quality cells that should be filtered out.
 * Given an array of values, outliers are defined as those that are more than some number of median absolute deviations (MADs) from the median value.
 * Outliers can be defined in both directions or just a single direction, depending on the interpretation of the QC metric.
 * We can also apply a log-transformation to the metrics to identify outliers with respect to their fold-change from the median.
 *
 * @tparam Value_ Numeric type of the input.
 * @tparam Float_ Floating-point type of the thresholds.
 *
 * @param num_cells Number of cells.
 * @param[in] metrics Pointer to an array of length `num_cells`, containing per-cell QC metrics.
 * NaNs are ignored.
 * @param buffer Pointer to an array of length `num_cells`, to be used as a workspace.
 * @param options Further options.
 *
 * @return The upper and lower thresholds derived from `mm`.
 */
template<typename Value_, typename Float_>
ChooseFilterThresholdsResults<Float_> choose_filter_thresholds(
    const std::size_t num_cells,
    const Value_* const metrics,
    Float_* const buffer,
    const ChooseFilterThresholdsOptions& options
) {
    // Only copying it if it's not exactly the same.
    [&](){
        if constexpr(std::is_same<Value_, Float_>::value) {
            if (metrics == buffer) {
                return;
            }
        }
        std::copy_n(metrics, num_cells, buffer);
    }();
    return choose_filter_thresholds_internal(num_cells, buffer, options);
}

/**
 * @brief Workspace for `choose_filter_thresholds_blocked()`.
 *
 * This can be re-used across multiple `choose_filter_thresholds_blocked()` calls to avoid reallocation.
 *
 * @tparam Float_ Floating-point type of the buffer.
 */
template<typename Float_>
struct ChooseFilterThresholdsBlockedWorkspace {
    /**
     * @tparam Block_ Integer type of the block identifiers.
     * @param num_cells Number of cells.
     * @param[in] block Pointer to an array of block identifiers. 
     * The array should be of length equal to `num_cells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * @param num_blocks Total number of blocks, i.e., \f$N\f$.
     */
    template<typename Block_>
    ChooseFilterThresholdsBlockedWorkspace(const std::size_t num_cells, const Block_* const block, const std::size_t num_blocks) {
        reset_choose_filter_thresholds_blocked_workspace(*this, num_cells, block, num_blocks);
    }

    /**
     * Default constructor.
     */
    ChooseFilterThresholdsBlockedWorkspace() = default;

/**
 * @cond
 */
public:
    std::vector<Float_> buffer;
    std::vector<std::size_t> block_starts;
    std::vector<std::size_t> block_offsets;
/**
 * @endcond
 */
};

/**
 * Reset a `ChooseFilterThresholdsBlockedWorkspace` object so that it can be used with a new blocking factor.
 *
 * @tparam Float_ Floating-point type of the buffer.
 * @tparam Block_ Integer type of the block identifiers.
 *
 * @param work Workspace object.
 * On return, `work` is equivalent to an object that was constructed with `num_cells`, `block`, and `num_blocks`.
 * @param num_cells New number of cells.
 * @param[in] block Pointer to an array of block identifiers.
 * The array should be of length equal to `num_cells`.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param num_blocks Total number of blocks, i.e., \f$N\f$.
 */
template<typename Float_, typename Block_>
void reset_choose_filter_thresholds_blocked_workspace(
    ChooseFilterThresholdsBlockedWorkspace<Float_>& work,
    const std::size_t num_cells,
    const Block_* const block,
    const std::size_t num_blocks
) {
    work.block_starts.clear();

    sanisizer::resize(work.block_starts, num_blocks);
    for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
        ++work.block_starts[block[i]];
    }

    std::size_t sofar = 0;
    for (auto& s : work.block_starts) {
        const auto last = sofar;
        sofar += s;
        s = last;
    }

    sanisizer::resize(work.buffer, num_cells
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );

    sanisizer::resize(work.block_offsets, num_blocks
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
}

/**
 * For datasets with multiple blocks, we can compute block-specific thresholds for each metric.
 * This is equivalent to calling `choose_filter_thresholds()` on the cells for each block.
 * Our assumption is that differences in the metric distributions between blocks are driven by uninteresting causes (e.g., differences in sequencing depth);
 * variable thresholds can adapt to each block's distribution for effective removal of outliers.
 *
 * That said, if the differences in the distributions between blocks are interesting,
 * it may be preferable to ignore the blocking factor and just use `choose_filter_thresholds()` instead.
 * This ensures that the MADs are increased appropriately to avoid filtering out interesting variation.
 *
 * @tparam Value_ Numeric type of the input.
 * @tparam Block_ Integer type of the block identifiers.
 * @tparam Float_ Floating-point type of the thresholds.
 *
 * @param num_cells Number of cells.
 * @param[in] metrics Pointer to an array of length `num_cells`, containing per-cell QC metrics.
 * NaNs are ignored.
 * @param[in] block Pointer to an array of length `num_cells`, containing block assignments for all cells.
 * Eacn entry should be an integer ID in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param num_blocks Total number of blocks, i.e., \f$N\f$.
 * @param workspace Pointer to a workspace object, either (i) constructed with `num_cells`, `block`, and `num_blocks`. 
 * or (ii) configured using `reset_choose_filter_thresholds_blocked_workspace()` on `num_cells`, `block`, and `num_blocks`. 
 * The same object can be re-used across multiple calls to `choose_filter_thresholds_blocked()` with the same `num_cells`, `block` and `num_blocks`.
 * @param options Further options.
 *
 * @return A vector containing the upper and lower thresholds for each block.
 */
template<typename Value_, typename Block_, typename Float_>
std::vector<ChooseFilterThresholdsResults<Float_> > choose_filter_thresholds_blocked(
    const std::size_t num_cells,
    const Value_* const metrics,
    const Block_* const block,
    const std::size_t num_blocks,
    ChooseFilterThresholdsBlockedWorkspace<Float_>& workspace,
    const ChooseFilterThresholdsOptions& options
) {
    std::vector<ChooseFilterThresholdsResults<Float_> > output;
    output.reserve(num_blocks);
    process_blocks_for_choose_filter_thresholds(
        num_cells,
        metrics,
        block,
        num_blocks,
        workspace,
        [&](const std::size_t len, Float_* const ptr) -> void {
            output.push_back(choose_filter_thresholds_internal<Float_>(len, ptr, options));
        }
    );
    return output;
}

/**
 * @cond
 */
template<typename Value_, typename Block_, typename Float_, class Function_>
void process_blocks_for_choose_filter_thresholds(
    const std::size_t num_cells,
    const Value_* const metrics,
    const Block_* const block,
    const std::size_t num_blocks,
    ChooseFilterThresholdsBlockedWorkspace<Float_>& workspace,
    Function_ fun 
) {
    assert(num_cells == workspace.buffer.size());
    assert(num_blocks == workspace.block_starts.size());

    auto& buffer = workspace.buffer;
    const auto& starts = workspace.block_starts;
    auto& offsets = workspace.block_offsets;
    std::copy(starts.begin(), starts.end(), offsets.begin());
    for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
        auto& pos = offsets[block[i]];
        buffer[pos] = metrics[i];
        ++pos;
    }

    for (I<decltype(num_blocks)> g = 0; g < num_blocks; ++g) {
        fun(offsets[g] - starts[g], buffer.data() + starts[g]);
    }
}



template<bool lower_, typename Float_>
std::vector<Float_> extract_filter_thresholds(const std::vector<ChooseFilterThresholdsResults<Float_> >& res) {
    std::vector<Float_> output;
    output.reserve(res.size());
    for (const auto& r : res) {
        if constexpr(lower_) {
            output.push_back(r.lower);
        } else {
            output.push_back(r.upper);
        }
    }
    return output;
}
/**
 * @endcond
 */

}

#endif
