#ifndef SCRAN_QC_CHOOSE_FILTER_THRESHOLDS_HPP
#define SCRAN_QC_CHOOSE_FILTER_THRESHOLDS_HPP

#include <vector>
#include <limits>
#include <cmath>

#include "find_median_mad.hpp"

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
     * By default, we require 3 MADs, which is motivated by the low probability (less than 1%) of obtaining such a value under the normal distribution.
     */
    double num_mads = 3;

    /**
     * Minimum difference from the median to define outliers.
     * This enforces a more relaxed threshold in cases where the MAD may be too small.
     * If `ChooseFilterThresholdsOptions::log = true`, this difference is interpreted as a unit on the log-scale.
     */
    double min_diff = 0;

    /**
     * Whether the supplied median and MAD should be computed on the log-scale (i.e., `FindMedianMadOptions::log = true`).
     * This focuses on the fold-change from the median when defining outliers.
     * In practice, this is useful for metrics that are always positive and have right-skewed distributions,
     * as the log-transformation symmetrizes the distribution and makes it more normal-like such that the `ChooseFilterThresholdsOptions::num_mads` interpretation can be applied.
     * It also ensures that the defined threshold is always positive.
     *
     * If this is set to true, the thresholds are converted back to the original scale of the metrics prior to filtering.
     */
    bool log = false;
};

/**
 * @brief Results of `compute_adt_qc_metrics()`.
 * @tparam Float_ Floating-point type for the thresholds.
 */
template<typename Float_>
struct ChooseFilterThresholdsResults {
    /**
     * Lower threshold.
     * Cells where the relevant QC metric is below this threshold are considered to be low quality.j
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
namespace internal {

template<typename Float_>
Float_ unlog_threshold(Float_ val, bool was_logged) {
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

template<bool lower_, typename Float_>
std::vector<Float_> strip_threshold(const std::vector<ChooseFilterThresholdsResults<Float_> >& res) {
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
 * @tparam Float_ Floating-point type for the thresholds.
 * @param mm Median and MADc computed by `find_median_mad()`.
 * If `ChooseFilterThresholdsOptions::log = true`, it is expected that the median and MAD are computed on the log-transformed metrics
 * (i.e., `FindMedianMadOptions::log = true`).
 * @param options Further options.
 * @return The upper and lower thresholds derived from `mm`.
 */
template<typename Float_>
ChooseFilterThresholdsResults<Float_> choose_filter_thresholds(const FindMedianMadResults<Float_>& mm, const ChooseFilterThresholdsOptions& options) {
    static_assert(std::is_floating_point<Float_>::value);
    ChooseFilterThresholdsResults<Float_> output;
    Float_& lthresh = output.lower;
    Float_& uthresh = output.upper;
    lthresh = -std::numeric_limits<Float_>::infinity();
    uthresh = std::numeric_limits<double>::infinity();

    auto median = mm.median;
    auto mad = mm.mad;
    if (!std::isnan(median) && !std::isnan(mad)) {
        auto delta = std::max(static_cast<Float_>(options.min_diff), options.num_mads * mad);
        if (options.lower) {
            lthresh = internal::unlog_threshold(median - delta, options.log);
        }
        if (options.upper) {
            uthresh = internal::unlog_threshold(median + delta, options.log);
        }
    }

    return output;
}

/**
 * This overload computes the median and MAD via `find_median_mad()` before deriving thresholds with `choose_filter_thresholds()`.
 *
 * @tparam Float_ Floating-point type for the metrics and thresholds.
 *
 * @param num Number of cells.
 * @param[in] metrics Pointer to an array of length `num`, containing a QC metric for each cell.
 * This is modified arbitrarily on output.
 * @param options Further options.
 *
 * @return The upper and lower thresholds derived from `metrics`.
 */
template<typename Float_>
ChooseFilterThresholdsResults<Float_> choose_filter_thresholds(std::size_t num, Float_* metrics, const ChooseFilterThresholdsOptions& options) {
    FindMedianMadOptions fopt;
    fopt.log = options.log;
    auto mm = find_median_mad(num, metrics, fopt);
    return choose_filter_thresholds(mm, options);
}

/**
 * Overload of `choose_filter_thresholds()` that uses an auxiliary buffer to avoid mutating `metrics`.
 *
 * @tparam Value_ Type for the input data.
 * @tparam Float_ Floating-point type for the metrics and thresholds.
 *
 * @param num Number of cells.
 * @param[in] metrics Pointer to an array of length `num`, containing a QC metric for each cell.
 * @param buffer Pointer to an array of length `num` in which to store intermediate results.
 * Alternatively NULL, in which case a buffer is automatically allocated.
 * @param options Further options.
 *
 * @return The upper and lower thresholds derived from `metrics`.
 */
template<typename Value_, typename Float_>
ChooseFilterThresholdsResults<Float_> choose_filter_thresholds(std::size_t num, const Value_* metrics, Float_* buffer, const ChooseFilterThresholdsOptions& options) {
    FindMedianMadOptions fopt;
    fopt.log = options.log;
    auto mm = find_median_mad(num, metrics, buffer, fopt);
    return choose_filter_thresholds(mm, options);
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
 * @tparam Float_ Floating-point type for the thresholds.
 * @param mms Vector of medians and MADs for each block.
 * @param options Further options.
 *
 * @return A vector containing the upper and lower thresholds for each block.
 */
template<typename Float_>
std::vector<ChooseFilterThresholdsResults<Float_> > choose_filter_thresholds_blocked(
    const std::vector<FindMedianMadResults<Float_> >& mms,
    const ChooseFilterThresholdsOptions& options)
{
    std::vector<ChooseFilterThresholdsResults<Float_> > output;
    output.reserve(mms.size());
    for (auto& mm : mms) {
        output.emplace_back(choose_filter_thresholds(mm, options));
    }
    return output;
}

/**
 * This overload computes the median and MAD for each block via `find_median_mad_blocked()` 
 * before deriving thresholds in each block with `choose_filter_thresholds_blocked()`.
 *
 * @tparam Value_ Type for the input data.
 * @tparam Float_ Floating-point type for the metrics and thresholds.
 *
 * @param num Number of cells.
 * @param[in] metrics Pointer to an array of length `num`, containing a QC metric for each cell.
 * @param[in] block Optional pointer to an array of block identifiers, see `find_median_mad_blocked()` for details.
 * @param workspace Pointer to a workspace object, see `find_median_mad_blocked()` for details.
 * @param options Further options.
 *
 * @return A vector containing the upper and lower thresholds for each block.
 */
template<typename Value_, typename Block_, typename Float_>
std::vector<ChooseFilterThresholdsResults<Float_> > choose_filter_thresholds_blocked(
    std::size_t num,
    const Value_* metrics,
    const Block_* block,
    FindMedianMadWorkspace<Float_>* workspace,
    const ChooseFilterThresholdsOptions& options)
{
    FindMedianMadOptions fopt;
    fopt.log = options.log;
    auto mms = find_median_mad_blocked(num, metrics, block, workspace, fopt);
    return choose_filter_thresholds_blocked(mms, options);
}

}

#endif
