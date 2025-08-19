#ifndef SCRAN_QC_ADT_QUALITY_CONTROL_HPP
#define SCRAN_QC_ADT_QUALITY_CONTROL_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "find_median_mad.hpp"
#include "per_cell_qc_metrics.hpp"
#include "choose_filter_thresholds.hpp"

/**
 * @file adt_quality_control.hpp
 * @brief Simple per-cell QC metrics from an ADT count matrix.
 */

namespace scran_qc {

/**
 * @brief Options for `compute_adt_qc_metrics()`.
 */
struct ComputeAdtQcMetricsOptions {
    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `compute_adt_qc_metrics()`.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 *
 * Note that, unlike `PerCellQcMetricsBuffers`, all pointers are expected to be non-NULL here.
 */
template<typename Sum_ = double, typename Detected_ = int>
struct ComputeAdtQcMetricsBuffers {
    /**
     * Pointer to an array of length equal to the number of cells, to store the sum of ADT counts for each cell.
     * This is analogous to `ComputeAdtQcMetricsResults::sum`.
     */
    Sum_* sum;

    /**
     * Pointer to an array of length equal to the number of cells, to store the number of detected ADTs for each cell.
     * This is analogous to `ComputeAdtQcMetricsResults::detected`. 
     */
    Detected_* detected;

    /**
     * Vector of pointers of length equal to the number of feature subsets, to store the sum of counts for each ADT subset in each cell.
     * Each entry should point to an array of length equal to the number of cells.
     * This is analogous to `ComputeAdtQcMetricsResults::subset_sum`. 
     */
    std::vector<Sum_*> subset_sum;
};

/**
 * Given a feature-by-cell ADT count matrix, this function uses `per_cell_qc_metrics()` to compute several ADT-relevant QC metrics:
 * 
 * - The sum of counts for each cell, which (in theory) represents the efficiency of library preparation and sequencing.
 *   This is less useful as a QC metric for ADT data given that the sum is strongly influenced by biological variation in the abundance of the targeted features.
 *   Nonetheless, we compute it for diagnostic purposes.
 * - The number of detected tags per cell.
 *   Even though ADTs are commonly applied in situations where few features are highly abundant, 
 *   we still expect detectable coverage of most features due to ambient contamination, non-specific binding or some background expression.
 *   The absence of detectable coverage indicates that library preparation or sequencing depth was suboptimal.
 * - The sum of counts in pre-defined feature subsets.
 *   While the exact interpretation depends on the nature of the subset, the most common use case involves isotype control (IgG) features.
 *   IgG antibodies should not bind to anything, so high coverage suggests that non-specific binding is a problem, e.g., due to antibody conjugates.
 *   (We do not use proportions here, as it is entirely possible for a cell to have no counts for other tags due to the absence of their targeted features;
 *   this would result in a high proportion even if the cell has a "normal" level of non-specific binding.)
 *
 * We use these metrics to define thresholds for filtering in `compute_adt_qc_filters()`.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 *
 * @param mat A **tatami** matrix containing count data.
 * Rows correspond to ADT features while columns correspond to cells.
 * @param[in] subsets Vector of feature subsets, typically IgG controls. 
 * See `per_cell_qc_metrics()` for more details on the expected format.
 * @param[out] output `ComputeAdtQcMetricsBuffers` object in which to store the output.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void compute_adt_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& output,
    const ComputeAdtQcMetricsOptions& options)
{
    PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_> tmp;
    tmp.sum = output.sum;
    tmp.detected = output.detected;
    tmp.subset_sum = output.subset_sum;

    PerCellQcMetricsOptions opt;
    opt.num_threads = options.num_threads;
    per_cell_qc_metrics(mat, subsets, tmp, opt);
}

/**
 * @brief Results of `compute_adt_qc_metrics()`.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 */
template<typename Sum_ = double, typename Detected_ = int>
struct ComputeAdtQcMetricsResults {
    /**
     * Vector of length equal to the number of cells in the dataset, containing the sum of counts for each cell.
     */
    std::vector<Sum_> sum;

    /**
     * Vector of length equal to the number of cells in the dataset, containing the number of detected features in each cell.
     */
    std::vector<Detected_> detected;

    /**
     * Sum of counts in each feature subset in each cell.
     * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
     */
    std::vector<std::vector<Sum_> > subset_sum;
};

/**
 * Overload of `compute_adt_qc_metrics()` that allocates memory for the results.
 *
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 *
 * @param mat A **tatami** matrix containing count data.
 * Rows correspond to ADT features while columns correspond to cells.
 * @param[in] subsets Vector of feature subsets, typically IgG controls.
 * See `per_cell_qc_metrics()` for more details on the expected format.
 * @param options Further options.
 *
 * @return An object containing the QC metrics.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Value_, typename Index_, typename Subset_>
ComputeAdtQcMetricsResults<Sum_, Detected_> compute_adt_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const ComputeAdtQcMetricsOptions& options)
{
    auto NC = mat.ncol();
    ComputeAdtQcMetricsBuffers<Sum_, Detected_> x;
    ComputeAdtQcMetricsResults<Sum_, Detected_> output;

    tatami::resize_container_to_Index_size(output.sum, NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    x.sum = output.sum.data();

    tatami::resize_container_to_Index_size(output.detected, NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    x.detected = output.detected.data();

    auto nsubsets = subsets.size();
    x.subset_sum.resize(sanisizer::cast<decltype(x.subset_sum.size())>(nsubsets));
    output.subset_sum.resize(sanisizer::cast<decltype(output.subset_sum.size())>(nsubsets));
    for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
        tatami::resize_container_to_Index_size(output.subset_sum[s], NC
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        x.subset_sum[s] = output.subset_sum[s].data();
    }

    compute_adt_qc_metrics(mat, subsets, x, options);
    return output;
}

/**
 * @brief Options for `compute_adt_qc_filters()`.
 */
struct ComputeAdtQcFiltersOptions {
    /**
     * Number of MADs below the median, to define the threshold for outliers in the number of detected features.
     * This should be non-negative.
     */
    double detected_num_mads = 3;

    /**
     * Minimum drop in the number of detected features from the median, in order to consider a cell to be of low quality.
     * This should lie in \f$[0, 1)\f$.
     */
    double detected_min_drop = 0.1;

    /**
     * Number of MADs above the median, to define the threshold for outliers in the subset sums.
     * This should be non-negative.
     */
    double subset_sum_num_mads = 3;
};

/**
 * @cond
 */
namespace internal {

template<typename Float_, class Host_, typename Sum_, typename Detected_, typename BlockSource_>
void adt_populate(Host_& host, std::size_t n, const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& res, BlockSource_ block, const ComputeAdtQcFiltersOptions& options) {
    constexpr bool unblocked = std::is_same<BlockSource_, bool>::value;
    auto buffer = [&]{
        if constexpr(unblocked) {
            return sanisizer::create<std::vector<Float_> >(n);
        } else {
            return FindMedianMadWorkspace<Float_>(n, block);
        }
    }();

    {
        ChooseFilterThresholdsOptions opts;
        opts.num_mads = options.detected_num_mads;
        opts.log = true;
        opts.upper = false;
        opts.min_diff = -std::log(1 - options.detected_min_drop);
        host.get_detected() = [&]{
            if constexpr(unblocked) {
                return choose_filter_thresholds(n, res.detected, buffer.data(), opts).lower;
            } else {
                return internal::strip_threshold<true>(choose_filter_thresholds_blocked(n, res.detected, block, &buffer, opts));
            }
        }();
    }

    {
        auto nsubsets = res.subset_sum.size();
        auto& host_subsets = host.get_subset_sum();
        host_subsets.resize(sanisizer::cast<decltype(host_subsets.size())>(nsubsets));

        ChooseFilterThresholdsOptions opts;
        opts.num_mads = options.subset_sum_num_mads;
        opts.log = true;
        opts.lower = false;

        for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
            auto sub = res.subset_sum[s];
            host.get_subset_sum()[s] = [&]{
                if constexpr(unblocked) {
                    return choose_filter_thresholds(n, sub, buffer.data(), opts).upper;
                } else {
                    return internal::strip_threshold<false>(choose_filter_thresholds_blocked(n, sub, block, &buffer, opts));
                }
            }();
        }
    }
}

template<class Host_, typename Sum_, typename Detected_, typename BlockSource_, typename Output_>
void adt_filter(const Host_& host, std::size_t n, const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& metrics, BlockSource_ block, Output_* output) {
    constexpr bool unblocked = std::is_same<BlockSource_, bool>::value;
    std::fill_n(output, n, 1);

    const auto& detected = host.get_detected();
    for (decltype(n) i = 0; i < n; ++i) {
        auto thresh = [&]{
            if constexpr(unblocked) {
                return detected;
            } else {
                return detected[block[i]];
            }
        }();
        output[i] = output[i] && (metrics.detected[i] >= thresh);
    }

    auto nsubsets = metrics.subset_sum.size();
    for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
        auto sub = metrics.subset_sum[s];
        const auto& sthresh = host.get_subset_sum()[s];
        for (decltype(n) i = 0; i < n; ++i) {
            auto thresh = [&]{
                if constexpr(unblocked) {
                    return sthresh;
                } else {
                    return sthresh[block[i]];
                }
            }();
            output[i] = output[i] && (sub[i] <= thresh);
        }
    }
}

template<typename Sum_, typename Detected_>
ComputeAdtQcMetricsBuffers<const Sum_, const Detected_> to_buffer(const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics) {
    ComputeAdtQcMetricsBuffers<const Sum_, const Detected_> buffer;
    buffer.sum = metrics.sum.data();
    buffer.detected = metrics.detected.data();
    buffer.subset_sum.reserve(metrics.subset_sum.size());
    for (const auto& s : metrics.subset_sum) {
        buffer.subset_sum.push_back(s.data());
    }
    return buffer;
}

}
/**
 * @endcond
 */

/**
 * @brief Filter for high-quality cells using ADT-based metrics. 
 * @tparam Float_ Floating-point type for filter thresholds.
 *
 * Instances of this class are typically created by `compute_adt_qc_filters()`.
 */
template<typename Float_ = double>
class AdtQcFilters {
public:
    /**
     * @return Lower threshold to apply to the number of detected tags.
     */
    Float_ get_detected() const {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of feature subsets,
     * containing the upper threshold to apply to each subset proportion.
     */
    const std::vector<Float_>& get_subset_sum() const {
        return my_subset_sum;
    }

    /**
     * @return Lower threshold to apply to the number of detected tags.
     */
    Float_& get_detected() {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of feature subsets,
     * containing the upper threshold to apply to each subset proportion.
     */
    std::vector<Float_>& get_subset_sum() {
        return my_subset_sum;
    }

private:
    Float_ my_detected = 0;
    std::vector<Float_> my_subset_sum;

public:
    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param num Number of cells.
     * @param metrics A collection of arrays containing ADT-based QC metrics, filled by `compute_adt_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_adt_qc_filters()`.
     *
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Output_>
    void filter(std::size_t num, const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& metrics, Output_* output) const {
        internal::adt_filter(*this, num, metrics, false, output);
    }

    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param metrics ADT-based QC metrics returned by `compute_adt_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_adt_qc_filters()`.
     * @param[out] output Pointer to an array of length `num`. 
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Output_>
    void filter(const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics, Output_* output) const {
        return filter(metrics.detected.size(), internal::to_buffer(metrics), output);
    }

    /**
     * @tparam Output_ Boolean type to store the high quality flags.
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     *
     * @param metrics ADT-based QC metrics returned by `compute_adt_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_adt_qc_filters()`.
     *
     * @return Vector of length `num`, containing the high-quality calls.
     */
    template<typename Output_ = unsigned char, typename Sum_, typename Detected_>
    std::vector<Output_> filter(const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics) const {
        auto output = sanisizer::create<std::vector<Output_> >(metrics.detected.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        filter(metrics, output.data());
        return output;
    }
};

/**
 * Using the ADT-relevant QC metrics from `compute_adt_qc_metrics()`,
 * we consider low-quality cells to be those with a low number of detected tags and high subset sums.
 * We define thresholds for each metric using an MAD-based outlier approach (see `choose_filter_thresholds()` for details).
 * For the number of detected features and the subset sums, the outliers are defined after log-transformation of the metrics.
 *
 * For the number of detected features, we supplement the MAD-based threshold with a minimum drop in the proportion from the median.
 * That is, cells are only considered to be low quality if the difference in the number of detected features from the median is greater than a certain percentage.
 * By default, the number must drop by at least 10% from the median.
 * This avoids overly aggressive filtering when the MAD is zero due to the discrete nature of this statistic in datasets with few tags.
 *
 * @tparam Float_ Floating-point type for the thresholds.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 *
 * @param num Number of cells.
 * @param metrics A collection of arrays containing ADT-based QC metrics, filled by `compute_adt_qc_metrics()`.
 * @param options Further options for filtering.
 *
 * @return An object containing the filter thresholds.
 */
template<typename Float_ = double, typename Sum_, typename Detected_>
AdtQcFilters<Float_> compute_adt_qc_filters(std::size_t num, const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& metrics, const ComputeAdtQcFiltersOptions& options) {
    AdtQcFilters<Float_> output;
    internal::adt_populate<Float_>(output, num, metrics, false, options);
    return output;
}

/**
 * @tparam Float_ Floating-point type for the thresholds.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 *
 * @param metrics ADT-based QC metrics from `compute_adt_qc_metrics()`.
 * @param options Further options for filtering.
 *
 * @return An object containing the filter thresholds.
 */
template<typename Float_ = double, typename Sum_, typename Detected_>
AdtQcFilters<Float_> compute_adt_qc_filters(const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics, const ComputeAdtQcFiltersOptions& options) {
    return compute_adt_qc_filters(metrics.detected.size(), internal::to_buffer(metrics), options);
}

/**
 * @brief Filter on ADT-based QC metrics with blocking.
 * @tparam Float_ Floating-point type for filter thresholds.
 *
 * Instances of this class are typically created by `compute_adt_qc_filters_blocked()`.
 */
template<typename Float_ = double>
class AdtQcBlockedFilters {
public:
    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the number of detected tags in each block.
     */
    const std::vector<Float_>& get_detected() const {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of blocks.
     * Each entry is a vector of length equal to the number of feature subsets,
     * containing the upper threshold to apply to the each subset proportion.
     */
    const std::vector<std::vector<Float_> >& get_subset_sum() const {
        return my_subset_sum;
    }

    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the number of detected tags in each block.
     */
    std::vector<Float_>& get_detected() {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of feature subsets.
     * Each entry is a vector of length equal to the number of blocks,
     * containing the upper threshold to apply to the subset proportion for that block.
     */
    std::vector<std::vector<Float_> >& get_subset_sum() {
        return my_subset_sum;
    }

private:
    std::vector<Float_> my_sum;
    std::vector<Float_> my_detected;
    std::vector<std::vector<Float_> > my_subset_sum;

public:
    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Block_ Integer type for the block assignment.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param num Number of cells.
     * @param metrics A collection of arrays containing ADT-based QC metrics, filled by `compute_adt_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_adt_qc_filters()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Block_, typename Output_>
    void filter(std::size_t num, const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& metrics, const Block_* block, Output_* output) const {
        internal::adt_filter(*this, num, metrics, block, output);
    }

    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Block_ Integer type for the block assignment.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param metrics ADT-based QC metrics computed by `compute_adt_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_adt_qc_filters()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Block_, typename Output_>
    void filter(const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics, const Block_* block, Output_* output) const {
        return filter(metrics.detected.size(), internal::to_buffer(metrics), block, output);
    }

    /**
     * @tparam Output_ Boolean type to store the high quality flags.
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Block_ Integer type for the block assignment.
     *
     * @param metrics ADT-based QC metrics computed by `compute_adt_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_adt_qc_filters()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     *
     * @return Vector of length `num`, containing the high-quality calls.
     */
    template<typename Output_ = unsigned char, typename Sum_, typename Detected_, typename Block_>
    std::vector<Output_> filter(const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics, const Block_* block) const {
        auto output = sanisizer::create<std::vector<Output_> >(metrics.detected.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        filter(metrics, block, output.data());
        return output;
    }
};

/**
 * This function computes filter thresholds for ADT-derived QC metrics in blocked datasets (e.g., cells from multiple batches or samples).
 * Each blocking level has its own thresholds, equivalent to calling `compute_adt_qc_filters()` on the cells from each block.
 * This ensures that uninteresting inter-block differences do not inflate the MAD, see `choose_filter_thresholds_blocked()` for more details.
 *
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param num Number of cells.
 * @param metrics A collection of arrays containing ADT-based QC metrics, filled by `compute_adt_qc_metrics()`.
 * @param[in] block Pointer to an array of length `num` containing block identifiers.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds for each block.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Block_>
AdtQcBlockedFilters<Float_> compute_adt_qc_filters_blocked(
    std::size_t num,
    const ComputeAdtQcMetricsBuffers<Sum_, Detected_>& metrics,
    const Block_* block,
    const ComputeAdtQcFiltersOptions& options)
{
    AdtQcBlockedFilters<Float_> output;
    internal::adt_populate<Float_>(output, num, metrics, block, options);
    return output;
}

/**
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param metrics ADT-based QC metrics computed by `compute_adt_qc_metrics()`.
 * @param[in] block Pointer to an array of length `num` containing block identifiers.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds for each block.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Block_>
AdtQcBlockedFilters<Float_> compute_adt_qc_filters_blocked(
    const ComputeAdtQcMetricsResults<Sum_, Detected_>& metrics,
    const Block_* block,
    const ComputeAdtQcFiltersOptions& options)
{
    return compute_adt_qc_filters_blocked(metrics.detected.size(), internal::to_buffer(metrics), block, options);
}

}

#endif
