#ifndef SCRAN_QC_CRISPR_QUALITY_CONTROL_HPP
#define SCRAN_QC_CRISPR_QUALITY_CONTROL_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <type_traits>

#include "tatami/tatami.hpp"

#include "find_median_mad.hpp"
#include "per_cell_qc_metrics.hpp"
#include "choose_filter_thresholds.hpp"

/**
 * @file crispr_quality_control.hpp
 * @brief Simple per-cell QC metrics from a CRISPR count matrix.
 */

namespace scran_qc {

/**
 * @brief Options for `compute_crispr_qc_metrics()`.
 */
struct ComputeCrisprQcMetricsOptions {
    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `compute_crispr_qc_metrics()`.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 *
 * Note that, unlike `PerCellQcMetricsBuffers`, all pointers are expected to be non-NULL here.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Value_ = double, typename Index_ = int>
struct ComputeCrisprQcMetricsBuffers {
    /**
     * Pointer to an array of length equal to the number of cells, to store the sum of CRISPR counts per cell.
     * This is analogous to `ComputeCrisprQcMetricsResults::sum`. 
     */
    Sum_* sum;

    /**
     * Pointer to an array of length equal to the number of cells, to store the number of detected guides per cell.
     * This is analogous to `ComputeCrisprQcMetricsResults::detected`.
     */
    Detected_* detected;

    /**
     * Pointer to an array of length equal to the number of cells, to store the maximum count for each cell.
     * This is analogous to `ComputeCrisprQcMetricsResults::max_value`.
     */
    Value_* max_value;

    /**
     * Pointer to an array of length equal to the number of cells, to store the index of the most abundant guide for each cell.
     * This is analogous to `ComputeCrisprQcMetricsResults::max_index`.
     */
    Index_* max_index;
};

/**
 * Given a feature-by-cell guide count matrix, this function uses `per_cell_qc_metrics()` to compute several CRISPR-relevant QC metrics:
 * 
 * - The sum of counts for each cell.
 *   Low counts indicate that the cell was not successfully transfected with a construct,
 *   or that library preparation and sequencing failed.
 * - The number of detected guides per cell.
 *   In theory, this should be 1, as each cell should express no more than one guide construct.
 *   However, ambient contamination may introduce non-zero counts for multiple guides, without necessarily interfering with downstream analyses.
 *   As such, this metric is less useful for guide data, though we compute it anyway.
 * - The maximum count in the most abundant guide construct.
 *   Low values indicate that the cell was not successfully transfected,
 *   or that library preparation and sequencing failed.
 *   The identity of the most abundant guide is also reported.
 *
 * We use these metrics to define thresholds for filtering in `compute_crispr_qc_filters()`.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Sum_ Numeric type to store the summed expression.
 *
 * Meaningful instances of this object should generally be constructed by calling the `compute_crispr_qc_metrics()` function.
 * @tparam Detected_ Integer type to store the number of cells.
 *
 * @param mat A **tatami** matrix containing count data.
 * Rows correspond to CRISPR guides while columns correspond to cells.
 * @param[out] output `ComputeCrisprQcMetricsBuffers` object in which to store the output.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Sum_, typename Detected_>
void compute_crispr_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    const ComputeCrisprQcMetricsOptions& options)
{
    PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_> tmp;
    tmp.sum = output.sum;
    tmp.detected = output.detected;
    tmp.max_value = output.max_value;
    tmp.max_index = output.max_index;

    PerCellQcMetricsOptions opt;
    opt.num_threads = options.num_threads;
    per_cell_qc_metrics(mat, std::vector<uint8_t*>{}, tmp, opt);
}

/**
 * @brief Results of `compute_crispr_qc_metrics()`.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Value_ = double, typename Index_ = int>
struct ComputeCrisprQcMetricsResults {
    /**
     * Vector of length equal to the number of cells in the dataset, containing the sum of counts for each cell.
     */
    std::vector<Sum_> sum;

    /**
     * Vector of length equal to the number of cells in the dataset, containing the number of detected features in each cell.
     */
    std::vector<Detected_> detected;

    /**
     * Vector of length equal to the number of cells in the dataset, containing the maximum count for each cell.
     */
    std::vector<Value_> max_value;

    /**
     * Vector of length equal to the number of cells in the dataset, containing the row index of the guide with the maximum count for each cell.
     */
    std::vector<Index_> max_index;
};

/**
 * Overload of `compute_crispr_qc_metrics()` that allocates memory for the results.
 *
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 *
 * @param mat A **tatami** matrix containing counts.
 * Each row should correspond to a guide while each column should correspond to a cell.
 * @param options Further options.
 *
 * @return An object containing the QC metrics.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Value_ = double, typename Index_ = int>
ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_> compute_crispr_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const ComputeCrisprQcMetricsOptions& options)
{
    auto NC = mat.ncol();
    ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_> x;
    ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_> output;

    output.sum.resize(NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    x.sum = output.sum.data();

    output.detected.resize(NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    x.detected = output.detected.data();

    output.max_value.resize(NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    x.max_value = output.max_value.data();

    output.max_index.resize(NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    x.max_index = output.max_index.data();

    compute_crispr_qc_metrics(mat, x, options);
    return output;
}

/**
 * @brief Options for `compute_crispr_qc_filters()`.
 */
struct ComputeCrisprQcFiltersOptions {
    /**
     * Number of MADs below the median, to define the threshold for outliers in the maximum count.
     * This should be non-negative.
     */
    double max_value_num_mads = 3;
};

/**
 * @cond
 */
namespace internal {

template<typename Float_, class Host_, typename Sum_, typename Detected_, typename Value_, typename Index_, typename BlockSource_>
void crispr_populate(Host_& host, size_t n, const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& res, BlockSource_ block, const ComputeCrisprQcFiltersOptions& options) {
    constexpr bool unblocked = std::is_same<BlockSource_, bool>::value;
    auto buffer = [&]{
        if constexpr(unblocked) {
            return std::vector<Float_>(n);
        } else {
            return FindMedianMadWorkspace<Float_, size_t>(n, block);
        }
    }();

    // Subsetting to the observations in the top 50% of proportions.
    static_assert(std::is_floating_point<Float_>::value);
    std::vector<Float_> maxprop;
    maxprop.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        maxprop.push_back(static_cast<Float_>(res.max_value[i]) / static_cast<Float_>(res.sum[i]));
    }

    FindMedianMadOptions fopt;
    fopt.median_only = true;
    auto prop_res = [&]{
        if constexpr(unblocked) {
            return find_median_mad(n, maxprop.data(), buffer.data(), fopt);
        } else {
            return find_median_mad_blocked(n, maxprop.data(), block, &buffer, fopt);
        }
    }();

    for (size_t i = 0; i < n; ++i) {
        auto limit = [&]{
            if constexpr(unblocked){
                return prop_res.median;
            } else {
                return prop_res[block[i]].median;
            }
        }();
        if (maxprop[i] >= limit) {
            maxprop[i] = res.max_value[i];
        } else {
            maxprop[i] = std::numeric_limits<Float_>::quiet_NaN(); // ignored during threshold calculation.
        }
    }

    // Filtering on the max counts.
    ChooseFilterThresholdsOptions copt;
    copt.num_mads = options.max_value_num_mads;
    copt.log = true;
    copt.upper = false;
    host.get_max_value() = [&]{
        if constexpr(unblocked) {
            return choose_filter_thresholds(n, maxprop.data(), buffer.data(), copt).lower;
        } else {
            return internal::strip_threshold<true>(choose_filter_thresholds_blocked(n, maxprop.data(), block, &buffer, copt));
        }
    }();
}

template<class Host_, typename Sum_, typename Detected_, typename Value_, typename Index_, typename BlockSource_, typename Output_>
void crispr_filter(const Host_& host, size_t n, const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& metrics, BlockSource_ block, Output_* output) {
    constexpr bool unblocked = std::is_same<BlockSource_, bool>::value;
    std::fill_n(output, n, 1);

    const auto& mv = host.get_max_value();
    for (size_t i = 0; i < n; ++i) {
        auto thresh = [&]{
            if constexpr(unblocked) {
                return mv;
            } else {
                return mv[block[i]];
            }
        }();
        output[i] = output[i] && (metrics.max_value[i] >= thresh);
    }
}

template<typename Sum_, typename Detected_, typename Value_, typename Index_>
ComputeCrisprQcMetricsBuffers<const Sum_, const Detected_, const Value_, const Index_> to_buffer(const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics) {
    ComputeCrisprQcMetricsBuffers<const Sum_, const Detected_, const Value_, const Index_> buffer;
    buffer.sum = metrics.sum.data();
    buffer.detected = metrics.detected.data();
    buffer.max_value = metrics.max_value.data();
    buffer.max_index = metrics.max_index.data();
    return buffer;
}

}
/**
 * @endcond
 */

/**
 * @brief Filter for high-quality cells using CRISPR-based metrics. 
 * @tparam Float_ Floating-point type for filter thresholds.
 *
 * Instances of this class are typically created by `compute_crispr_qc_filters()`.
 */
template<typename Float_ = double>
class CrisprQcFilters {
public:
    /**
     * @return Lower threshold to apply to the maximum count.
     */
    Float_ get_max_value() const {
        return my_max_value;
    }

    /**
     * @return Lower threshold to apply to the maximum count.
     */
    Float_& get_max_value() {
        return my_max_value;
    }

private:
    Float_ my_max_value = 0;

public:
    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Value_ Type of matrix value.
     * @tparam Index_ Type of the matrix indices.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param num Number of cells.
     * @param metrics A collection of arrays containing CRISPR-based QC metrics, filled by `compute_crispr_qc_metrics()`.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Value_, typename Index_, typename Output_>
    void filter(size_t num, const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& metrics, Output_* output) const {
        internal::crispr_filter(*this, num, metrics, false, output);
    }

    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Value_ Type of matrix value.
     * @tparam Index_ Type of the matrix indices.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param metrics CRISPR-based QC metrics returned by `compute_crispr_qc_metrics()`.
     * @param[out] output Pointer to an array of length `num`. 
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Value_, typename Index_, typename Output_>
    void filter(const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics, Output_* output) const {
        return filter(metrics.max_value.size(), internal::to_buffer(metrics), output);
    }

    /**
     * @tparam Output_ Boolean type to store the high quality flags.
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Value_ Type of matrix value.
     * @tparam Index_ Type of the matrix indices.
     *
     * @param metrics CRISPR-based QC metrics returned by `compute_crispr_qc_metrics()`.
     * @return Vector of length `num`, containing the high-quality calls.
     */
    template<typename Output_ = uint8_t, typename Sum_ = double, typename Detected_ = int, typename Value_ = double, typename Index_ = int>
    std::vector<Output_> filter(const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics) const {
        std::vector<Output_> output(metrics.max_value.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        filter(metrics, output.data());
        return output;
    }
};

/**
 * In CRISPR data, low-quality cells are defined as those with a low count for the most abundant guides.
 * However, directly defining a threshold on the maximum count is somewhat tricky as unsuccessful transfection is not uncommon.
 * This often results in a large subpopulation with low maximum counts, inflating the MAD and compromising the threshold calculation.
 * Instead, we use the following approach:
 *
 * 1. Compute the median of the proportion of counts in the most abundant guide (i.e., the maximum proportion),
 * 2. Subset the cells to only those with maximum proportions above the median.
 * 3. Define a threshold for low outliers on the log-transformed maximum count within the subset (see `choose_filter_thresholds()` for details).
 *
 * This assumes that over 50% of cells were successfully transfected with a single guide construct and have high maximum proportions.
 * In contrast, unsuccessful transfections will be dominated by ambient contamination and have low proportions.
 * By taking the subset above the median proportion, we remove all of the unsuccessful transfections and enrich for mostly-high-quality cells.
 * From there, we can apply the usual outlier detection methods on the maximum count, with log-transformation to avoid a negative threshold.
 *
 * Keep in mind that the maximum proportion is only used to define the subset for threshold calculation.
 * Once the maximum count threshold is computed, they are applied to all cells, regardless of their maximum proportions.
 * This allows us to recover good cells that would have been filtered out by our aggressive median subset.
 * It also ensures that we do not remove cells transfected with multiple guides - such cells are not necessarily uninteresting, e.g., for examining interaction effects,
 * so we will err on the side of caution and leave them in.
 *
 * @tparam Float_ Floating-point type for the thresholds.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 *
 * @param num Number of cells.
 * @param metrics A collection of arrays containing CRISPR-based QC metrics, filled by `compute_crispr_qc_metrics()`.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Value_, typename Index_>
CrisprQcFilters<Float_> compute_crispr_qc_filters(
    size_t num,
    const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& metrics,
    const ComputeCrisprQcFiltersOptions& options)
{
    CrisprQcFilters<Float_> output;
    internal::crispr_populate<Float_>(output, num, metrics, false, options);
    return output;
}

/**
 * @tparam Float_ Floating-point type for the thresholds.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 *
 * @param metrics CRISPR-based QC metrics from `compute_crispr_qc_metrics()`.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds.
 */
template<typename Float_ = double, typename Sum_ = double, typename Detected_ = int, typename Value_ = double, typename Index_ = int>
CrisprQcFilters<Float_> compute_crispr_qc_filters(
    const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics,
    const ComputeCrisprQcFiltersOptions& options)
{
    return compute_crispr_qc_filters(metrics.max_value.size(), internal::to_buffer(metrics), options);
}

/**
 * @brief Filter on using CRISPR-based QC metrics with blocking.
 * @tparam Float_ Floating-point type for filter thresholds.
 * Instances of this class are typically created by `compute_crispr_qc_filters_blocked()`.
 */
template<typename Float_ = double>
class CrisprQcBlockedFilters {
public:
    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the maximum count in each block.
     */
    const std::vector<Float_>& get_max_value() const {
        return my_max_value;
    }

    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the maximum count in each block.
     */
    std::vector<Float_>& get_max_value() {
        return my_max_value;
    }

private:
    std::vector<Float_> my_sum;
    std::vector<Float_> my_max_value;

public:
    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Value_ Type of matrix value.
     * @tparam Index_ Type of the matrix indices.
     * @tparam Block_ Integer type for the block assignment.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param num Number of cells.
     * @param metrics A collection of arrays containing CRISPR-based QC metrics, filled by `compute_crispr_qc_metrics()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Value_, typename Index_, typename Block_, typename Output_>
    void filter(size_t num, const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& metrics, const Block_* block, Output_* output) const {
        internal::crispr_filter(*this, num, metrics, block, output);
    }

    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Value_ Type of matrix value.
     * @tparam Index_ Type of the matrix indices.
     * @tparam Block_ Integer type for the block assignment.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param metrics CRISPR-based QC metrics computed by `compute_crispr_qc_metrics()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Value_, typename Index_, typename Block_, typename Output_>
    void filter(const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics, const Block_* block, Output_* output) const {
        filter(metrics.max_value.size(), internal::to_buffer(metrics), block, output);
    }

    /**
     * @tparam Output_ Boolean type to store the high quality flags.
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Value_ Type of matrix value.
     * @tparam Index_ Type of the matrix indices.
     * @tparam Block_ Integer type for the block assignment.
     * 
     * @param metrics CRISPR-based QC metrics computed by `compute_crispr_qc_metrics()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     *
     * @return Vector of length `num`, containing the high-quality calls.
     */
    template<typename Output_ = uint8_t, typename Sum_ = double, typename Detected_ = int, typename Value_ = double, typename Index_ = int, typename Block_ = int>
    std::vector<Output_> filter(const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics, const Block_* block) const {
        std::vector<Output_> output(metrics.max_value.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        filter(metrics, block, output.data());
        return output;
    }
};

/**
 * This function computes filter thresholds for CRISPR-derived QC metrics in blocked datasets (e.g., cells from multiple batches or samples).
 * Each blocking level has its own thresholds, equivalent to calling `compute_crispr_qc_filters()` on the cells from each block.
 * This ensures that uninteresting inter-block differences do not inflate the MAD, see `choose_filter_thresholds_blocked()` for more details.

 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param num Number of cells.
 * @param metrics A collection of arrays containing CRISPR-based QC metrics, filled by `compute_crispr_qc_metrics()`.
 * @param[in] block Pointer to an array of length `num` containing block identifiers.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds for each block.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Value_, typename Index_, typename Block_>
CrisprQcBlockedFilters<Float_> compute_crispr_qc_filters_blocked(
    size_t num,
    const ComputeCrisprQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& metrics,
    const Block_* block,
    const ComputeCrisprQcFiltersOptions& options)
{
    CrisprQcBlockedFilters<Float_> output;
    internal::crispr_populate<Float_>(output, num, metrics, block, options);
    return output;
}

/**
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param metrics CRISPR-based QC metrics computed by `compute_crispr_qc_metrics()`.
 * @param[in] block Pointer to an array of length `num` containing block identifiers.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds for each block.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Value_, typename Index_, typename Block_>
CrisprQcBlockedFilters<Float_> compute_crispr_qc_filters_blocked(
    const ComputeCrisprQcMetricsResults<Sum_, Detected_, Value_, Index_>& metrics,
    const Block_* block,
    const ComputeCrisprQcFiltersOptions& options)
{
    return compute_crispr_qc_filters_blocked(metrics.max_value.size(), internal::to_buffer(metrics), block, options);
}

}

#endif
