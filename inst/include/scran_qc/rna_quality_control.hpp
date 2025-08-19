#ifndef SCRAN_QC_RNA_QUALITY_CONTROL_HPP
#define SCRAN_QC_RNA_QUALITY_CONTROL_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstddef>

#include "tatami/tatami.hpp"

#include "find_median_mad.hpp"
#include "per_cell_qc_metrics.hpp"
#include "choose_filter_thresholds.hpp"

/**
 * @file rna_quality_control.hpp
 * @brief Simple per-cell QC metrics from an RNA count matrix.
 */

namespace scran_qc {

/**
 * @brief Options for `compute_rna_qc_metrics()`.
 */
struct ComputeRnaQcMetricsOptions {
    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `compute_rna_qc_metrics()`.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 * 
 * Note that, unlike `PerCellQcMetricsBuffers`, all pointers are expected to be non-NULL here.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Proportion_ = double>
struct ComputeRnaQcMetricsBuffers {
    /**
     * Pointer to an array of length equal to the number of cells, to store the sum of counts in each cell.
     * This is analogous to `ComputeRnaQcMetricsResults::sum`.
     */
    Sum_* sum = NULL;

    /**
     * Pointer to an array of length equal to the number of cells, to store the number of detected genes in each cell.
     * This is analogous to `ComputeRnaQcMetricsResults::detected`.
     */
    Detected_* detected = NULL;

    /**
     * Vector of pointers of length equal to the number of feature subsets.
     * Each entry should point to an array of length equal to the number of cells, to store the subset proportion in each cell.
     * This is analogous to `ComputeRnaQcMetricsResults::subset_proportion`.
     */
    std::vector<Proportion_*> subset_proportion;
};

/**
 * Given a feature-by-cell RNA count matrix, we compute several metrics for filtering high-quality cells:
 * 
 * - The total sum of counts for each cell, which represents the efficiency of library preparation and sequencing.
 *   Low totals indicate that the library was not successfully captured.
 * - The number of detected features.
 *   This also quantifies the library preparation efficiency, but with a greater focus on capturing the transcriptional complexity.
 * - The proportion of counts in pre-defined feature subsets, the exact interpretation of which depends on the nature of the subset.
 *   Typically, one subset contains all genes on the mitochondrial chromosome, where higher proportions are representative of cell damage;
 *   the assumption is that cytoplasmic transcripts leak through tears in the cell membrane while the mitochondria are still trapped inside.
 *   The prportion of spike-in transcripts can be interpreted in a similar manner, where the loss of endogenous transcripts results in higher spike-in proportions.
 *
 * We use these metrics to define thresholds for filtering in `compute_rna_qc_filters()`.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 *
 * @param mat A **tatami** matrix containing counts.
 * Rows should correspond to genes while columns should correspond to cells.
 * @param[in] subsets Vector of feature subsets, typically mitochondrial genes or spike-in transcripts. 
 * See `per_cell_qc_metrics()` for more details on the expected format.
 * @param[out] output Collection of buffers in which to store the output.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_, typename Proportion_>
void compute_rna_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& output,
    const ComputeRnaQcMetricsOptions& options)
{
    auto NC = mat.ncol();
    auto nsubsets = subsets.size();

    PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_> tmp;
    tmp.sum = output.sum;
    tmp.detected = output.detected;

    constexpr bool same_type = std::is_same<Sum_, Proportion_>::value;
    typename std::conditional<same_type, bool, std::vector<std::vector<Sum_> > >::type placeholder_subset;

    if (output.subset_proportion.size()) {
        // Providing space for the subset sums if they're not the same type.
        if constexpr(same_type) {
            tmp.subset_sum = output.subset_proportion;
        } else {
            placeholder_subset.resize(sanisizer::cast<decltype(placeholder_subset.size())>(nsubsets));
            tmp.subset_sum.resize(sanisizer::cast<decltype(tmp.subset_sum.size())>(nsubsets));
            for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
                auto& b = placeholder_subset[s];
                tatami::resize_container_to_Index_size(b, NC);
                tmp.subset_sum[s] = b.data();
            }
        }
    }

    PerCellQcMetricsOptions opt;
    opt.num_threads = options.num_threads;
    per_cell_qc_metrics(mat, subsets, tmp, opt);

    for (decltype(nsubsets) s = 0 ; s < nsubsets; ++s) {
        auto dest = output.subset_proportion[s];
        if (dest) {
            auto src = tmp.subset_sum[s];
            for (Index_ c = 0; c < NC; ++c) {
                dest[c] = static_cast<Proportion_>(src[c]) / static_cast<Proportion_>(tmp.sum[c]);
            }
        }
    }
}

/**
 * @brief Results of `compute_rna_qc_metrics()`.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Proportion_ = double>
struct ComputeRnaQcMetricsResults {
    /**
     * Vector of length equal to the number of cells in the dataset, containing the sum of counts for each cell.
     */
    std::vector<Sum_> sum;

    /**
     * Vector of length equal to the number of cells in the dataset, containing the number of detected features in each cell.
     */
    std::vector<Detected_> detected;

    /**
     * Proportion of counts in each feature subset in each cell.
     * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
     */
    std::vector<std::vector<Proportion_> > subset_proportion;
};

/**
 * Overload of `compute_rna_qc_metrics()` that allocates memory for the results.
 *
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 *
 * @param mat A **tatami** matrix containing counts.
 * Rows should correspond to genes while columns should correspond to cells.
 * @param[in] subsets Vector of feature subsets, typically mitochondrial genes or spike-in transcripts. 
 * See `per_cell_qc_metrics()` for more details on the expected format.
 * @param options Further options.
 *
 * @return An object containing the QC metrics.
 * Subset proportions are returned depending on the `subsets`.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Proportion_ = double, typename Value_, typename Index_, typename Subset_>
ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_> compute_rna_qc_metrics(const tatami::Matrix<Value_, Index_>& mat, const std::vector<Subset_>& subsets, const ComputeRnaQcMetricsOptions& options) {
    auto NC = mat.ncol();
    ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_> buffers;
    ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_> output;

    tatami::resize_container_to_Index_size(output.sum, NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    buffers.sum = output.sum.data();

    tatami::resize_container_to_Index_size(output.detected, NC
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    buffers.detected = output.detected.data();

    auto nsubsets = subsets.size();
    buffers.subset_proportion.resize(sanisizer::cast<decltype(buffers.subset_proportion.size())>(nsubsets));
    output.subset_proportion.resize(sanisizer::cast<decltype(output.subset_proportion.size())>(nsubsets));
    for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
        tatami::resize_container_to_Index_size(output.subset_proportion[s], NC
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.subset_proportion[s] = output.subset_proportion[s].data();
    }

    compute_rna_qc_metrics(mat, subsets, buffers, options);
    return output;
}

/**
 * @brief Options for `compute_rna_qc_filters()`.
 */
struct ComputeRnaQcFiltersOptions {
    /**
     * Number of MADs below the median, to define the threshold for outliers in the number of detected features.
     * This should be non-negative.
     */
    double detected_num_mads = 3;

    /**
     * Number of MADs below the median, to define the threshold for outliers in the total count per cell.
     * This should be non-negative.
     */
    double sum_num_mads = 3;

    /**
     * Number of MADs above the median, to define the threshold for outliers in the subset proportions.
     * This should be non-negative.
     */
    double subset_proportion_num_mads = 3;
};

/**
 * @cond
 */
namespace internal {

template<typename Float_, class Host_, typename Sum_, typename Detected_, typename Proportion_, typename BlockSource_>
void rna_populate(Host_& host, std::size_t n, const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& res, BlockSource_ block, const ComputeRnaQcFiltersOptions& options) {
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
        opts.num_mads = options.sum_num_mads;
        opts.log = true;
        opts.upper = false;
        host.get_sum() = [&]{
            if constexpr(unblocked) {
                return choose_filter_thresholds(n, res.sum, buffer.data(), opts).lower;
            } else {
                return internal::strip_threshold<true>(choose_filter_thresholds_blocked(n, res.sum, block, &buffer, opts));
            }
        }();
    }

    {
        ChooseFilterThresholdsOptions opts;
        opts.num_mads = options.detected_num_mads;
        opts.log = true;
        opts.upper = false;
        host.get_detected() = [&]{
            if constexpr(unblocked) {
                return choose_filter_thresholds(n, res.detected, buffer.data(), opts).lower;
            } else {
                return internal::strip_threshold<true>(choose_filter_thresholds_blocked(n, res.detected, block, &buffer, opts));
            }
        }();
    }

    {
        ChooseFilterThresholdsOptions opts;
        opts.num_mads = options.subset_proportion_num_mads;
        opts.lower = false;

        auto nsubsets = res.subset_proportion.size();
        auto& subhost = host.get_subset_proportion();
        subhost.resize(sanisizer::cast<decltype(subhost.size())>(nsubsets));
        for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
            auto sub = res.subset_proportion[s];
            subhost[s] = [&]{
                if constexpr(unblocked) {
                    return choose_filter_thresholds(n, sub, buffer.data(), opts).upper;
                } else {
                    return internal::strip_threshold<false>(choose_filter_thresholds_blocked(n, sub, block, &buffer, opts));
                }
            }();
        }
    }
}

template<class Host_, typename Sum_, typename Detected_, typename Proportion_, typename BlockSource_, typename Output_>
void rna_filter(const Host_& host, std::size_t n, const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& metrics, BlockSource_ block, Output_* output) {
    constexpr bool unblocked = std::is_same<BlockSource_, bool>::value;
    std::fill_n(output, n, 1);

    const auto& sum = host.get_sum();
    for (decltype(n) i = 0; i < n; ++i) {
        auto thresh = [&]{
            if constexpr(unblocked) {
                return sum;
            } else {
                return sum[block[i]];
            }
        }();
        output[i] = output[i] && (metrics.sum[i] >= thresh);
    }

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

    auto nsubsets = metrics.subset_proportion.size();
    for (decltype(nsubsets) s = 0; s < nsubsets; ++s) {
        auto sub = metrics.subset_proportion[s];
        const auto& sthresh = host.get_subset_proportion()[s];
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

template<typename Sum_, typename Detected_, typename Proportion_>
ComputeRnaQcMetricsBuffers<const Sum_, const Detected_, const Proportion_> to_buffer(const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics) {
    ComputeRnaQcMetricsBuffers<const Sum_, const Detected_, const Proportion_> buffer;
    buffer.sum = metrics.sum.data();
    buffer.detected = metrics.detected.data();
    buffer.subset_proportion.reserve(metrics.subset_proportion.size());
    for (const auto& s : metrics.subset_proportion) {
        buffer.subset_proportion.push_back(s.data());
    }
    return buffer;
}

}
/**
 * @endcond
 */

/**
 * @brief Filter for high-quality cells using RNA-based metrics. 
 * @tparam Float_ Floating-point type for filter thresholds.
 */
template<typename Float_ = double>
class RnaQcFilters {
public:
    /**
     * @return Lower threshold to apply to the sums.
     */
    Float_ get_sum() const {
        return my_sum;
    }

    /**
     * @return Lower threshold to apply to the number of detected genes.
     */
    Float_ get_detected() const {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of feature subsets,
     * containing the upper threshold to apply to each subset proportion.
     */
    const std::vector<Float_>& get_subset_proportion() const {
        return my_subset_proportion;
    }

    /**
     * @return Lower threshold to apply to the sums.
     */
    Float_& get_sum() {
        return my_sum;
    }

    /**
     * @return Lower threshold to apply to the number of detected genes.
     */
    Float_& get_detected() {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of feature subsets,
     * containing the upper threshold to apply to each subset proportion.
     */
    std::vector<Float_>& get_subset_proportion() {
        return my_subset_proportion;
    }

private:
    Float_ my_sum = 0;
    Float_ my_detected = 0;
    std::vector<Float_> my_subset_proportion;

public:
    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Proportion_ Floating-point type to store the proportions.
     * @tparam Output_ Boolean type to store the high quality flags.
     * @param num Number of cells.
     * @param metrics A collection of arrays containing RNA-based QC metrics, filled by `compute_rna_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_rna_qc_filters()`.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Proportion_, typename Output_>
    void filter(std::size_t num, const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& metrics, Output_* output) const {
        internal::rna_filter(*this, num, metrics, false, output);
    }

    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Proportion_ Floating-point type to store the proportions.
     * @tparam Output_ Boolean type to store the high quality flags.
     * @param metrics RNA-based QC metrics returned by `compute_rna_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_rna_qc_filters()`.
     * @param[out] output Pointer to an array of length `num`. 
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Proportion_, typename Output_>
    void filter(const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics, Output_* output) const {
        return filter(metrics.sum.size(), internal::to_buffer(metrics), output);
    }

    /**
     * @tparam Output_ Boolean type to store the high quality flags.
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Proportion_ Floating-point type to store the proportions.
     * @param metrics RNA-based QC metrics returned by `compute_rna_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_rna_qc_filters()`.
     * @return Vector of length `num`, containing the high-quality calls.
     */
    template<typename Output_ = unsigned char, typename Sum_, typename Detected_, typename Proportion_>
    std::vector<Output_> filter(const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics) const {
        auto output = sanisizer::create<std::vector<Output_> >(metrics.sum.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        filter(metrics, output.data());
        return output;
    }
};

/**
 * Using the RNA-relevant QC metrics from `compute_rna_qc_metrics()`,
 * we consider low-quality cells to be those with a low sum, a low number of detected genes, and high subset proportions.
 * we define thresholds for each metric using an MAD-based outlier approach.
 * For the total counts and number of detected features, the outliers are defined after log-transformation of the metrics.
 *
 * @tparam Float_ Floating-point type for the thresholds.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 *
 * @param num Number of cells.
 * @param metrics A collection of buffers containing RNA-based QC metrics, filled by `compute_rna_qc_metrics()`.
 * @param options Further options for filtering.
 * 
 * @return An object containing the filter thresholds.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Proportion_>
RnaQcFilters<Float_> compute_rna_qc_filters(std::size_t num, const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& metrics, const ComputeRnaQcFiltersOptions& options) {
    RnaQcFilters<Float_> output;
    internal::rna_populate<Float_>(output, num, metrics, false, options);
    return output;
}

/**
 * This function computes filter thresholds for RNA-derived QC metrics in blocked datasets (e.g., cells from multiple batches or samples).
 * Each blocking level has its own thresholds, equivalent to calling `compute_rna_qc_filters()` on the cells from each block.
 * This ensures that uninteresting inter-block differences do not inflate the MAD, see `choose_filter_thresholds_blocked()` for more details.
 *
 * @tparam Float_ Floating-point type for the thresholds.
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 *
 * @param metrics RNA-based QC metrics from `compute_rna_qc_metrics()`.
 * @param options Further options for filtering.
 *
 * @return An object containing the filter thresholds.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Proportion_>
RnaQcFilters<Float_> compute_rna_qc_filters(const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics, const ComputeRnaQcFiltersOptions& options) {
    return compute_rna_qc_filters(metrics.sum.size(), internal::to_buffer(metrics), options);
}

/**
 * @brief Filter for high-quality cells using RNA-based metrics with blocking.
 * @tparam Float_ Floating-point type for filter thresholds.
 */
template<typename Float_ = double>
class RnaQcBlockedFilters {
public:
    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the sums in each block.
     */
    const std::vector<Float_>& get_sum() const {
        return my_sum;
    }

    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the number of detected genes in each block.
     */
    const std::vector<Float_>& get_detected() const {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of blocks.
     * Each entry is a vector of length equal to the number of feature subsets,
     * containing the upper threshold to apply to the each subset proportion.
     */
    const std::vector<std::vector<Float_> >& get_subset_proportion() const {
        return my_subset_proportion;
    }

    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the sums in each block.
     */
    std::vector<Float_>& get_sum() {
        return my_sum;
    }

    /**
     * @return Vector of length equal to the number of blocks,
     * containing the lower threshold on the number of detected genes in each block.
     */
    std::vector<Float_>& get_detected() {
        return my_detected;
    }

    /**
     * @return Vector of length equal to the number of feature subsets.
     * Each entry is a vector of length equal to the number of blocks,
     * containing the upper threshold to apply to the subset proportion for that block.
     */
    std::vector<std::vector<Float_> >& get_subset_proportion() {
        return my_subset_proportion;
    }

private:
    std::vector<Float_> my_sum;
    std::vector<Float_> my_detected;
    std::vector<std::vector<Float_> > my_subset_proportion;

public:
    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Proportion_ Floating-point type to store the proportions.
     * @tparam Block_ Integer type for the block assignment.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param num Number of cells.
     * @param metrics A collection of arrays containing RNA-based QC metrics, filled by `compute_rna_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_rna_qc_filters()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Proportion_, typename Block_, typename Output_>
    void filter(std::size_t num, const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& metrics, const Block_* block, Output_* output) const {
        internal::rna_filter(*this, num, metrics, block, output);
    }

    /**
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Proportion_ Floating-point type to store the proportions.
     * @tparam Block_ Integer type for the block assignment.
     * @tparam Output_ Boolean type to store the high quality flags.
     *
     * @param metrics RNA-based QC metrics computed by `compute_rna_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_rna_qc_filters()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     * @param[out] output Pointer to an array of length `num`.
     * On output, this is truthy for cells considered to be of high quality, and false otherwise.
     */
    template<typename Sum_, typename Detected_, typename Proportion_, typename Block_, typename Output_>
    void filter(const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics, const Block_* block, Output_* output) const {
        return filter(metrics.sum.size(), internal::to_buffer(metrics), block, output);
    }

    /**
     * @tparam Output_ Boolean type to store the high quality flags.
     * @tparam Sum_ Numeric type to store the summed expression.
     * @tparam Detected_ Integer type to store the number of cells.
     * @tparam Proportion_ Floating-point type to store the proportions.
     * @tparam Block_ Integer type for the block assignment.
     *
     * @param metrics RNA-based QC metrics computed by `compute_rna_qc_metrics()`.
     * The feature subsets should be the same as those used in the `metrics` supplied to `compute_rna_qc_filters()`.
     * @param[in] block Pointer to an array of length `num` containing block identifiers.
     * Each identifier should correspond to the same blocks used in the constructor.
     *
     * @return Vector of length `num`, containing the high-quality calls.
     */
    template<typename Output_ = unsigned char, typename Sum_, typename Detected_, typename Proportion_, typename Block_>
    std::vector<Output_> filter(const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics, const Block_* block) const {
        auto output = sanisizer::create<std::vector<Output_> >(metrics.sum.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        filter(metrics, block, output.data());
        return output;
    }
};

/**
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param num Number of cells.
 * @param metrics A collection of buffers containing RNA-based QC metrics, filled by `compute_rna_qc_metrics()`.
 * @param[in] block Pointer to an array of length `num` containing block identifiers.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds for each block.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Proportion_, typename Block_>
RnaQcBlockedFilters<Float_> compute_rna_qc_filters_blocked(
    std::size_t num,
    const ComputeRnaQcMetricsBuffers<Sum_, Detected_, Proportion_>& metrics,
    const Block_* block,
    const ComputeRnaQcFiltersOptions& options) 
{
    RnaQcBlockedFilters<Float_> output;
    internal::rna_populate<Float_>(output, num, metrics, block, options);
    return output;
}

/**
 * @tparam Sum_ Numeric type to store the summed expression.
 * @tparam Detected_ Integer type to store the number of cells.
 * @tparam Proportion_ Floating-point type to store the proportions.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param metrics RNA-based QC metrics computed by `compute_rna_qc_metrics()`.
 * @param[in] block Pointer to an array of length `num` containing block identifiers.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * @param options Further options for filtering.
 *
 * @return Object containing filter thresholds for each block.
 */
template<typename Float_ = double, typename Sum_, typename Detected_, typename Proportion_, typename Block_>
RnaQcBlockedFilters<Float_> compute_rna_qc_filters_blocked(
    const ComputeRnaQcMetricsResults<Sum_, Detected_, Proportion_>& metrics,
    const Block_* block,
    const ComputeRnaQcFiltersOptions& options)
{
    return compute_rna_qc_filters_blocked(metrics.sum.size(), internal::to_buffer(metrics), block, options);
}

}

#endif
