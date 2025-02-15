#ifndef SCRAN_QC_PER_CELL_QC_METRICS_HPP
#define SCRAN_QC_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

/**
 * @file per_cell_qc_metrics.hpp
 * @brief Compute per-cell quality control metrics.
 */

namespace scran_qc {

/**
 * @brief Options for `per_cell_qc_metrics()`.
 */
struct PerCellQcMetricsOptions {
    /**
     * Whether to compute the sum of expression values for each cell.
     * This option only affects the `per_cell_qc_metrics()` overload that returns a `PerCellQcMetricsResults` object.
     */
    bool compute_sum = true;

    /**
     * Whether to compute the number of detected features for each cell.
     * This option only affects the `per_cell_qc_metrics()` overload that returns a `PerCellQcMetricsResults` object.
     */
    bool compute_detected = true;

    /**
     * Whether to compute the maximum expression value for each cell.
     * This option only affects the `per_cell_qc_metrics()` overload that returns a `PerCellQcMetricsResults` object.
     */
    bool compute_max_value = true;

    /**
     * Whether to store the index of the feature with the maximum value for each cell.
     * This option only affects the `per_cell_qc_metrics()` overload that returns a `PerCellQcMetricsResults` object.
     */
    bool compute_max_index = true;

    /**
     * Whether to compute the sum expression in each feature subset.
     * This option only affects the `per_cell_qc_metrics()` overload that returns a `PerCellQcMetricsResults` object.
     */
    bool compute_subset_sum = true;

    /**
     * Whether to compute the number of detected features in each feature subset.
     * This option only affects the `per_cell_qc_metrics()` overload that returns a `PerCellQcMetricsResults` object.
     */
    bool compute_subset_detected = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `per_cell_qc_metrics()`.
 *
 * @tparam Sum_ Floating point type to store the sums.
 * @tparam Detected_ Integer type to store the number of detected cells.
 * @tparam Value_ Type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 */
template<typename Sum_, typename Detected_, typename Value_, typename Index_>
struct PerCellQcMetricsBuffers {
    /**
     * @cond
     */
    PerCellQcMetricsBuffers() = default;

    PerCellQcMetricsBuffers(size_t nsubsets) : subset_sum(nsubsets, NULL), subset_detected(nsubsets, NULL) {}
    /**
     * @endcond
     */

    /**
     * Pointer to an array of length equal to the number of cells, equivalent to `PerCellQcMetricsResults::sum`.
     * Set to `NULL` to skip this calculation.
     */
    Sum_* sum = NULL;

    /**
     * Pointer to an array of length equal to the number of cells, equivalent to `PerCellQcMetricsResults::detected`.
     * Set to `NULL` to skip this calculation.
     */
    Detected_* detected = NULL;

    /**
     * Pointer to an array of length equal to the number of cells, equivalent to `PerCellQcMetricsResults::max_index`.
     * Set to `NULL` to skip this calculation.
     */
    Index_* max_index = NULL;

    /**
     * Pointer to an array of length equal to the number of cells, equivalent to `PerCellQcMetricsResults::max_value`.
     * Set to `NULL` to skip this calculation.
     */
    Value_* max_value = NULL;

    /**
     * Vector of pointers of length equal to the number of feature subsets,
     * where each point is to an array of length equal to the number of cells; equivalent to `PerCellQcMetricsResults::subset_sum`.
     * Set any value to `NULL` to skip the calculation for the corresponding feature subset,
     * or leave empty to skip calculations for all feature subsets.
     */
    std::vector<Sum_*> subset_sum;

    /**
     * Vector of pointers of length equal to the number of feature subsets,
     * where each point is to an array of length equal to the number of cells; equivalent to `PerCellQcMetricsResults::subset_detected`.
     * Set any value to `NULL` to skip the calculation for the corresponding feature subset,
     * or leave empty to skip calculations for all feature subsets.
     */
    std::vector<Detected_*> subset_detected;
};

/**
 * @cond
 */
namespace internal {

template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void compute_qc_direct_dense(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    int num_threads)
{
    std::vector<std::vector<Index_> > subset_indices;
    if (!output.subset_sum.empty() || !output.subset_detected.empty()) {
        if constexpr(std::is_pointer<Subset_>::value) {
            size_t nsubsets = subsets.size();
            subset_indices.resize(nsubsets);
            Index_ NR = mat.nrow();

            for (size_t s = 0; s < nsubsets; ++s) {
                auto& current = subset_indices[s];
                const auto& source = subsets[s];
                for (Index_ i = 0; i < NR; ++i) {
                    if (source[i]) {
                        current.push_back(i);
                    }
                }
            }
        }
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto NR = mat.nrow();
        auto ext = tatami::consecutive_extractor<false>(&mat, false, start, length);
        std::vector<Value_> vbuffer(NR);

        bool do_max = output.max_index || output.max_value;

        size_t nsubsets = subsets.size();

        for (Index_ c = start, end = start + length; c < end; ++c) {
            auto ptr = ext->fetch(c, vbuffer.data());

            if (output.sum) {
                output.sum[c] = std::accumulate(ptr, ptr + NR, static_cast<Sum_>(0));
            }

            if (output.detected) {
                Detected_ count = 0;
                for (Index_ r = 0; r < NR; ++r) {
                    count += (ptr[r] != 0);
                }
                output.detected[c] = count;
            }

            if (do_max) {
                Index_ max_index = 0;
                Value_ max_value = 0;

                if (NR) {
                    max_value = ptr[0];
                    for (Index_ r = 1; r < NR; ++r) {
                        if (max_value < ptr[r]) {
                            max_value = ptr[r];
                            max_index = r;
                        }
                    }
                }

                if (output.max_index) {
                    output.max_index[c] = max_index;
                }
                if (output.max_value) {
                    output.max_value[c] = max_value;
                }
            }

            if (!output.subset_sum.empty() || !output.subset_detected.empty()) { // protect against accessing an empty subset_indices.
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = [&]() -> const auto& {
                        if constexpr(std::is_pointer<Subset_>::value) {
                            return subset_indices[s];
                        } else {
                            return subsets[s];
                        }
                    }();

                    if (!output.subset_sum.empty() && output.subset_sum[s]) {
                        Sum_ current = 0;
                        for (auto r : sub) {
                            current += ptr[r];
                        }
                        output.subset_sum[s][c] = current;
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        Detected_ current = 0;
                        for (auto r : sub) {
                            current += ptr[r] != 0;
                        }
                        output.subset_detected[s][c] = current;
                    }
                }
            }
        }
    }, mat.ncol(), num_threads);
}

template<typename Index_, typename Subset_, typename Sum_, typename Detected_, typename Value_>
std::vector<std::vector<uint8_t> > boolify_subsets(Index_ NR, const std::vector<Subset_>& subsets, const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output) {
    std::vector<std::vector<uint8_t> > is_in_subset;

    if (!output.subset_sum.empty() || !output.subset_detected.empty()) {
        if constexpr(!std::is_pointer<Subset_>::value) {
            size_t nsubsets = subsets.size();
            for (size_t s = 0; s < nsubsets; ++s) {
                is_in_subset.emplace_back(NR);
                auto& last = is_in_subset.back();
                for (auto i : subsets[s]) {
                    last[i] = 1;
                }
            }
        }
    }

    return is_in_subset;
}

template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void compute_qc_direct_sparse(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    int num_threads)
{
    auto is_in_subset = boolify_subsets(mat.nrow(), subsets, output);

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto NR = mat.nrow();
        auto ext = tatami::consecutive_extractor<true>(&mat, false, start, length);
        std::vector<Value_> vbuffer(NR);
        std::vector<Index_> ibuffer(NR);

        bool do_max = output.max_index || output.max_value;

        size_t nsubsets = subsets.size();

        for (Index_ c = start, end = start + length; c < end; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            if (output.sum) {
                output.sum[c] = std::accumulate(range.value, range.value + range.number, static_cast<Sum_>(0));
            }

            if (output.detected) {
                Detected_ current = 0;
                for (Index_ i = 0; i < range.number; ++i) {
                    current += (range.value[i] != 0);
                }
                output.detected[c] = current;
            }

            if (do_max) {
                Index_ max_index = 0;
                Value_ max_value = 0;

                if (range.number) {
                    max_value = range.value[0];
                    max_index = range.index[0];
                    for (Index_ i = 1; i < range.number; ++i) {
                        if (max_value < range.value[i]) {
                            max_value = range.value[i];
                            max_index = range.index[i];
                        }
                    }

                    if (max_value <= 0 && range.number < NR) {
                        if (output.max_index) {
                            // Figuring out the index of the first zero, assuming range.index is sorted.
                            Index_ last = 0;
                            for (Index_ i = 0; i < range.number; ++i) {
                                if (range.index[i] > last) { // must be at least one intervening structural zero.
                                    break;
                                } else if (range.value[i] == 0) { // appears earlier than any structural zero, so it's already the maximum.
                                    break;
                                }
                                last = range.index[i] + 1;
                            }
                            max_index = last;
                        }
                        max_value = 0;
                    }
                } else if (NR) {
                    max_value = 0;
                }

                if (output.max_index) {
                    output.max_index[c] = max_index;
                }
                if (output.max_value) {
                    output.max_value[c] = max_value;
                }
            }

           if (!output.subset_sum.empty() || !output.subset_detected.empty()) { // protect against accessing an empty is_in_subset.
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = [&]() -> const auto& {
                        if constexpr(std::is_pointer<Subset_>::value) {
                            return subsets[s];
                        } else {
                            return is_in_subset[s];
                        }
                    }();

                    if (!output.subset_sum.empty() && output.subset_sum[s]) {
                        Sum_ current = 0;
                        for (Index_ i = 0; i < range.number; ++i) {
                            current += (sub[range.index[i]] != 0) * range.value[i];
                        }
                        output.subset_sum[s][c] = current;
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        Detected_ current = 0;
                        for (Index_ i = 0; i < range.number; ++i) {
                            current += (sub[range.index[i]] != 0) * (range.value[i] != 0);
                        }
                        output.subset_detected[s][c] = current;
                    }
                }
            }
        }
    }, mat.ncol(), num_threads);
}

template<typename Sum_, typename Detected_, typename Value_, typename Index_>
class PerCellQcMetricsRunningBuffers {
public:
    PerCellQcMetricsRunningBuffers(const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output, int thread, Index_ start, Index_ len) {
        if (output.sum) {
            my_sum = tatami_stats::LocalOutputBuffer<Sum_>(thread, start, len, output.sum);
        }

        if (output.detected) {
            my_detected = tatami_stats::LocalOutputBuffer<Detected_>(thread, start, len, output.detected);
        }

        if (output.max_value) {
            my_max_value = tatami_stats::LocalOutputBuffer<Value_>(thread, start, len, output.max_value);
        } else if (output.max_index) {
            my_holding_max_value.resize(len);
        }

        if (output.max_index) {
            my_max_index = tatami_stats::LocalOutputBuffer<Index_>(thread, start, len, output.max_index);
        }

        my_subset_sum.resize(output.subset_sum.size());
        for (size_t s = 0, send = output.subset_sum.size(); s < send; ++s) {
            if (output.subset_sum[s]) {
                my_subset_sum[s] = tatami_stats::LocalOutputBuffer<Sum_>(thread, start, len, output.subset_sum[s]);
            }
        }

        my_subset_detected.resize(output.subset_detected.size());
        for (size_t s = 0, send = output.subset_detected.size(); s < send; ++s) {
            if (output.subset_detected[s]) {
                my_subset_detected[s] = tatami_stats::LocalOutputBuffer<Detected_>(thread, start, len, output.subset_detected[s]);
            }
        }
    }

private:
    tatami_stats::LocalOutputBuffer<Sum_> my_sum;
    tatami_stats::LocalOutputBuffer<Detected_> my_detected;

    tatami_stats::LocalOutputBuffer<Value_> my_max_value;
    std::vector<Value_> my_holding_max_value;
    tatami_stats::LocalOutputBuffer<Index_> my_max_index;

    std::vector<tatami_stats::LocalOutputBuffer<Sum_> > my_subset_sum;
    std::vector<tatami_stats::LocalOutputBuffer<Detected_> > my_subset_detected;

public:
    Sum_* sum_data() {
        return my_sum.data();
    }

    Detected_* detected_data() {
        return my_detected.data();
    }

    Value_* max_value_data() {
        auto dptr = my_max_value.data();
        return (dptr ? dptr : my_holding_max_value.data());
    }

    Index_* max_index_data() {
        return my_max_index.data();
    }

    std::vector<Sum_*> subset_sum_data() {
        std::vector<Sum_*> output;
        output.reserve(my_subset_sum.size());
        for (auto& s : my_subset_sum) {
            output.push_back(s.data());
        }
        return output;
    }

    std::vector<Detected_*> subset_detected_data() {
        std::vector<Detected_*> output;
        output.reserve(my_subset_detected.size());
        for (auto& s : my_subset_detected) {
            output.push_back(s.data());
        }
        return output;
    }

    void transfer() {
        if (my_sum.data()) {
            my_sum.transfer();
        }
        if (my_detected.data()) {
            my_detected.transfer();
        }

        if (my_max_value.data()) {
            my_max_value.transfer();
        }
        if (my_max_index.data()) {
            my_max_index.transfer();
        }

        for (auto& s : my_subset_sum) {
            if (s.data()) {
                s.transfer();
            }
        }

        for (auto& s : my_subset_detected) {
            if (s.data()) {
                s.transfer();
            }
        }
    }
};

template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void compute_qc_running_dense(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    int num_threads)
{
    auto is_in_subset = boolify_subsets(mat.nrow(), subsets, output);

    tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
        auto NR = mat.nrow();
        auto ext = tatami::consecutive_extractor<false>(&mat, true, static_cast<Index_>(0), NR, start, len);
        std::vector<Value_> vbuffer(len);

        PerCellQcMetricsRunningBuffers<Sum_, Detected_, Value_, Index_> locals(output, thread, start, len);
        auto outt = locals.sum_data();
        auto outd = locals.detected_data();
        auto outmi = locals.max_index_data();
        auto outmc = locals.max_value_data();
        bool do_max = (outmi || outmc);
        auto outst = locals.subset_sum_data();
        auto outsd = locals.subset_detected_data();

        size_t nsubsets = subsets.size();

        for (Index_ r = 0; r < NR; ++r) {
            auto ptr = ext->fetch(vbuffer.data());

            if (outt) {
                for (Index_ i = 0; i < len; ++i) {
                    outt[i] += ptr[i];
                }
            }

            if (outd) {
                for (Index_ i = 0; i < len; ++i) {
                    outd[i] += (ptr[i] != 0);
                }
            }

            if (do_max) {
                if (r == 0) {
                    std::copy_n(ptr, len, outmc);
                    if (outmi) {
                        std::fill_n(outmi, len, 0);
                    }
                } else {
                    for (Index_ i = 0; i < len; ++i) {
                        auto& curmax = outmc[i];
                        if (curmax < ptr[i]) {
                            curmax = ptr[i];
                            if (outmi) {
                                outmi[i] = r;
                            }
                        }
                    }
                }
            }

            if (!outst.empty() || !outsd.empty()) { // protect against accessing an empty is_in_subset.
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = [&]() -> const auto& {
                        if constexpr(std::is_pointer<Subset_>::value) {
                            return subsets[s];
                        } else {
                            return is_in_subset[s];
                        }
                    }();
                    if (sub[r] == 0) {
                        continue;
                    }

                    if (!outst.empty() && outst[s]) {
                        auto current = outst[s];
                        for (Index_ i = 0; i < len; ++i) {
                            current[i] += ptr[i];
                        }
                    }

                    if (!outsd.empty() && outsd[s]) {
                        auto current = outsd[s];
                        for (Index_ i = 0; i < len; ++i) {
                            current[i] += (ptr[i] != 0);
                        }
                    }
                }
            }
        }

        locals.transfer();
    }, mat.ncol(), num_threads);
}

template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void compute_qc_running_sparse(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    int num_threads)
{
    tatami::Options opt;
    opt.sparse_ordered_index = false;
    auto is_in_subset = boolify_subsets(mat.nrow(), subsets, output);

    tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
        auto NR = mat.nrow();
        auto ext = tatami::consecutive_extractor<true>(&mat, true, static_cast<Index_>(0), NR, start, len, opt);
        std::vector<Value_> vbuffer(len);
        std::vector<Index_> ibuffer(len);

        PerCellQcMetricsRunningBuffers<Sum_, Detected_, Value_, Index_> locals(output, thread, start, len);
        auto outt = locals.sum_data();
        auto outd = locals.detected_data();
        auto outmi = locals.max_index_data();
        auto outmc = locals.max_value_data();
        bool do_max = (outmi || outmc);
        auto outst = locals.subset_sum_data();
        auto outsd = locals.subset_detected_data();

        size_t nsubsets = subsets.size();

        std::vector<Index_> last_consecutive_nonzero(do_max ? len : 0);

        for (Index_ r = 0; r < NR; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            if (outt) {
                for (Index_ i = 0; i < range.number; ++i) {
                    outt[range.index[i] - start] += range.value[i];
                }
            }

            if (outd) {
                for (Index_ i = 0; i < range.number; ++i) {
                    outd[range.index[i] - start] += (range.value[i] != 0);
                }
            }

            if (do_max) {
                if (r == 0) {
                    std::fill_n(outmc, len, 0);
                    for (Index_ i = 0; i < range.number; ++i) {
                        auto j = range.index[i] - start;
                        outmc[j] = range.value[i];
                        last_consecutive_nonzero[j] = 1; // see below
                    }
                    if (outmi) {
                        std::fill_n(outmi, len, 0);
                    }

                } else {
                    for (Index_ i = 0; i < range.number; ++i) {
                        auto j = range.index[i] - start;
                        auto& curmax = outmc[j];

                        auto val = range.value[i];
                        if (curmax < val) {
                            curmax = val;
                            if (outmi) {
                                outmi[j] = r;
                            }
                        }

                        // Getting the index of the last consecutive structural
                        // non-zero, so that we can check if zero is the max
                        // and gets its first occurrence, if necessary.
                        auto& last = last_consecutive_nonzero[j];
                        if (last == r) {
                            ++last;
                        }
                    }
                }
            }

            if (!outst.empty() || !outsd.empty()) { // protect against accessing an empty is_in_subset.
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = [&]() -> const auto& {
                        if constexpr(std::is_pointer<Subset_>::value) {
                            return subsets[s];
                        } else {
                            return is_in_subset[s];
                        }
                    }();
                    if (sub[r] == 0) {
                        continue;
                    }

                    if (!outst.empty() && outst[s]) {
                        auto current = outst[s];
                        for (Index_ i = 0; i < range.number; ++i) {
                            current[range.index[i] - start] += range.value[i];
                        }
                    }

                    if (!outsd.empty() && outsd[s]) {
                        auto current = outsd[s];
                        for (Index_ i = 0; i < range.number; ++i) {
                            current[range.index[i] - start] += (range.value[i] != 0);
                        }
                    }
                }
            }
        }

        if (do_max) {
            auto NR = mat.nrow();

            // Checking anything with non-positive maximum, and replacing it
            // with zero if there are any structural zeros.
            for (Index_ c = 0; c < len; ++c) {
                auto last_nz = last_consecutive_nonzero[c];
                if (last_nz == NR) { // i.e., no structural zeros.
                    continue;
                }

                auto& current = outmc[c];
                if (current > 0) { // doesn't defeat the current maximum.
                    continue;
                }

                current = 0;
                if (outmi) {
                    outmi[c] = last_nz;
                }
            }
        }

        locals.transfer();
    }, mat.ncol(), num_threads);
}

}
/**
 * @endcond
 */

/**
 * @brief Result store for QC metric calculations.
 * 
 * @tparam Sum_ Floating point type to store the sums.
 * @tparam Detected_ Integer type to store the number of detected cells.
 * @tparam Value_ Type of the matrix value.
 * @tparam Index_ Integer type to store the gene index.
 *
 * Meaningful instances of this object should generally be constructed by calling the `per_cell_qc_metrics()` functions.
 * Empty instances can be default-constructed as placeholders.
 */
template<typename Sum_, typename Detected_, typename Value_, typename Index_>
struct PerCellQcMetricsResults {
    /**
     * @cond
     */
    PerCellQcMetricsResults() = default;

    PerCellQcMetricsResults(size_t nsubsets) : subset_sum(nsubsets), subset_detected(nsubsets) {}
    /**
     * @endcond
     */

    /**
     * Sum of expression values for each cell.
     * Empty if `PerCellQcMetricsOptions::compute_sum` is false.
     */
    std::vector<Sum_> sum;

    /**
     * Number of detected features in each cell.
     * Empty if `PerCellQcMetricsOptions::compute_detected` is false.
     */
    std::vector<Detected_> detected;

    /**
     * Row index of the most-expressed feature in each cell.
     * On ties, the first feature is arbitrarily chosen.
     * Empty if `PerCellQcMetricsOptions::compute_max_index` is false.
     */
    std::vector<Index_> max_index;

    /**
     * Maximum value in each cell.
     * Empty if `PerCellQcMetricsOptions::compute_max_value` is false.
     */
    std::vector<Value_> max_value;

    /**
     * Sum of expression values for each feature subset in each cell.
     * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
     * Empty if there are no feature subsets or if `PerCellQcMetricsOptions::compute_subset_sum` is false.
     */
    std::vector<std::vector<Sum_> > subset_sum;

    /**
     * Number of detected features in each feature subset in each cell.
     * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
     * Empty if there are no feature subsets or if `PerCellQcMetricsOptions::compute_subset_detected` is false.
     */
    std::vector<std::vector<Detected_> > subset_detected;
};

/**
 * Given a feature-by-cell expression matrix (usually containing counts), we compute several QC metrics:
 * 
 * - The sum of expression values for each cell, which represents the efficiency of library preparation and sequencing.
 *   Low sums indicate that the library was not successfully captured.
 * - The number of detected features (i.e., with non-zero counts).
 *   This also quantifies the library preparation efficiency, but with a greater focus on capturing the transcriptional complexity.
 * - The maximum value across all features.
 *   This is useful in situations where only one feature is expected to be present, e.g., CRISPR guides, hash tags.
 * - The row index of the feature with the maximum count.
 *   If multiple features are tied for the maximum count, the earliest feature is reported.
 * - The sum of expression values in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset -
 *   most commonly, one subset will contain all genes on the mitochondrial chromosome,
 *   where higher proportions of counts in the mitochondrial subset indicate cell damage due to loss of cytoplasmic transcripts.
 *   Spike-in proportions can be interpreted in a similar manner.
 * - The number of detected features in pre-defined feature subsets.
 *   Analogous to the number of detected features for the entire feature space.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 * @tparam Sum_ Floating point type to store the sums.
 * @tparam Detected_ Integer type to store the number of detected cells.
 *
 * @param mat A **tatami** matrix, typically containing count data.
 * Rows should correspond to features (e.g., genes) while columns should correspond to cells.
 * @param[in] subsets Vector of feature subsets, where each entry represents a feature subset and may be either:
 * - A pointer to an array of length equal to `mat.nrow()` where each entry is interpretable as a boolean.
 *   This indicates whether each row in `mat` belongs to the subset.
 * - A `std::vector` containing sorted and unique row indices.
 *   This specifies the rows in `mat` that belong to the subset.
 * @param[out] output Collection of buffers in which the computed statistics are to be stored.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void per_cell_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets, 
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    const PerCellQcMetricsOptions& options)
{
    if (mat.sparse()) {
        if (mat.prefer_rows()) {
            internal::compute_qc_running_sparse(mat, subsets, output, options.num_threads);
        } else {
            internal::compute_qc_direct_sparse(mat, subsets, output, options.num_threads);
        }
    } else {
        if (mat.prefer_rows()) {
            internal::compute_qc_running_dense(mat, subsets, output, options.num_threads);
        } else {
            internal::compute_qc_direct_dense(mat, subsets, output, options.num_threads);
        }
    }
}

/**
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `vector` of indices.
 * @tparam Sum_ Floating point type to store the sums.
 * @tparam Detected_ Integer type to store the number of detected cells.
 *
 * @param mat A **tatami** matrix, typically containing count data.
 * Rows should correspond to features (e.g., genes) while columns should correspond to cells.
 * @param[in] subsets Vector of feature subsets, where each entry represents a feature subset and may be either:
 * - A pointer to an array of length equal to `mat.nrow()` where each entry is interpretable as a boolean.
 *   This indicates whether each row in `mat` belongs to the subset.
 * - A `std::vector` containing sorted and unique row indices.
 *   This specifies the rows in `mat` that belong to the subset.
 * @param options Further options.
 *
 * @return Object containing the QC metrics.
 * Not all metrics may be computed depending on `options`.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Value_, typename Index_, typename Subset_>
PerCellQcMetricsResults<Sum_, Detected_, Value_, Index_> per_cell_qc_metrics(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsOptions& options)
{
    PerCellQcMetricsResults<Sum_, Detected_, Value_, Index_> output;
    PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_> buffers;
    auto ncells = mat.ncol();

    if (options.compute_sum) {
        output.sum.resize(ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.sum = output.sum.data();
    }
    if (options.compute_detected) {
        output.detected.resize(ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.detected = output.detected.data();
    }
    if (options.compute_max_index) {
        output.max_index.resize(ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.max_index = output.max_index.data();
    }
    if (options.compute_max_value) {
        output.max_value.resize(ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.max_value = output.max_value.data();
    }

    size_t nsubsets = subsets.size();

    if (options.compute_subset_sum) {
        output.subset_sum.resize(nsubsets);
        buffers.subset_sum.resize(nsubsets);
        for (size_t s = 0; s < nsubsets; ++s) {
            output.subset_sum[s].resize(ncells
#ifdef SCRAN_QC_TEST_INIT
                , SCRAN_QC_TEST_INIT
#endif
            );
            buffers.subset_sum[s] = output.subset_sum[s].data();
        }
    }

    if (options.compute_subset_detected) {
        output.subset_detected.resize(nsubsets);
        buffers.subset_detected.resize(nsubsets);
        for (size_t s = 0; s < nsubsets; ++s) {
            output.subset_detected[s].resize(ncells
#ifdef SCRAN_QC_TEST_INIT
                , SCRAN_QC_TEST_INIT
#endif
            );
            buffers.subset_detected[s] = output.subset_detected[s].data();
        }
    }

    per_cell_qc_metrics(mat, subsets, buffers, options);
    return output;
}

}

#endif
