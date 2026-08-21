#ifndef SCRAN_QC_PER_CELL_QC_METRICS_HPP
#define SCRAN_QC_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

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
     * Whether to report the index of the feature with the maximum value for each cell.
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
     * Whether the `subsets` supplied to `per_cell_qc_metrics()` contain the row indices of the features within each subset.
     *
     * This option is only relevant if `Subset_` is not a pointer type.
     * Instead, `Subset_` is expected to be a container-like type with `operator[]`, `size`, `begin` and `end` methods.
     * 
     * - If `true`, each entry of `subsets` is assumed to be a container of unique row indices.
     *   This specifies the rows of the input matrix corresponding to the features in this subset..
     * - If `false`, each entry of `subsets` is assumed to be a container of length equal to the number of features in the input matrix. 
     *   Each container element is interpreted as a boolean indicating whether the corresponding row of the matrix belongs to the subset.
     *
     * If `Subset_` is a pointer type, this option is ignored.
     * Each entry of `subsets` is assumed to be a pointer to an array of length equal to the number features in the input matrix,
     * where each array element is treated as a boolean indicating whether the coresponding row of the matrix belongs to the subset.
     */
    bool subset_containers_have_indices = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `per_cell_qc_metrics()`.
 *
 * @tparam Sum_ Numeric type of the sums, usually floating-point.
 * If integer, this should be large enough to avoid overflow.
 * @tparam Detected_ Integer type of the number of detected cells.
 * This should be large enough to avoid integer overflow, typically set to the same type as `Index_`.
 * @tparam Value_ Type of the matrix value.
 * @tparam Index_ Integer type of the matrix index.
 */
template<typename Sum_, typename Detected_, typename Value_, typename Index_>
struct PerCellQcMetricsBuffers {
    /**
     * @cond
     */
    PerCellQcMetricsBuffers() = default;

    PerCellQcMetricsBuffers(const std::size_t nsubsets) : 
        subset_sum(sanisizer::cast<I<decltype(subset_sum.size())> >(nsubsets), NULL),
        subset_detected(sanisizer::cast<I<decltype(subset_detected.size())> >(nsubsets), NULL)
    {}
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
     * Pointer to an array of length equal to the number of cells, equivalent to `PerCellQcMetricsResults::max_value`.
     * Set to `NULL` to skip this calculation.
     */
    Value_* max_value = NULL;

    /**
     * Pointer to an array of length equal to the number of cells, equivalent to `PerCellQcMetricsResults::max_index`.
     * On ties, the first feature is arbitrarily chosen.
     * Set to `NULL` to skip this calculation.
     */
    Index_* max_index = NULL;

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
template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void per_cell_qc_metrics_direct_dense(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    const PerCellQcMetricsOptions& options
) {
    const auto NR = mat.nrow();
    const bool report_max = output.max_value || output.max_index;
    const auto nsubsets = subsets.size();
    const bool report_subsets = output.subset_sum.size() || output.subset_detected.size();

    std::optional<std::vector<std::vector<Index_> > > subset_indices;
    if (report_subsets) {
        [&](){ // use an IIFE for easy control via return.
            if constexpr(!std::is_pointer<Subset_>::value) {
                if (options.subset_containers_have_indices) {
                    return;
                }
            }

            subset_indices.emplace(sanisizer::cast<I<decltype(subset_indices->size())> >(nsubsets));
            for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                auto& current = (*subset_indices)[s];
                const auto& source = subsets[s];
                for (I<decltype(NR)> i = 0; i < NR; ++i) {
                    if (source[i]) {
                        current.push_back(i);
                    }
                }
            }
        }();
    }

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(mat, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);

        for (Index_ c = start, end = start + length; c < end; ++c) {
            auto ptr = ext->fetch(c, vbuffer.data());

            if (output.sum) {
                output.sum[c] = std::accumulate(ptr, ptr + NR, static_cast<Sum_>(0));
            }

            if (output.detected) {
                Detected_ count = 0;
                for (I<decltype(NR)> r = 0; r < NR; ++r) {
                    count += (ptr[r] != 0);
                }
                output.detected[c] = count;
            }

            if (report_max) {
                if (NR) {
                    const auto it = std::max_element(ptr, ptr + NR);
                    if (output.max_value) {
                        output.max_value[c] = *it;
                    }
                    if (output.max_index) {
                        output.max_index[c] = it - ptr;
                    }
                } else {
                    if (output.max_value) {
                        output.max_value[c] = 0;
                    }
                    if (output.max_index) {
                        output.max_index[c] = 0;
                    }
                }
            }

            if (report_subsets) {
                for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                    if constexpr(!std::is_pointer<Subset_>::value) {
                        // Seeing if the subsets are already indices, in which case we use it directly.
                        // It is assumed that there are no duplicate values here.
                        if (options.subset_containers_have_indices) {
                            const auto& sub = subsets[s];
                            if (!output.subset_sum.empty() && output.subset_sum[s]) {
                                Sum_ current = 0;
                                for (const auto r : sub) {
                                    current += ptr[r];
                                }
                                output.subset_sum[s][c] = current;
                            }
                            if (!output.subset_detected.empty() && output.subset_detected[s]) {
                                Detected_ current = 0;
                                for (const auto r : sub) {
                                    current += ptr[r] != 0;
                                }
                                output.subset_detected[s][c] = current;
                            }
                            continue;
                        }
                    }

                    const auto& sub = (*subset_indices)[s];
                    if (!output.subset_sum.empty() && output.subset_sum[s]) {
                        Sum_ current = 0;
                        for (const auto r : sub) {
                            current += ptr[r];
                        }
                        output.subset_sum[s][c] = current;
                    }
                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        Detected_ current = 0;
                        for (const auto r : sub) {
                            current += ptr[r] != 0;
                        }
                        output.subset_detected[s][c] = current;
                    }
                }
            }
        }
    }, mat.ncol(), options.num_threads);
}

template<typename Index_, typename Subset_>
std::vector<std::vector<unsigned char> > boolify_subsets(const Index_ NR, const std::vector<Subset_>& subsets) {
    const auto nsubsets = subsets.size();
    auto output = sanisizer::create<std::vector<std::vector<unsigned char> > >(nsubsets);
    for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
        auto& current = output[s];
        tatami::resize_container_to_Index_size(current, NR);
        for (const auto i : subsets[s]) {
            current[i] = 1;
        }
    }
    return output;
}

template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void per_cell_qc_metrics_direct_sparse(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    const PerCellQcMetricsOptions& options
) {
    const auto NR = mat.nrow();
    const bool report_max = output.max_value || output.max_index;
    const auto nsubsets = subsets.size();
    const bool report_subsets = output.subset_sum.size() || output.subset_detected.size();

    std::optional<std::vector<std::vector<unsigned char> > > is_in_subset;
    if (report_subsets) {
        if constexpr(!std::is_pointer<Subset_>::value) {
            if (options.subset_containers_have_indices) {
                is_in_subset = boolify_subsets(NR, subsets);
            }
        }
    }

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(mat, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

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

            if (report_max) {
                if (range.number) {
                    const auto it = std::max_element(range.value, range.value + range.number);
                    if (*it > 0 || range.number == NR) {
                        if (output.max_value) {
                            output.max_value[c] = *it;
                        }
                        if (output.max_index) {
                            output.max_index[c] = range.index[it - range.value];
                        }
                    } else {
                        if (output.max_value) {
                            output.max_value[c] = 0;
                        }
                        if (output.max_index) {
                            if (*it < 0) {
                                // Find the first structural zero. 
                                output.max_index[c] = range.number;
                                for (Index_ i = 0; i < range.number; ++i) {
                                    if (range.index[i] != i) { 
                                        output.max_index[c] = i;
                                        break;
                                    }
                                }
                            } else {
                                // Find the first structural zero that occurs before the structural non-zero with a value of zero.
                                const Index_ candidate = it - range.value; 
                                output.max_index[c] = range.index[candidate];
                                for (Index_ i = 0; i <= candidate; ++i) {
                                    if (range.index[i] != i) {
                                        output.max_index[c] = i;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (output.max_value) {
                        output.max_value[c] = 0;
                    }
                    if (output.max_index) {
                        output.max_index[c] = 0;
                    } 
                }
            }

            if (report_subsets) {
                for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                    if constexpr(!std::is_pointer<Subset_>::value) {
                        if (options.subset_containers_have_indices) {
                            const auto& sub = (*is_in_subset)[s];
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
                            continue;
                        }
                    }

                    const auto& sub = subsets[s];
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
    }, mat.ncol(), options.num_threads);
}

template<typename Value_, typename Index_, typename Subset_, typename Sum_, typename Detected_>
void per_cell_qc_metrics_running(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<Subset_>& subsets,
    const PerCellQcMetricsBuffers<Sum_, Detected_, Value_, Index_>& output,
    const PerCellQcMetricsOptions& options
) {
    const auto NR = mat.nrow();
    const auto NC = mat.ncol();
    const bool is_sparse = mat.is_sparse();

    /************************************
     *** Setting up result containers ***
     ************************************/

    const bool report_max = output.max_value || output.max_index;
    std::optional<std::vector<Value_> > tmp_max_value;
    Value_* max_value_output_ptr;
    if (report_max) {
        if (!output.max_value) {
            tmp_max_value.emplace(tatami::cast_Index_to_container_size<std::vector<Index_> >(NC));
            max_value_output_ptr = tmp_max_value->data();
        } else {
            max_value_output_ptr = output.max_value;
        }
    }

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Sum_> > > > partial_sum;
    std::optional<std::vector<std::optional<std::vector<Detected_> > > > partial_detected;
    std::optional<std::vector<std::optional<std::vector<Value_> > > > partial_max_value;
    std::optional<std::vector<std::optional<std::vector<Index_> > > > partial_max_index;
    std::optional<std::vector<std::optional<std::vector<std::vector<Sum_> > > > > partial_subset_sum;
    std::optional<std::vector<std::optional<std::vector<std::vector<Detected_> > > > > partial_subset_detected;
    if (do_parallel) {
        if (output.sum) {
            partial_sum.emplace(tatami::cast_Index_to_container_size<I<decltype(*partial_sum)> >(options.num_threads - 1));
        }
        if (output.detected) {
            partial_detected.emplace(tatami::cast_Index_to_container_size<I<decltype(*partial_detected)> >(options.num_threads - 1));
        }
        if (report_max) {
            partial_max_value.emplace(tatami::cast_Index_to_container_size<I<decltype(*partial_max_value)> >(options.num_threads - 1));
            if (output.max_index) {
                partial_max_index.emplace(tatami::cast_Index_to_container_size<I<decltype(*partial_max_index)> >(options.num_threads - 1));
            }
        }
        if (output.subset_sum.size()) {
            partial_subset_sum.emplace(tatami::cast_Index_to_container_size<I<decltype(*partial_subset_sum)> >(options.num_threads - 1));
        }
        if (output.subset_detected.size()) {
            partial_subset_detected.emplace(tatami::cast_Index_to_container_size<I<decltype(*partial_subset_detected)> >(options.num_threads - 1));
        }
    }

    // Zeroing the output arrays. 
    if (output.sum) {
        std::fill_n(output.sum, NC, 0);
    }
    if (output.detected) {
        std::fill_n(output.detected, NC, 0);
    }
    if (report_max && NR == 0) { // no need to zero if it's not empty, as it'll get filled by thread 0 upon encountering the first column.
        std::fill_n(output.max_value, NC, 0);
        if (output.max_index) {
            std::fill_n(output.max_index, NC, 0);
        }
    }
    for (const auto sptr : output.subset_sum) {
        if (sptr) {
            std::fill_n(sptr, NC, 0);
        }
    }
    for (const auto sptr : output.subset_detected) {
        if (sptr) {
            std::fill_n(sptr, NC, 0);
        }
    }

    const auto nsubsets = subsets.size();
    const bool report_subsets = output.subset_sum.size() || output.subset_detected.size();
    std::optional<std::vector<std::vector<unsigned char> > > is_in_subset;
    if (report_subsets) {
        if constexpr(!std::is_pointer<Subset_>::value) {
            if (options.subset_containers_have_indices) {
                is_in_subset = boolify_subsets(NR, subsets);
            }
        }
    }

    const auto num_used = tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
        /*********************************************************
         *** Thread-local containers to mitigate false sharing ***
         *********************************************************/

        Sum_* sum_ptr = NULL;
        std::optional<std::vector<Sum_> > sum_buffer;
        Detected_* detected_ptr = NULL;
        std::optional<std::vector<Detected_> > detected_buffer;
        Value_* max_value_ptr = NULL;
        std::optional<std::vector<Value_> > max_value_buffer;
        Index_* max_index_ptr = NULL;
        std::optional<std::vector<Index_> > max_index_buffer;

        Sum_* const * subset_sum_ptr = NULL;
        std::optional<std::vector<Sum_*> > subset_sum_ptrs;
        std::optional<std::vector<std::vector<Sum_> > > subset_sum_buffers;
        Detected_* const * subset_detected_ptr = NULL;
        std::optional<std::vector<Detected_*> > subset_detected_ptrs;
        std::optional<std::vector<std::vector<Detected_> > > subset_detected_buffers;

        if (!do_parallel || thread == 0) {
            sum_ptr = output.sum;
            detected_ptr = output.detected;
            max_value_ptr = max_value_output_ptr;
            max_index_ptr = output.max_index;
            if (output.subset_sum.size()) {
                subset_sum_ptr = output.subset_sum.data();
            }
            if (output.subset_detected.size()) {
                subset_detected_ptr = output.subset_detected.data();
            }

        } else {
            if (output.sum) {
                sum_buffer.emplace(tatami::cast_Index_to_container_size<I<decltype(*sum_buffer)> >(NC));
                sum_ptr = sum_buffer->data();
            }
            if (output.detected) {
                detected_buffer.emplace(tatami::cast_Index_to_container_size<I<decltype(*detected_buffer)> >(NC));
                detected_ptr = detected_buffer->data();
            }
            if (report_max) {
                max_value_buffer.emplace(tatami::cast_Index_to_container_size<I<decltype(*max_value_buffer)> >(NC));
                max_value_ptr = max_value_buffer->data();
                if (output.max_index) {
                    max_index_buffer.emplace(tatami::cast_Index_to_container_size<I<decltype(*max_index_buffer)> >(NC));
                    max_index_ptr = max_index_buffer->data();
                }
            }
            if (output.subset_sum.size()) {
                subset_sum_ptrs.emplace(tatami::cast_Index_to_container_size<I<decltype(*subset_sum_ptrs)> >(nsubsets));
                subset_sum_buffers.emplace(tatami::cast_Index_to_container_size<I<decltype(*subset_sum_buffers)> >(nsubsets));
                for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                    if (output.subset_sum[s]) {
                        tatami::resize_container_to_Index_size((*subset_sum_buffers)[s], NC); 
                        (*subset_sum_ptrs)[s] = (*subset_sum_buffers)[s].data();
                    } else {
                        (*subset_sum_ptrs)[s] = NULL;
                    }

                }
            }
            if (output.subset_detected.size()) {
                subset_detected_ptrs.emplace(tatami::cast_Index_to_container_size<I<decltype(*subset_detected_ptrs)> >(nsubsets));
                subset_detected_buffers.emplace(tatami::cast_Index_to_container_size<I<decltype(*subset_detected_buffers)> >(nsubsets));
                for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                    if (output.subset_detected[s]) {
                        tatami::resize_container_to_Index_size((*subset_detected_buffers)[s], NC); 
                        (*subset_detected_ptrs)[s] = (*subset_detected_buffers)[s].data();
                    } else {
                        (*subset_detected_ptrs)[s] = NULL;
                    }
                }
            }
        }

        if (is_sparse) {
            /*****************************
             *** Sparse loop over rows ***
             *****************************/

            tatami::Options opt;
            opt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, true, start, len, opt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            // nonzeros_at_start contains the number of consecutive non-zero elements at the start, i.e., from r = 0.
            std::optional<std::vector<Index_> > nonzeros_at_start;
            if (max_value_ptr) {
                nonzeros_at_start.emplace(tatami::cast_Index_to_container_size<std::vector<Index_> >(NC));
            }

            for (Index_ r = 0; r < len; ++r) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                if (sum_ptr) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        sum_ptr[range.index[i]] += range.value[i];
                    }
                }
                if (detected_ptr) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        detected_ptr[range.index[i]] += (range.value[i] != 0);
                    }
                }

                if (report_max) {
                    if (r == 0) {
                        std::fill_n(max_value_ptr, NC, 0);
                        for (Index_ i = 0; i < range.number; ++i) {
                            const auto j = range.index[i];
                            max_value_ptr[j] = range.value[i];
                            (*nonzeros_at_start)[j] = 1; 
                        }
                        if (max_index_ptr) {
                            std::fill_n(max_index_ptr, NC, start);
                        }

                    } else {
                        for (Index_ i = 0; i < range.number; ++i) {
                            const auto val = range.value[i];
                            const auto j = range.index[i]; 
                            auto& curmax = max_value_ptr[j];
                            if (curmax < val) {
                                curmax = val;
                                if (max_index_ptr) {
                                    max_index_ptr[j] = start + r;
                                }
                            }
                            auto& last = (*nonzeros_at_start)[j];
                            if (last == r) {
                                // If we don't have an unbroken run of structural non-zeros from the start, we stop incrementing.
                                ++last;
                            }
                        }
                    }
                }

                if (report_subsets) {
                    for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                        const auto in_subset = [&]() -> bool {
                            if constexpr(!std::is_pointer<Subset_>::value) {
                                if (options.subset_containers_have_indices) {
                                    return (*is_in_subset)[s][r];
                                }
                            }
                            return subsets[s][r];
                        }();
                        if (!in_subset) {
                            continue;
                        }

                        if (subset_sum_ptr && subset_sum_ptr[s]) {
                            const auto current = subset_sum_ptr[s];
                            for (Index_ i = 0; i < range.number; ++i) {
                                current[range.index[i]] += range.value[i];
                            }
                        }

                        if (subset_detected_ptr && subset_detected_ptr[s]) {
                            const auto current = subset_detected_ptr[s];
                            for (Index_ i = 0; i < range.number; ++i) {
                                current[range.index[i]] += (range.value[i] != 0);
                            }
                        }
                    }
                }
            }

            if (report_max) {
                // Checking anything with non-positive maximum, and replacing it with zero if there are any structural zeros.
                for (Index_ c = 0; c < NC; ++c) {
                    const auto last_nz = (*nonzeros_at_start)[c];
                    if (last_nz == len) { // i.e., no structural zeros.
                        continue;
                    }
                    auto& current = max_value_ptr[c];
                    if (current > 0) { // doesn't defeat the current maximum.
                        continue;
                    }

                    if (current < 0) {
                        current = 0;
                        if (max_index_ptr) {
                            max_index_ptr[c] = start + last_nz;
                        }
                    } else {
                        // Sometimes, structural non-zeros have a value of zero.
                        // This check ensures that we return the first occurrence of any zero.
                        if (max_index_ptr) {
                            const Index_ first_structural_zero = start + last_nz;
                            if (first_structural_zero < max_index_ptr[c]) {
                                max_index_ptr[c] = first_structural_zero;
                            }
                        }
                    }
                }
            }

        } else {
            /****************************
             *** Dense loop over rows ***
             ****************************/

            auto ext = tatami::consecutive_extractor<false>(mat, true, start, len);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            for (Index_ r = 0; r < len; ++r) {
                auto ptr = ext->fetch(vbuffer.data());

                if (sum_ptr) {
                    for (Index_ i = 0; i < NC; ++i) {
                        sum_ptr[i] += ptr[i];
                    }
                }

                if (detected_ptr) {
                    for (Index_ i = 0; i < NC; ++i) {
                        detected_ptr[i] += (ptr[i] != 0);
                    }
                }

                if (report_max) {
                    if (r == 0) {
                        std::copy_n(ptr, NC, max_value_ptr);
                        if (max_index_ptr) {
                            std::fill_n(max_index_ptr, NC, start);
                        }
                    } else {
                        for (Index_ i = 0; i < NC; ++i) {
                            auto& curmax = max_value_ptr[i];
                            if (curmax < ptr[i]) {
                                curmax = ptr[i];
                                if (max_index_ptr) {
                                    max_index_ptr[i] = start + r;
                                }
                            }
                        }
                    }
                }

                if (report_subsets) {
                    for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                        const auto in_subset = [&]() -> bool {
                            if constexpr(!std::is_pointer<Subset_>::value) {
                                if (options.subset_containers_have_indices) {
                                    return (*is_in_subset)[s][r];
                                }
                            }
                            return subsets[s][r];
                        }();
                        if (!in_subset) {
                            continue;
                        }

                        if (subset_sum_ptr && subset_sum_ptr[s]) {
                            const auto current = subset_sum_ptr[s];
                            for (Index_ i = 0; i < NC; ++i) {
                                current[i] += ptr[i];
                            }
                        }

                        if (subset_detected_ptr && subset_detected_ptr[s]) {
                            const auto current = subset_detected_ptr[s];
                            for (Index_ i = 0; i < NC; ++i) {
                                current[i] += (ptr[i] != 0);
                            }
                        }
                    }
                }
            }
        }

        /********************************************
         *** Migrate results to serial containers ***
         ********************************************/

        if (do_parallel) {
            if (thread > 0) {
                if (output.sum) {
                    (*partial_sum)[thread - 1] = std::move(sum_buffer);
                }
                if (output.detected) {
                    (*partial_detected)[thread - 1] = std::move(detected_buffer);
                }
                if (report_max) {
                    (*partial_max_value)[thread - 1] = std::move(max_value_buffer);
                    if (output.max_index) {
                        (*partial_max_index)[thread - 1] = std::move(max_index_buffer);
                    }
                }
                if (output.subset_sum.size()) {
                    (*partial_subset_sum)[thread - 1] = std::move(subset_sum_buffers);
                }
                if (output.subset_detected.size()) {
                    (*partial_subset_detected)[thread - 1] = std::move(subset_detected_buffers);
                }
            }
        }
    }, NR, options.num_threads);

    /************************************
     *** Reduction into output arrays ***
     ************************************/

    if (do_parallel) {
        if (output.sum) {
            for (int u = 1; u < num_used; ++u) {
                const auto& cursum = *((*partial_sum)[u - 1]);
                for (Index_ c = 0; c < NC; ++c) {
                    output.sum[c] += cursum[c];
                }
            }
        }

        if (output.detected) {
            for (int u = 1; u < num_used; ++u) {
                const auto& curdetected = *((*partial_detected)[u - 1]);
                for (Index_ c = 0; c < NC; ++c) {
                    output.detected[c] += curdetected[c];
                }
            }
        }

        // All used threads will have processed non-empty ranges, so we don't need to worry about the validity of the maxima from each thread.
        if (output.max_index) {
            for (int u = 1; u < num_used; ++u) {
                const auto& curmaxval = *((*partial_max_value)[u - 1]);
                const auto& curmaxidx = *((*partial_max_index)[u - 1]);
                for (Index_ c = 0; c < NC; ++c) {
                    if (curmaxval[c] > max_value_output_ptr[c]) {
                        max_value_output_ptr[c] = curmaxval[c];
                        output.max_index[c] = curmaxidx[c];
                    }
                }
            }
        } else if (output.max_value) {
            for (int u = 1; u < num_used; ++u) {
                const auto& curmaxval = *((*partial_max_value)[u - 1]);
                for (Index_ c = 0; c < NC; ++c) {
                    if (curmaxval[c] > output.max_value[c]) {
                        output.max_value[c] = curmaxval[c];
                    }
                }
            }
        }

        if (output.subset_sum.size()) {
            for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                const auto outptr = output.subset_sum[s];
                if (outptr == NULL) {
                    continue;
                }
                for (int u = 1; u < num_used; ++u) {
                    const auto& cursubset = (*((*partial_subset_sum)[u - 1]))[s];
                    for (Index_ c = 0; c < NC; ++c) {
                        outptr[c] += cursubset[c];
                    }
                }
            }
        }

        if (output.subset_detected.size()) {
            for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
                const auto outptr = output.subset_detected[s];
                if (outptr == NULL) {
                    continue;
                }
                for (int u = 1; u < num_used; ++u) {
                    const auto& cursubset = (*((*partial_subset_detected)[u - 1]))[s];
                    for (Index_ c = 0; c < NC; ++c) {
                        outptr[c] += cursubset[c];
                    }
                }
            }
        }
    }
}
/**
 * @endcond
 */

/**
 * @brief Result store for QC metric calculations.
 * 
 * @tparam Sum_ Numeric type of the sums, typically floating-point.
 * If integer, this should be large enough to avoid overflow.
 * @tparam Detected_ Integer type of the number of detected cells.
 * This should be large enough to avoid integer overflow, typically set to the same type as `Index_`.
 * @tparam Value_ Type of the matrix value.
 * @tparam Index_ Integer type of the gene index.
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

    PerCellQcMetricsResults(const std::size_t nsubsets) : 
        subset_sum(sanisizer::cast<I<decltype(subset_sum.size())> >(nsubsets)),
        subset_detected(sanisizer::cast<I<decltype(subset_detected.size())> >(nsubsets))
    {}
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
     * Maximum value in each cell.
     * Empty if `PerCellQcMetricsOptions::compute_max_value` is false.
     */
    std::vector<Value_> max_value;

    /**
     * Row index of the most-expressed feature in each cell.
     * On ties, the first feature is arbitrarily chosen.
     * Empty if either `PerCellQcMetricsOptions::compute_max_index` or `PerCellQcMetricsOptions::compute_max_value` is false.
     */
    std::vector<Index_> max_index;

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
 * Given a feature-by-cell expression matrix (usually containing non-negative counts), we compute several QC metrics:
 * 
 * - The sum of expression values for each cell, which represents the efficiency of library preparation and sequencing.
 *   Low sums indicate that the library was not successfully captured.
 * - The number of detected features, i.e., with non-zero counts.
 *   This also quantifies the library preparation efficiency, but with a greater focus on capturing the transcriptional complexity.
 * - The maximum value across all features.
 *   This is useful in situations where only one feature is expected to be present, e.g., CRISPR guides, hash tags.
 * - The row index of the feature with the maximum count.
 *   If multiple features are tied for the maximum count, the earliest feature is reported.
 * - The sum of expression values in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset -
 *   for example, one subset for RNA data will typically contain all genes on the mitochondrial chromosome,
 *   where higher proportions of counts in the mitochondrial subset indicate cell damage due to loss of cytoplasmic transcripts.
 *   Spike-in proportions can be interpreted in a similar manner.
 * - The number of detected features in pre-defined feature subsets.
 *   Analogous to the number of detected features for the entire feature space.
 *
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a container of booleans/indices,
 * see `PerCellQcMetricsOptions::subset_containers_have_indices` for more details.
 * @tparam Sum_ Numeric type of the sums, typically floating-point.
 * If integer, this should be large enough to avoid overflow.
 * @tparam Detected_ Integer type of the number of detected cells.
 * This should be large enough to avoid integer overflow, typically set to the same type as `Index_`.
 *
 * @param mat A matrix of non-negative counts.
 * Rows should correspond to features (e.g., genes) while columns should correspond to cells.
 * @param[in] subsets Vector of feature subsets, where each entry represents a feature subset and may be either:
 * - A pointer to an array of length equal to `mat.nrow()`.
 *   Each entry is interpreted as a boolean that indicates whether the corresponding row in `mat` belongs to this subset.
 * - A container (e.g., `std::vector`) of any length containing unique row indices, if `PerCellQcMetricsOptions::subset_containers_have_indices = true`.
 *   This specifies the rows in `mat` that belong to this subset.
 * - A container (e.g., `std::vector`) of length equal to `mat.nrow()`, if `PerCellQcMetricsOptions::subset_containers_have_indices = false`.
 *   Each element is interpreted as a boolean that indicates whether the corresponding row in `mat` belongs to this subset.
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
    if (mat.prefer_rows()) {
        per_cell_qc_metrics_running(mat, subsets, output, options);
    } else {
        if (mat.sparse()) {
            per_cell_qc_metrics_direct_sparse(mat, subsets, output, options);
        } else {
            per_cell_qc_metrics_direct_dense(mat, subsets, output, options);
        }
    }
}

/**
 * @tparam Value_ Type of matrix value.
 * @tparam Index_ Type of the matrix indices.
 * @tparam Subset_ Either a pointer to an array of booleans or a `std::vector` of indices.
 * @tparam Sum_ Numeric type of the sums, typically floating-point.
 * If integer, this should be large enough to avoid overflow.
 * @tparam Detected_ Integer type of the number of detected cells.
 * This should be large enough to avoid integer overflow, typically set to the same type as `Index_`.
 *
 * @param mat A matrix of non-negative counts.
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
    const auto ncells = mat.ncol();

    if (options.compute_sum) {
        tatami::resize_container_to_Index_size(output.sum, ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.sum = output.sum.data();
    }

    if (options.compute_detected) {
        tatami::resize_container_to_Index_size(output.detected, ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.detected = output.detected.data();
    }

    if (options.compute_max_value) {
        tatami::resize_container_to_Index_size(output.max_value, ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.max_value = output.max_value.data();
    }
    if (options.compute_max_index) {
        tatami::resize_container_to_Index_size(output.max_index, ncells
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffers.max_index = output.max_index.data();
    }

    const auto nsubsets = subsets.size();

    if (options.compute_subset_sum) {
        sanisizer::resize(output.subset_sum, nsubsets);
        sanisizer::resize(buffers.subset_sum, nsubsets);
        for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
            tatami::resize_container_to_Index_size(output.subset_sum[s], ncells
#ifdef SCRAN_QC_TEST_INIT
                , SCRAN_QC_TEST_INIT
#endif
            );
            buffers.subset_sum[s] = output.subset_sum[s].data();
        }
    }

    if (options.compute_subset_detected) {
        sanisizer::resize(output.subset_detected, nsubsets);
        sanisizer::resize(buffers.subset_detected, nsubsets);
        for (I<decltype(nsubsets)> s = 0; s < nsubsets; ++s) {
            tatami::resize_container_to_Index_size(output.subset_detected[s], ncells
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
