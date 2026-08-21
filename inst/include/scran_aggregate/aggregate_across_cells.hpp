#ifndef SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP
#define SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP

#include <algorithm>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>
#include <optional>

#include "tatami/tatami.hpp"
#include "quickstats/quickstats.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file aggregate_across_cells.hpp
 * @brief Aggregate expression values across cells.
 */

namespace scran_aggregate {

/**
 * @brief Options for `aggregate_across_cells()`.
 */
struct AggregateAcrossCellsOptions {
    /**
     * Whether to compute the sum of expression within each group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_sum = true;

    /**
     * Whether to compute the number of detected cells within each group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_detected = true;

    /**
     * Whether to compute the median expression withine ach group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_median = false; // false by default as we usually don't need this.

    /**
     * Number of threads to use. 
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `aggregate_across_cells()`.
 * @tparam Sum_ Numeric type of the sum, typically floating-point.
 * If integer, this should be large enough to avoid integer overflow.
 * @tparam Detected_ Type of the number of detected cells, usually integer.
 * This should be large enough to avoid integer overflow.
 * @tparam Float_ Floating-point type to be used for other statistics, e.g., median.
 */
template <typename Sum_, typename Detected_, typename Float_>
struct AggregateAcrossCellsBuffers {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the number of genes,
     * to be filled with the summed expression across all cells in the corresponding group for each gene.
     *
     * If this is empty, the sums for each group are not computed.
     */
    std::vector<Sum_*> sum;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the number of genes,
     * to be filled with the number of cells in the corresponding group with detected expression for each gene.
     * 
     * If this is empty, the number of detected cells for each group is not computed.
     */
    std::vector<Detected_*> detected;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the number of genes,
     * to be filled with the median expression across all cells in the corresponding group for each gene.
     * 
     * If this is empty, the median for each group is not computed.
     */
    std::vector<Float_*> median;
};

/**
 * @brief Results of `aggregate_across_cells()`.
 * @tparam Sum_ Numeric type of the sum, typically floating-point.
 * If integer, this should be large enough to avoid integer overflow.
 * @tparam Detected_ Type of the number of detected cells, usually integer.
 * This should be large enough to avoid integer overflow.
 * @tparam Float_ Floating-point type to be used for other statistics, e.g., median.
 */
template <typename Sum_, typename Detected_, typename Float_>
struct AggregateAcrossCellsResults {
    /**
     * Vector of length equal to the number of groups.
     * Each inner vector is of length equal to the number of genes.
     * Each entry contains the summed expression across all cells in the corresponding group for each gene.
     *
     * If `AggregateAcrossCellsOptions::compute_sum = false`, this vector is empty.
     */
    std::vector<std::vector<Sum_> > sum;

    /**
     * Vector of length equal to the number of groups.
     * Each inner vector is of length equal to the number of genes.
     * Each entry contains the number of cells in the corresponding group with detected expression for each gene.
     *
     * If `AggregateAcrossCellsOptions::compute_detected = false`, this vector is empty.
     */
    std::vector<std::vector<Detected_> > detected;

    /**
     * Vector of length equal to the number of groups.
     * Each inner vector is of length equal to the number of genes.
     * Each entry contains the median expression across all cells in the corresponding group for each gene.
     *
     * If `AggregateAcrossCellsOptions::compute_median = false`, this vector is empty.
     */
    std::vector<std::vector<Float_> > median;
};

/**
 * @cond
 */
template<typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_, typename Float_>
void aggregate_across_cells_by_row(
    const tatami::Matrix<Data_, Index_>& p,
    const Group_* const group,
    const std::size_t num_groups,
    const AggregateAcrossCellsBuffers<Sum_, Detected_, Float_>& buffers,
    const AggregateAcrossCellsOptions& options
) {
    const bool is_sparse = p.is_sparse();
    const auto NC = p.ncol();

    std::optional<std::vector<Index_> > group_sizes;
    if (!buffers.median.empty()) {
        group_sizes.emplace(sanisizer::cast<I<decltype(group_sizes->size())> >(num_groups));
        for (Index_ c = 0; c < NC; ++c) {
            (*group_sizes)[group[c]] += 1;
        }
    }

    const auto nsum = buffers.sum.size();
    if (nsum) {
        assert(nsum == num_groups);
    }

    const auto ndetected = buffers.detected.size();
    if (ndetected) {
        assert(ndetected == num_groups);
    }

    const auto nmedian = buffers.median.size();
    if (nmedian) {
        assert(nmedian == num_groups);
    }

    tatami::parallelize([&](const int, const Index_ s, const Index_ l) -> void {
        // Create buffers to reduce false sharing during summations.
        std::optional<std::vector<Sum_> > tmp_sum;
        if (nsum) {
            tmp_sum.emplace(sanisizer::cast<I<decltype(tmp_sum->size())> >(nsum));
        }

        std::optional<std::vector<Detected_> > tmp_detected;
        if (ndetected) {
            tmp_detected.emplace(sanisizer::cast<I<decltype(tmp_detected->size())> >(ndetected));
        }

        std::optional<std::vector<std::vector<Float_> > > tmp_median;
        if (nmedian) {
            tmp_median.emplace(sanisizer::cast<I<decltype(tmp_median->size())> >(nmedian));
            for (I<decltype(nmedian)> l = 0; l < nmedian; ++l) {
                sanisizer::reserve((*tmp_median)[l], (*group_sizes)[l]);
            }
        }

        if (is_sparse) {
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(p, true, s, l, opt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            for (Index_ x = s, end = s + l; x < end; ++x) {
                const auto row = ext->fetch(vbuffer.data(), ibuffer.data());

                if (nsum) {
                    std::fill(tmp_sum->begin(), tmp_sum->end(), 0);
                    for (Index_ j = 0; j < row.number; ++j) {
                        (*tmp_sum)[group[row.index[j]]] += row.value[j];
                    }
                    for (I<decltype(nsum)> l = 0; l < nsum; ++l) {
                        buffers.sum[l][x] = (*tmp_sum)[l];
                    }
                }

                if (ndetected) {
                    std::fill(tmp_detected->begin(), tmp_detected->end(), 0);
                    for (Index_ j = 0; j < row.number; ++j) {
                        (*tmp_detected)[group[row.index[j]]] += (row.value[j] > 0);
                    }
                    for (I<decltype(ndetected)> l = 0; l < ndetected; ++l) {
                        buffers.detected[l][x] = (*tmp_detected)[l];
                    }
                }

                if (nmedian) {
                    quickstats::MedianOptions<Float_> medopt;
                    medopt.placeholder = std::numeric_limits<Float_>::quiet_NaN();
                    for (Index_ j = 0; j < row.number; ++j) {
                        (*tmp_median)[group[row.index[j]]].push_back(row.value[j]);
                    }
                    for (I<decltype(ndetected)> l = 0; l < nmedian; ++l) {
                        auto& current = (*tmp_median)[l];
                        buffers.median[l][x] = quickstats::median<Float_>((*group_sizes)[l], current.size(), current.data(), medopt);
                        current.clear();
                    }
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(p, true, s, l);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NC);
            for (Index_ x = s, end = s + l; x < end; ++x) {
                const auto row = ext->fetch(vbuffer.data());

                if (nsum) {
                    std::fill(tmp_sum->begin(), tmp_sum->end(), 0);
                    for (Index_ j = 0; j < NC; ++j) {
                        (*tmp_sum)[group[j]] += row[j];
                    }
                    for (I<decltype(nsum)> l = 0; l < nsum; ++l) {
                        buffers.sum[l][x] = (*tmp_sum)[l];
                    }
                }

                if (ndetected) {
                    std::fill(tmp_detected->begin(), tmp_detected->end(), 0);
                    for (Index_ j = 0; j < NC; ++j) {
                        (*tmp_detected)[group[j]] += (row[j] > 0);
                    }
                    for (I<decltype(ndetected)> l = 0; l < ndetected; ++l) {
                        buffers.detected[l][x] = (*tmp_detected)[l];
                    }
                }

                if (nmedian) {
                    quickstats::MedianOptions<Float_> medopt;
                    medopt.placeholder = std::numeric_limits<Float_>::quiet_NaN();
                    for (Index_ j = 0; j < NC; ++j) {
                        (*tmp_median)[group[j]].push_back(row[j]);
                    }
                    for (I<decltype(ndetected)> l = 0; l < nmedian; ++l) {
                        auto& current = (*tmp_median)[l];
                        buffers.median[l][x] = quickstats::median<Float_>(current.size(), current.data(), medopt);
                        current.clear();
                    }
                }
            }
        }

    }, p.nrow(), options.num_threads);
}

template<typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_, typename Float_>
void aggregate_across_cells_by_column(
    const tatami::Matrix<Data_, Index_>& p,
    const Group_* const group,
    const std::size_t num_groups,
    const AggregateAcrossCellsBuffers<Sum_, Detected_, Float_>& buffers,
    const AggregateAcrossCellsOptions& options
) {
    const auto is_sparse = p.is_sparse();
    const auto NR = p.nrow();
    assert(buffers.median.empty());
    const bool do_parallel = options.num_threads > 1;

    const auto nsum = buffers.sum.size();
    std::optional<std::vector<std::optional<std::vector<std::vector<Float_> > > > > per_thread_sum;
    if (nsum) {
        assert(nsum == num_groups);
        for (std::size_t g = 0; g < num_groups; ++g) {
            std::fill_n(buffers.sum[g], NR, 0);
        }
        if (do_parallel) {
            per_thread_sum.emplace(sanisizer::cast<I<decltype(per_thread_sum->size())> >(options.num_threads - 1));
        }
    }

    const auto ndetected = buffers.detected.size();
    std::optional<std::vector<std::optional<std::vector<std::vector<Detected_> > > > > per_thread_detected;
    if (ndetected) {
        assert(ndetected == num_groups);
        for (std::size_t g = 0; g < num_groups; ++g) {
            std::fill_n(buffers.detected[g], NR, 0);
        }
        if (do_parallel) {
            per_thread_detected.emplace(sanisizer::cast<I<decltype(per_thread_detected->size())> >(options.num_threads - 1));
        }
    }

    const auto nused = tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        std::optional<std::vector<std::vector<Float_> > > tmp_sum;
        std::optional<std::vector<Float_*> > tmp_sum_ptrs;
        std::optional<std::vector<std::vector<Detected_> > > tmp_detected;
        std::optional<std::vector<Detected_*> > tmp_detected_ptrs;

        Float_* const * sum_ptrs = NULL;
        Detected_* const * det_ptrs = NULL;
        if (t > 0) {
            if (nsum) {
                tmp_sum.emplace(sanisizer::cast<I<decltype(tmp_sum->size())> >(num_groups));
                tmp_sum_ptrs.emplace(sanisizer::cast<I<decltype(tmp_sum->size())> >(num_groups));
                for (std::size_t g = 0; g < num_groups; ++g) {
                    tatami::resize_container_to_Index_size((*tmp_sum)[g], NR);
                    (*tmp_sum_ptrs)[g] = (*tmp_sum)[g].data();
                }
                sum_ptrs = tmp_sum_ptrs->data();
            }
            if (ndetected) {
                tmp_detected.emplace(sanisizer::cast<I<decltype(tmp_detected->size())> >(num_groups));
                tmp_detected_ptrs.emplace(sanisizer::cast<I<decltype(tmp_detected->size())> >(num_groups));
                for (std::size_t g = 0; g < num_groups; ++g) {
                    tatami::resize_container_to_Index_size((*tmp_detected)[g], NR);
                    (*tmp_detected_ptrs)[g] = (*tmp_detected)[g].data();
                }
                det_ptrs = tmp_detected_ptrs->data();
            }

        } else {
            if (nsum) {
                sum_ptrs = buffers.sum.data();
            }
            if (ndetected) {
                det_ptrs = buffers.detected.data();
            }
        }

        if (is_sparse) {
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(p, false, start, length, opt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

            for (Index_ x = 0; x < length; ++x) {
                const auto col = ext->fetch(vbuffer.data(), ibuffer.data());
                const auto curgroup = group[start + x];

                if (nsum) {
                    const auto cursum = sum_ptrs[curgroup];
                    for (Index_ i = 0; i < col.number; ++i) {
                        cursum[col.index[i]] += col.value[i];
                    }
                }

                if (ndetected) {
                    const auto curdetected = det_ptrs[curgroup];
                    for (Index_ i = 0; i < col.number; ++i) {
                        curdetected[col.index[i]] += (col.value[i] > 0);
                    }
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(p, false, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NR);

            for (Index_ x = 0; x < length; ++x) {
                const auto col = ext->fetch(vbuffer.data());
                const auto curgroup = group[start + x];

                if (nsum) {
                    const auto cursum = sum_ptrs[curgroup];
                    for (Index_ i = 0; i < NR; ++i) {
                        cursum[i] += col[i];
                    }
                }

                if (ndetected) {
                    const auto curdetected = det_ptrs[curgroup];
                    for (Index_ i = 0; i < NR; ++i) {
                        curdetected[i] += (col[i] > 0);
                    }
                }
            }
        }

        if (t > 0) {
            if (nsum) {
                (*per_thread_sum)[t - 1] = std::move(tmp_sum);
            }
            if (ndetected) {
                (*per_thread_detected)[t - 1] = std::move(tmp_detected);
            }
        }
    }, p.ncol(), options.num_threads);

    if (do_parallel) {
        if (nsum) {
            for (std::size_t g = 0; g < num_groups; ++g) {
                const auto out = buffers.sum[g];
                for (int u = 1; u < nused; ++u) {
                    const auto ptrs = (*((*per_thread_sum)[u - 1]))[g];
                    for (Index_ r = 0; r < NR; ++r) {
                        out[r] += ptrs[r];
                    }
                }
            }
        }

        if (ndetected) {
            for (std::size_t g = 0; g < num_groups; ++g) {
                const auto out = buffers.detected[g];
                for (int u = 1; u < nused; ++u) {
                    const auto ptrs = (*((*per_thread_detected)[u - 1]))[g];
                    for (Index_ r = 0; r < NR; ++r) {
                        out[r] += ptrs[r];
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
 * Aggregate expression values across groups of cells for each gene.
 * We report the sum of expression values, the number of cells with detected (i.e., positive) expression values, and the median of expression values in each group.
 * This is typically used to create pseudo-bulk expression profiles for cluster/sample combinations.
 * Expression values are generally expected to be counts so that the sums can be used as if they were counts from bulk data, e.g., for differential analyses with **edgeR**.
 *
 * @tparam Data_ Numeric type of data in the input matrix.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Group_ Integer type of the group assignments.
 * @tparam Sum_ Numeric type of the sum, typically floating-point.
 * If integer, it should be large enough to avoid overflow.
 * @tparam Detected_ Numeric type (usually integer) of the number of detected cells. 
 * This should be large enough to avoid integer overflow, so setting it to be the same as `Index_` is a safe choice.
 * @tparam Float_ Floating-point type to be used for other statistics, e.g., median.
 *
 * @param input The input matrix, usually containing non-negative counts.
 * Rows are features and columns are cells.
 * @param[in] group Pointer to an array of length equal to the number of columns of `input`, containing the assigned group for each cell.
 * All entries should be integers in `[0, num_groups)`.
 * @param num_groups Number of groups.
 * @param[out] buffers Pre-allocated buffers in which to store the computed statistics. 
 * @param options Further options.
 */
template<typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_, typename Float_>
void aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Group_* const group,
    const std::size_t num_groups,
    const AggregateAcrossCellsBuffers<Sum_, Detected_, Float_>& buffers,
    const AggregateAcrossCellsOptions& options
) {
    if (input.prefer_rows() || !buffers.median.empty()) {
        aggregate_across_cells_by_row(input, group, num_groups, buffers, options);
    } else {
        aggregate_across_cells_by_column(input, group, num_groups, buffers, options);
    }
} 

/**
 * Overload of `aggregate_across_cells()` that allocates memory for the results.
 *
 * @tparam Sum_ Numeric type of the sum, typically floating-point.
 * If integer, it should be large enough to avoid overflow.
 * @tparam Detected_ Numeric type (usually integer) of the number of detected cells. 
 * This should be large enough to avoid integer overflow, so setting it to be the same as `Index_` is a safe choice.
 * @tparam Float_ Floating-point type to be used for other statistics, e.g., median.
 * @tparam Data_ Type of data in the input matrix, should be numeric.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Group_ Integer type of the group assignments.
 *
 * @param input The input matrix, usually containing non-negative counts.
 * Rows are features and columns are cells.
 * @param[in] group Pointer to an array of length equal to the number of columns of `input`, containing the assigned group for each cell.
 * All entries should be integers in `[0, num_groups)`.
 * @param num_groups Number of groups.
 * @param options Further options.
 *
 * @return Results of the aggregation, where the available statistics depend on `AggregateAcrossCellsOptions`.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Float_ = double, typename Data_, typename Index_, typename Group_>
AggregateAcrossCellsResults<Sum_, Detected_, Float_> aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Group_* const group,
    const std::size_t num_groups,
    const AggregateAcrossCellsOptions& options
) {
    const Index_ NR = input.nrow();

    AggregateAcrossCellsResults<Sum_, Detected_, Float_> output;
    AggregateAcrossCellsBuffers<Sum_, Detected_, Float_> buffers;

    if (options.compute_sum) {
        sanisizer::resize(output.sum, num_groups);
        sanisizer::resize(buffers.sum, num_groups);
        for (I<decltype(num_groups)> l = 0; l < num_groups; ++l) {
            auto& cursum = output.sum[l];
            tatami::resize_container_to_Index_size<I<decltype(cursum)>>(cursum, NR
#ifdef SCRAN_AGGREGATE_TEST_INIT
                , SCRAN_AGGREGATE_TEST_INIT
#endif
            );
            buffers.sum[l] = cursum.data();
        }
    }

    if (options.compute_detected) {
        sanisizer::resize(output.detected, num_groups);
        sanisizer::resize(buffers.detected, num_groups);
        for (I<decltype(num_groups)> l = 0; l < num_groups; ++l) {
            auto& curdet = output.detected[l];
            tatami::resize_container_to_Index_size<I<decltype(curdet)>>(curdet, NR
#ifdef SCRAN_AGGREGATE_TEST_INIT
                , SCRAN_AGGREGATE_TEST_INIT
#endif
            );
            buffers.detected[l] = curdet.data();
        }
    }

    if (options.compute_median) {
        sanisizer::resize(output.median, num_groups);
        sanisizer::resize(buffers.median, num_groups);
        for (I<decltype(num_groups)> l = 0; l < num_groups; ++l) {
            auto& curmed = output.median[l];
            tatami::resize_container_to_Index_size<I<decltype(curmed)>>(curmed, NR
#ifdef SCRAN_AGGREGATE_TEST_INIT
                , SCRAN_AGGREGATE_TEST_INIT
#endif
            );
            buffers.median[l] = curmed.data();
        }
    }


    aggregate_across_cells(input, group, num_groups, buffers, options);
    return output;
} 

}

#endif
