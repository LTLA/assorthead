#ifndef SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP
#define SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP

#include <algorithm>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
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
    bool compute_sums = true;

    /**
     * Whether to compute the number of detected cells within each group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_detected = true;

    /**
     * Whether to compute the median expression withine ach group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_medians = false; // false by default as we usually don't need this.

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
    std::vector<Sum_*> sums;

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
    std::vector<Float_*> medians;
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
     * If `AggregateAcrossCellsOptions::compute_sums = false`, this vector is empty.
     */
    std::vector<std::vector<Sum_> > sums;

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
    std::vector<std::vector<Float_> > medians;
};

/**
 * @cond
 */
template<bool sparse_, typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_, typename Float_>
void aggregate_across_cells_by_row(
    const tatami::Matrix<Data_, Index_>& p,
    const Group_* const group,
    const AggregateAcrossCellsBuffers<Sum_, Detected_, Float_>& buffers,
    const AggregateAcrossCellsOptions& options
) {
    tatami::Options opt;
    opt.sparse_ordered_index = false;

    std::optional<std::vector<Index_> > group_sizes;
    const auto NC = p.ncol();
    if (!buffers.medians.empty()) {
        group_sizes = tatami_stats::tabulate_groups(group, NC);
    }

    tatami::parallelize([&](const int, const Index_ s, const Index_ l) -> void {
        auto ext = tatami::consecutive_extractor<sparse_>(p, true, s, l, opt);

        std::vector<Sum_> tmp_sums;
        const auto nsums = buffers.sums.size();
        if (nsums) {
            sanisizer::resize(tmp_sums, nsums);
        }

        std::vector<Detected_> tmp_detected;
        const auto ndetected = buffers.detected.size();
        if (ndetected) {
            sanisizer::resize(tmp_detected, ndetected);
        }

        std::vector<std::vector<Float_> > tmp_medians;
        const auto nmedians = buffers.medians.size();
        if (nmedians) {
            sanisizer::resize(tmp_medians, nmedians);
            for (I<decltype(nmedians)> l = 0; l < nmedians; ++l) {
                sanisizer::reserve(tmp_medians[l], (*group_sizes)[l]);
            }
        }

        const auto NC = p.ncol();
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NC);
        auto ibuffer = [&]{
            if constexpr(sparse_) {
                return tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
            } else {
                return false;
            }
        }();

        for (Index_ x = s, end = s + l; x < end; ++x) {
            const auto row = [&]{
                if constexpr(sparse_) {
                    return ext->fetch(vbuffer.data(), ibuffer.data());
                } else {
                    return ext->fetch(vbuffer.data());
                }
            }();

            if (nsums) {
                std::fill(tmp_sums.begin(), tmp_sums.end(), 0);

                if constexpr(sparse_) {
                    for (Index_ j = 0; j < row.number; ++j) {
                        tmp_sums[group[row.index[j]]] += row.value[j];
                    }
                } else {
                    for (Index_ j = 0; j < NC; ++j) {
                        tmp_sums[group[j]] += row[j];
                    }
                }

                // Computing before transferring for more cache-friendliness.
                for (I<decltype(nsums)> l = 0; l < nsums; ++l) {
                    buffers.sums[l][x] = tmp_sums[l];
                }
            }

            if (ndetected) {
                std::fill(tmp_detected.begin(), tmp_detected.end(), 0);

                if constexpr(sparse_) {
                    for (Index_ j = 0; j < row.number; ++j) {
                        tmp_detected[group[row.index[j]]] += (row.value[j] > 0);
                    }
                } else {
                    for (Index_ j = 0; j < NC; ++j) {
                        tmp_detected[group[j]] += (row[j] > 0);
                    }
                }

                for (I<decltype(ndetected)> l = 0; l < ndetected; ++l) {
                    buffers.detected[l][x] = tmp_detected[l];
                }
            }

            if (nmedians) {
                if constexpr(sparse_) {
                    for (Index_ j = 0; j < row.number; ++j) {
                        tmp_medians[group[row.index[j]]].push_back(row.value[j]);
                    }
                    for (I<decltype(ndetected)> l = 0; l < nmedians; ++l) {
                        auto& current = tmp_medians[l];
                        buffers.medians[l][x] = tatami_stats::medians::direct<Float_>(current.data(), static_cast<Index_>(current.size()), (*group_sizes)[l], false);
                        current.clear();
                    }

                } else {
                    for (Index_ j = 0; j < NC; ++j) {
                        tmp_medians[group[j]].push_back(row[j]);
                    }
                    for (I<decltype(ndetected)> l = 0; l < nmedians; ++l) {
                        auto& current = tmp_medians[l];
                        buffers.medians[l][x] = tatami_stats::medians::direct(current.data(), current.size(), false);
                        current.clear();
                    }
                }
            }
        }
    }, p.nrow(), options.num_threads);
}

template<bool sparse_, typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_, typename Float_>
void aggregate_across_cells_by_column(
    const tatami::Matrix<Data_, Index_>& p,
    const Group_* const group,
    const AggregateAcrossCellsBuffers<Sum_, Detected_, Float_>& buffers,
    const AggregateAcrossCellsOptions& options
) {
    tatami::Options opt;
    opt.sparse_ordered_index = false;
    assert(buffers.medians.empty());

    tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        const auto NC = p.ncol();
        auto ext = tatami::consecutive_extractor<sparse_>(p, false, static_cast<Index_>(0), NC, start, length, opt);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(length);
        auto ibuffer = [&]{
            if constexpr(sparse_) {
                return tatami::create_container_of_Index_size<std::vector<Index_> >(length);
            } else {
                return false;
            }
        }();

        const auto num_sums = buffers.sums.size();
        auto get_sum = [&](Index_ i) -> Sum_* { return buffers.sums[i]; };
        tatami_stats::LocalOutputBuffers<Sum_, I<decltype(get_sum)>> local_sums(t, num_sums, start, length, std::move(get_sum));

        const auto num_detected = buffers.detected.size();
        auto get_detected = [&](Index_ i) -> Detected_* { return buffers.detected[i]; };
        tatami_stats::LocalOutputBuffers<Detected_, I<decltype(get_detected)>> local_detected(t, num_detected, start, length, std::move(get_detected));

        for (Index_ x = 0; x < NC; ++x) {
            const auto current = group[x];

            if constexpr(sparse_) {
                const auto col = ext->fetch(vbuffer.data(), ibuffer.data());
                if (num_sums) {
                    const auto cursum = local_sums.data(current);
                    for (Index_ i = 0; i < col.number; ++i) {
                        cursum[col.index[i] - start] += col.value[i];
                    }
                }
                if (num_detected) {
                    const auto curdetected = local_detected.data(current);
                    for (Index_ i = 0; i < col.number; ++i) {
                        curdetected[col.index[i] - start] += (col.value[i] > 0);
                    }
                }

            } else {
                const auto col = ext->fetch(vbuffer.data());
                if (num_sums) {
                    const auto cursum = local_sums.data(current);
                    for (Index_ i = 0; i < length; ++i) {
                        cursum[i] += col[i];
                    }
                }
                if (num_detected) {
                    const auto curdetected = local_detected.data(current);
                    for (Index_ i = 0; i < length; ++i) {
                        curdetected[i] += (col[i] > 0);
                    }
                }
            }
        }

        local_sums.transfer();
        local_detected.transfer();
    }, p.nrow(), options.num_threads);
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
 * All entries should be integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param[out] buffers Pre-allocated buffers in which to store the computed statistics. 
 * @param options Further options.
 */
template<typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_, typename Float_>
void aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Group_* const group,
    const AggregateAcrossCellsBuffers<Sum_, Detected_, Float_>& buffers,
    const AggregateAcrossCellsOptions& options
) {
    if (input.prefer_rows() || !buffers.medians.empty()) {
        if (input.sparse()) {
            aggregate_across_cells_by_row<true>(input, group, buffers, options);
        } else {
            aggregate_across_cells_by_row<false>(input, group, buffers, options);
        }
    } else {
        if (input.sparse()) {
            aggregate_across_cells_by_column<true>(input, group, buffers, options);
        } else {
            aggregate_across_cells_by_column<false>(input, group, buffers, options);
        }
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
 * All entries should be integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param options Further options.
 *
 * @return Results of the aggregation, where the available statistics depend on `AggregateAcrossCellsOptions`.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Float_ = double, typename Data_, typename Index_, typename Group_>
AggregateAcrossCellsResults<Sum_, Detected_, Float_> aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Group_* const group,
    const AggregateAcrossCellsOptions& options
) {
    const Index_ NR = input.nrow();
    const Index_ NC = input.ncol();
    const std::size_t ngroups = [&]{
        if (NC) {
            return sanisizer::sum<std::size_t>(*std::max_element(group, group + NC), 1);
        } else {
            return static_cast<std::size_t>(0);
        }
    }();

    AggregateAcrossCellsResults<Sum_, Detected_, Float_> output;
    AggregateAcrossCellsBuffers<Sum_, Detected_, Float_> buffers;

    if (options.compute_sums) {
        sanisizer::resize(output.sums, ngroups);
        sanisizer::resize(buffers.sums, ngroups);
        for (I<decltype(ngroups)> l = 0; l < ngroups; ++l) {
            auto& cursum = output.sums[l];
            tatami::resize_container_to_Index_size<I<decltype(cursum)>>(cursum, NR
#ifdef SCRAN_AGGREGATE_TEST_INIT
                , SCRAN_AGGREGATE_TEST_INIT
#endif
            );
            buffers.sums[l] = cursum.data();
        }
    }

    if (options.compute_detected) {
        sanisizer::resize(output.detected, ngroups);
        sanisizer::resize(buffers.detected, ngroups);
        for (I<decltype(ngroups)> l = 0; l < ngroups; ++l) {
            auto& curdet = output.detected[l];
            tatami::resize_container_to_Index_size<I<decltype(curdet)>>(curdet, NR
#ifdef SCRAN_AGGREGATE_TEST_INIT
                , SCRAN_AGGREGATE_TEST_INIT
#endif
            );
            buffers.detected[l] = curdet.data();
        }
    }

    if (options.compute_medians) {
        sanisizer::resize(output.medians, ngroups);
        sanisizer::resize(buffers.medians, ngroups);
        for (I<decltype(ngroups)> l = 0; l < ngroups; ++l) {
            auto& curdet = output.medians[l];
            tatami::resize_container_to_Index_size<I<decltype(curdet)>>(curdet, NR
#ifdef SCRAN_AGGREGATE_TEST_INIT
                , SCRAN_AGGREGATE_TEST_INIT
#endif
            );
            buffers.medians[l] = curdet.data();
        }
    }


    aggregate_across_cells(input, group, buffers, options);
    return output;
} 

}

#endif
