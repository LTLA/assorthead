#ifndef SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP
#define SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP

#include <algorithm>
#include <vector>
#include <cstddef>
#include <type_traits>

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
     * Whether to compute the sum within each group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_sums = true;

    /**
     * Whether to compute the number of detected cells within each group.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_detected = true;

    /**
     * Number of threads to use. 
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `aggregate_across_cells()`.
 * @tparam Sum_ Type of the sum, should be numeric.
 * @tparam Detected_ Type for the number of detected cells, usually integer.
 */
template <typename Sum_, typename Detected_>
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

};

/**
 * @brief Results of `aggregate_across_cells()`.
 * @tparam Sum_ Type of the sum, should be numeric.
 * @tparam Detected_ Type for the number of detected cells, usually integer.
 */
template <typename Sum_, typename Detected_>
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
};

/**
 * @cond
 */
template<bool sparse_, typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_>
void aggregate_across_cells_by_row(
    const tatami::Matrix<Data_, Index_>& p,
    const Group_* const group,
    const AggregateAcrossCellsBuffers<Sum_, Detected_>& buffers,
    const AggregateAcrossCellsOptions& options)
{
    tatami::Options opt;
    opt.sparse_ordered_index = false;

    tatami::parallelize([&](const int, const Index_ s, const Index_ l) -> void {
        auto ext = tatami::consecutive_extractor<sparse_>(p, true, s, l, opt);
        const auto nsums = buffers.sums.size();
        auto tmp_sums = sanisizer::create<std::vector<Sum_> >(nsums);
        const auto ndetected = buffers.detected.size();
        auto tmp_detected = sanisizer::create<std::vector<Detected_> >(ndetected);

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
        }
    }, p.nrow(), options.num_threads);
}

template<bool sparse_, typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_>
void aggregate_across_cells_by_column(
    const tatami::Matrix<Data_, Index_>& p,
    const Group_* const group,
    const AggregateAcrossCellsBuffers<Sum_, Detected_>& buffers,
    const AggregateAcrossCellsOptions& options)
{
    tatami::Options opt;
    opt.sparse_ordered_index = false;

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
 * We report the sum of expression values and the number of cells with detected (i.e., positive) expression values in each group.
 * This is typically used to create pseudo-bulk expression profiles for cluster/sample combinations.
 * Expression values are generally expected to be counts so that the sums can be used as if they were counts from bulk data, e.g., for differential analyses with **edgeR**.
 *
 * @tparam Data_ Type of data in the input matrix, should be numeric.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Group_ Integer type of the group assignments.
 * @tparam Sum_ Numeric type of the sum, often the same as `Data_`.
 * @tparam Detected_ Numeric type (usually integer) of the number of detected cells. 
 *
 * @param input The input matrix, usually containing non-negative counts.
 * Rows are features and columns are cells.
 * @param[in] group Pointer to an array of length equal to the number of columns of `input`, containing the assigned group for each cell.
 * All entries should be integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param options Further options.
 */
template<typename Data_, typename Index_, typename Group_, typename Sum_, typename Detected_>
void aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Group_* const group,
    const AggregateAcrossCellsBuffers<Sum_, Detected_>& buffers,
    const AggregateAcrossCellsOptions& options)
{
    if (input.prefer_rows()) {
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
 * @tparam Sum_ Numerict ype of the sum.
 * @tparam Detected_ Numeric type (usually integer) of the number of detected cells. 
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
template<typename Sum_ = double, typename Detected_ = int, typename Data_, typename Index_, typename Group_>
AggregateAcrossCellsResults<Sum_, Detected_> aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Group_* const group,
    const AggregateAcrossCellsOptions& options)
{
    const Index_ NR = input.nrow();
    const Index_ NC = input.ncol();
    const std::size_t ngroups = [&]{
        if (NC) {
            return sanisizer::sum<std::size_t>(*std::max_element(group, group + NC), 1);
        } else {
            return static_cast<std::size_t>(0);
        }
    }();

    AggregateAcrossCellsResults<Sum_, Detected_> output;
    AggregateAcrossCellsBuffers<Sum_, Detected_> buffers;

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

    aggregate_across_cells(input, group, buffers, options);
    return output;
} 

}

#endif
