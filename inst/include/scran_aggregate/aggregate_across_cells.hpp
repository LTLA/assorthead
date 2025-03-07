#ifndef SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP
#define SCRAN_AGGREGATE_AGGREGATE_ACROSS_CELLS_HPP

#include <algorithm>
#include <vector>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

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
     * Whether to compute the sum within each factor level.
     * This option only affects the `aggregate_across_cells()` overload where an `AggregateAcrossCellsResults` object is returned.
     */
    bool compute_sums = true;

    /**
     * Whether to compute the number of detected cells within each factor level.
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
     * Vector of length equal to the number of factor levels.
     * Each element is a pointer to an array of length equal to the number of genes,
     * to be filled with the summed expression across all cells in the corresponding level for each gene.
     *
     * If this is empty, the sums for each level are not computed.
     */
    std::vector<Sum_*> sums;

    /**
     * Vector of length equal to the number of factor levels.
     * Each element is a pointer to an array of length equal to the number of genes,
     * to be filled with the number of cells in the corresponding level with detected expression for each gene.
     * 
     * If this is empty, the number of detected cells for each level is not computed.
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
     * Vector of length equal to the number of factor levels.
     * Each inner vector is of length equal to the number of genes.
     * Each entry contains the summed expression across all cells in the corresponding level for each gene.
     *
     * If `AggregateAcrossCellsOptions::compute_sums = false`, this vector is empty.
     */
    std::vector<std::vector<Sum_> > sums;

    /**
     * Vector of length equal to the number of factor levels.
     * Each inner vector is of length equal to the number of genes.
     * Each entry contains the number of cells in the corresponding level with detected expression for each gene.
     *
     * If `AggregateAcrossCellsOptions::compute_detected = false`, this vector is empty.
     */
    std::vector<std::vector<Detected_> > detected;
};

/**
 * @cond
 */
namespace internal {

template<bool sparse_, typename Data_, typename Index_, typename Factor_, typename Sum_, typename Detected_>
void compute_aggregate_by_row(
    const tatami::Matrix<Data_, Index_>& p,
    const Factor_* factor,
    const AggregateAcrossCellsBuffers<Sum_, Detected_>& buffers,
    const AggregateAcrossCellsOptions& options)
{
    tatami::Options opt;
    opt.sparse_ordered_index = false;

    tatami::parallelize([&](size_t, Index_ s, Index_ l) {
        auto ext = tatami::consecutive_extractor<sparse_>(&p, true, s, l, opt);
        size_t nsums = buffers.sums.size();
        std::vector<Sum_> tmp_sums(nsums);
        size_t ndetected = buffers.detected.size();
        std::vector<Detected_> tmp_detected(ndetected);

        auto NC = p.ncol();
        std::vector<Data_> vbuffer(NC);
        typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(NC);

        for (Index_ x = s, end = s + l; x < end; ++x) {
            auto row = [&]() {
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
                        tmp_sums[factor[row.index[j]]] += row.value[j];
                    }
                } else {
                    for (Index_ j = 0; j < NC; ++j) {
                        tmp_sums[factor[j]] += row[j];
                    }
                }

                // Computing before transferring for more cache-friendliness.
                for (size_t l = 0; l < nsums; ++l) {
                    buffers.sums[l][x] = tmp_sums[l];
                }
            }

            if (ndetected) {
                std::fill(tmp_detected.begin(), tmp_detected.end(), 0);

                if constexpr(sparse_) {
                    for (Index_ j = 0; j < row.number; ++j) {
                        tmp_detected[factor[row.index[j]]] += (row.value[j] > 0);
                    }
                } else {
                    for (Index_ j = 0; j < NC; ++j) {
                        tmp_detected[factor[j]] += (row[j] > 0);
                    }
                }

                for (size_t l = 0; l < ndetected; ++l) {
                    buffers.detected[l][x] = tmp_detected[l];
                }
            }
        }
    }, p.nrow(), options.num_threads);
}

template<bool sparse_, typename Data_, typename Index_, typename Factor_, typename Sum_, typename Detected_>
void compute_aggregate_by_column(
    const tatami::Matrix<Data_, Index_>& p,
    const Factor_* factor,
    const AggregateAcrossCellsBuffers<Sum_, Detected_>& buffers,
    const AggregateAcrossCellsOptions& options)
{
    tatami::Options opt;
    opt.sparse_ordered_index = false;

    tatami::parallelize([&](size_t t, Index_ s, Index_ l) {
        auto NC = p.ncol();
        auto ext = tatami::consecutive_extractor<sparse_>(&p, false, static_cast<Index_>(0), NC, s, l, opt);
        std::vector<Data_> vbuffer(l);
        typename std::conditional<sparse_, std::vector<Index_>, Index_>::type ibuffer(l);

        size_t num_sums = buffers.sums.size();
        std::vector<tatami_stats::LocalOutputBuffer<Sum_> > local_sums;
        local_sums.reserve(num_sums);
        for (auto ptr : buffers.sums) {
            local_sums.emplace_back(t, s, l, ptr);
        }
        size_t num_detected = buffers.detected.size();
        std::vector<tatami_stats::LocalOutputBuffer<Detected_> > local_detected;
        local_detected.reserve(num_detected);
        for (auto ptr : buffers.detected) {
            local_detected.emplace_back(t, s, l, ptr);
        }

        for (Index_ x = 0; x < NC; ++x) {
            auto current = factor[x];

            if constexpr(sparse_) {
                auto col = ext->fetch(vbuffer.data(), ibuffer.data());
                if (num_sums) {
                    auto cursum = local_sums[current].data();
                    for (Index_ i = 0; i < col.number; ++i) {
                        cursum[col.index[i] - s] += col.value[i];
                    }
                }

                if (num_detected) {
                    auto curdetected = local_detected[current].data();
                    for (Index_ i = 0; i < col.number; ++i) {
                        curdetected[col.index[i] - s] += (col.value[i] > 0);
                    }
                }

            } else {
                auto col = ext->fetch(vbuffer.data());
                if (num_sums) {
                    auto cursum = local_sums[current].data();
                    for (Index_ i = 0; i < l; ++i) {
                        cursum[i] += col[i];
                    }
                }

                if (num_detected) {
                    auto curdetected = local_detected[current].data();
                    for (Index_ i = 0; i < l; ++i) {
                        curdetected[i] += (col[i] > 0);
                    }
                }
            }
        }

        for (auto& lsums : local_sums) {
            lsums.transfer();
        }
        for (auto& ldetected : local_detected) {
            ldetected.transfer();
        }
    }, p.nrow(), options.num_threads);
}

}
/**
 * @endcond
 */

/**
 * Aggregate expression values across groups of cells for each gene.
 * We report the sum of expression values and the number of cells with detected (i.e., positive) expression values in each group.
 * This is typically used to create pseudo-bulk expression profiles for cluster/sample combinations.
 * Expression values are generally expected to be counts, though the same function can be used to compute the average log-expression.
 *
 * @tparam Data_ Type of data in the input matrix, should be numeric.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Factor_ Integer type of the factor.
 * @tparam Sum_ Type of the sum, usually the same as `Data`.
 * @tparam Detected_ Type for the number of detected cells, usually integer.
 *
 * @param input The input matrix where rows are features and columns are cells.
 * @param[in] factor Grouping factor.
 * This is a pointer to an array of length equal to the number of columns of `input`, containing the factor level (i.e., assigned group) for each cell.
 * All levels should be integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique levels/groups.
 * @param[out] buffers Collection of buffers in which to store the aggregate statistics (e.g., sums, number of detected cells) for each level and gene.
 * @param options Further options.
 */
template<typename Data_, typename Index_, typename Factor_, typename Sum_, typename Detected_>
void aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Factor_* factor,
    const AggregateAcrossCellsBuffers<Sum_, Detected_>& buffers,
    const AggregateAcrossCellsOptions& options)
{
    if (input.prefer_rows()) {
        if (input.sparse()) {
            internal::compute_aggregate_by_row<true>(input, factor, buffers, options);
        } else {
            internal::compute_aggregate_by_row<false>(input, factor, buffers, options);
        }
    } else {
        if (input.sparse()) {
            internal::compute_aggregate_by_column<true>(input, factor, buffers, options);
        } else {
            internal::compute_aggregate_by_column<false>(input, factor, buffers, options);
        }
    }
} 

/**
 * Overload of `aggregate_across_cells()` that allocates memory for the results.
 *
 * @tparam Sum_ Type of the sum, should be numeric.
 * @tparam Detected_ Type for the number of detected cells, usually integer.
 * @tparam Data_ Type of data in the input matrix, should be numeric.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Factor_ Integer type of the factor.
 *
 * @param input The input matrix where rows are features and columns are cells.
 * @param[in] factor Grouping factor.
 * This is a pointer to an array of length equal to the number of columns of `input`, containing the factor level (i.e., assigned group) for each cell.
 * All levels should be integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique levels/groups.
 * @param options Further options.
 *
 * @return Results of the aggregation, where the available statistics depend on `AggregateAcrossCellsOptions`.
 */
template<typename Sum_ = double, typename Detected_ = int, typename Data_, typename Index_, typename Factor_>
AggregateAcrossCellsResults<Sum_, Detected_> aggregate_across_cells(
    const tatami::Matrix<Data_, Index_>& input,
    const Factor_* factor,
    const AggregateAcrossCellsOptions& options)
{
    size_t NC = input.ncol();
    size_t nlevels = (NC ? *std::max_element(factor, factor + NC) + 1 : 0);
    size_t ngenes = input.nrow();

    AggregateAcrossCellsResults<Sum_, Detected_> output;
    AggregateAcrossCellsBuffers<Sum_, Detected_> buffers;

    if (options.compute_sums) {
        output.sums.resize(nlevels, std::vector<Sum_>(ngenes));
        buffers.sums.resize(nlevels);
        for (size_t l = 0; l < nlevels; ++l) {
            buffers.sums[l] = output.sums[l].data();
        }
    }

    if (options.compute_detected) {
        output.detected.resize(nlevels, std::vector<Detected_>(ngenes));
        buffers.detected.resize(nlevels);
        for (size_t l = 0; l < nlevels; ++l) {
            buffers.detected[l] = output.detected[l].data();
        }
    }

    aggregate_across_cells(input, factor, buffers, options);
    return output;
} 

}

#endif
