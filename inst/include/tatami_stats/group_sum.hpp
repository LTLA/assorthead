#ifndef TATAMI_STATS_GROUPED_SUMS_HPP
#define TATAMI_STATS_GROUPED_SUMS_HPP

#include "utils.hpp"
#include "sum.hpp"

#include <vector>
#include <algorithm>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file group_sum.hpp
 *
 * @brief Compute group-wise sums from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `group_sum()`.
 */
struct GroupSumOptions {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the summation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing sums across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Group_, typename Output_>
void group_sum_direct(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* group,
    const std::size_t num_groups,
    std::vector<Output_*>& output,
    const GroupSumOptions& opt
) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    const Index_ otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.sparse()) {
        tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, start, len);
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);
            auto tmp = sanisizer::create<std::vector<Output_> >(num_groups);

            for (Index_ x = 0; x < len; ++x) {
                auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                std::fill(tmp.begin(), tmp.end(), static_cast<Output_>(0));

                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        for (Index_ j = 0; j < range.number; ++j) {
                            const auto val = range.value[j];
                            if (!std::isnan(val)) {
                                tmp[group[range.index[j]]] += val;
                            }
                        }
                    },
                    [&]() -> void {
                        for (Index_ j = 0; j < range.number; ++j) {
                            tmp[group[range.index[j]]] += range.value[j];
                        }
                    }
                );

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    output[g][start + x] = tmp[g];
                }
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, row, start, len);
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto tmp = sanisizer::create<std::vector<Output_> >(num_groups);

            for (Index_ x = 0; x < len; ++x) {
                auto ptr = ext->fetch(xbuffer.data());
                std::fill(tmp.begin(), tmp.end(), static_cast<Output_>(0));

                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        for (Index_ j = 0; j < otherdim; ++j) {
                            const auto val = ptr[j];
                            if (!std::isnan(val)) {
                                tmp[group[j]] += val;
                            }
                        }
                    },
                    [&]() -> void {
                        for (Index_ j = 0; j < otherdim; ++j) {
                            tmp[group[j]] += ptr[j];
                        }
                    }
                );

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    output[g][start + x] = tmp[g];
                }
            }
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Group_, typename Output_>
void group_sum_running(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* group,
    const std::size_t num_groups,
    std::vector<Output_*>& output,
    const GroupSumOptions& opt
) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    const Index_ otherdim = (row ? mat.ncol() : mat.nrow());
    const bool is_sparse = mat.is_sparse();

    const auto do_parallel = opt.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > all_partial_sums;
    if (do_parallel) {
        all_partial_sums.emplace(sanisizer::cast<I<decltype(all_partial_sums->size())> >(opt.num_threads - 1));
    }

    for (std::size_t g = 0; g < num_groups; ++g) {
        std::fill_n(output[g], dim, 0);
    }

    const auto nused = tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
        // If we can, directly dump it to the output pointers, otherwise put it into a temporary.
        // This will eventually be moved to all_partial_sums but we use a local variable to try to mitigate false sharing.
        Output_** sum_ptrs;
        std::optional<std::vector<std::vector<Output_> > > cur_sums;
        std::optional<std::vector<Output_*> > cur_ptrs;
        if (!do_parallel) {
            sum_ptrs = output.data();
        } else {
            if (thread == 0) {
                sum_ptrs = output.data();
            } else {
                cur_sums.emplace(tatami::cast_Index_to_container_size<std::vector<std::vector<Output_> > >(num_groups));
                cur_ptrs.emplace(tatami::cast_Index_to_container_size<std::vector<Output_*> >(num_groups));
                for (std::size_t g = 0; g < num_groups; ++g) {
                    tatami::resize_container_to_Index_size((*cur_sums)[g], dim);
                    (*cur_ptrs)[g] = (*cur_sums)[g].data();
                }
                sum_ptrs = cur_ptrs->data();
            }
        }

        if (is_sparse) {
            // Order within each observed vector doesn't affect numerical precision of the outcome,
            // as addition order for each objective vector is already well-defined for a running calculation.
            tatami::Options topt;
            topt.sparse_ordered_index = false; 
            auto ext = tatami::consecutive_extractor<true>(mat, !row, start, len, topt);
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);

            for (Index_ x = 0; x < len; ++x) {
                auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                const auto sum_ptr = sum_ptrs[group[start + x]];

                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        for (Index_ i = 0; i < range.number; ++i) {
                            const auto val = range.value[i];
                            if (!std::isnan(val)) {
                                sum_ptr[range.index[i]] += val;
                            }
                        }
                    },
                    [&]() -> void {
                        for (Index_ i = 0; i < range.number; ++i) {
                            sum_ptr[range.index[i]] += range.value[i];
                        }
                    }
                );
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, !row, start, len);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);

            for (Index_ x = 0; x < len; ++x) {
                auto ptr = ext->fetch(buffer.data());
                const auto sum_ptr = sum_ptrs[group[start + x]];

                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        for (Index_ d = 0; d < dim; ++d) {
                            const auto val = ptr[d];
                            if (!std::isnan(val)) {
                                sum_ptr[d] += val;
                            }
                        }
                    },
                    [&]() -> void {
                        for (Index_ d = 0; d < dim; ++d) {
                            sum_ptr[d] += ptr[d];
                        }
                    }
                );
            }
        }

        if (do_parallel) {
            if (thread > 0) {
                (*all_partial_sums)[thread - 1] = std::move(cur_sums);
            }
        }
    }, otherdim, opt.num_threads);

    if (do_parallel) {
        for (std::size_t g = 0; g < num_groups; ++g) {
            const auto cur_out = output[g];
            for (int u = 1; u < nused; ++u) {
                const auto& cur_sum = (*((*all_partial_sums)[u - 1]))[g];
                for (Index_ d = 0; d < dim; ++d) {
                    cur_out[d] += cur_sum[d];
                }
            }
        }
    }
}
/**
 * @endcond
 */

/**
 * Compute per-group sums for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row.
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 *
 * @param row Whether to compute group-wise sums within each row.
 * If false, sums are computed within the column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * This can be determined by calling `tatami_stats::total_groups()` on `group`.
 * @param[out] output Vector of length equal to the number of groups.
 * Each element is a pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, each array will contain the row/column sums for the corresponding group. 
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Group_, typename Output_>
void group_sum(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* group,
    const std::size_t num_groups,
    std::vector<Output_*>& output,
    const GroupSumOptions& opt
) {
    if (mat.prefer_rows() == row) {
        group_sum_direct(row, mat, group, num_groups, output, opt);
    } else {
        group_sum_running(row, mat, group, num_groups, output, opt);
    }
}

/**
 * Overload of `group_sum()` that allocates memory for the output sums.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row.
 *
 * @param row Whether to compute group-wise sums within each row.
 * If false, sums are computed within the column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * This can be determined by calling `tatami_stats::total_groups()` on `group`.
 * @param opt Further options.
 *
 * @return Vector of length equal to the number of groups.
 * Each element is a vector of length equal to the number of rows (if `row = true`) or columns (otherwise),
 * containing the row/column sums for the corresponding group. 
 */
template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > group_sum(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* group,
    const std::size_t num_groups,
    const GroupSumOptions& opt
) {
    auto output = sanisizer::create<std::vector<std::vector<Output_> > >(num_groups);
    auto ptrs = sanisizer::create<std::vector<Output_*> >(num_groups);
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    for (std::size_t g = 0; g < num_groups; ++g) {
        tatami::resize_container_to_Index_size(output[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        ptrs[g] = output[g].data();
    }
    group_sum(row, mat, group, num_groups, ptrs, opt);
    return output;
}

}

#endif
