#ifndef TATAMI_STATS_GROUPED_SUMS_HPP
#define TATAMI_STATS_GROUPED_SUMS_HPP

#include "utils.hpp"
#include "sums.hpp"

#include <vector>
#include <algorithm>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file grouped_sums.hpp
 *
 * @brief Compute group-wise sums from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Functions for computing dimension-wise grouped sums.
 * @namespace tatami_stats::grouped_sums
 */
namespace grouped_sums {

/**
 * @brief Grouped summation options.
 */
struct Options {
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
 * @param[out] output Pointer to an array of pointers of length equal to the number of groups.
 * Each inner pointer should reference an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column sums for each group (indexed according to the assignment in `group`).
 * @param sopt Summation options.
 */
template<typename Value_, typename Index_, typename Group_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, const Group_* group, std::size_t num_groups, Output_** output, const Options& sopt) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    const Index_ otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.sparse()) {
        if (mat.prefer_rows() == row) {
            tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, row, start, len);
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);
                auto tmp = sanisizer::create<std::vector<Output_> >(num_groups);

                for (Index_ i = 0; i < len; ++i) {
                    auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                    std::fill(tmp.begin(), tmp.end(), static_cast<Output_>(0));

                    internal::nanable_ifelse<Value_>(
                        sopt.skip_nan,
                        [&]() -> void {
                            for (Index_ j = 0; j < range.number; ++j) {
                                auto val = range.value[j];
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
                        output[g][i + start] = tmp[g];
                    }
                }
            }, dim, sopt.num_threads);

        } else {
            // Order within each observed vector doesn't affect numerical
            // precision of the outcome, as addition order for each objective
            // vector is already well-defined for a running calculation.
            tatami::Options opt;
            opt.sparse_ordered_index = false; 

            tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
                std::vector<sums::RunningSparse<Output_, Value_, Index_> > runners;
                runners.reserve(num_groups);
                std::vector<LocalOutputBuffer<Output_> > local_output;
                local_output.reserve(num_groups);

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    local_output.emplace_back(thread, start, len, output[g]);
                    runners.emplace_back(local_output.back().data(), sopt.skip_nan, start);
                }

                auto ext = tatami::consecutive_extractor<true>(mat, !row, static_cast<Index_>(0), otherdim, start, len, opt);
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(len);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(len);

                for (Index_ i = 0; i < otherdim; ++i) {
                    auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                    runners[group[i]].add(range.value, range.index, range.number);
                }

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    local_output[g].transfer();
                }
            }, dim, sopt.num_threads);
        }

    } else {
        if (mat.prefer_rows() == row) {
            tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, row, start, len);
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                auto tmp = sanisizer::create<std::vector<Output_> >(num_groups);

                for (Index_ i = 0; i < len; ++i) {
                    auto ptr = ext->fetch(xbuffer.data());
                    std::fill(tmp.begin(), tmp.end(), static_cast<Output_>(0));

                    internal::nanable_ifelse<Value_>(
                        sopt.skip_nan,
                        [&]() -> void {
                            for (Index_ j = 0; j < otherdim; ++j) {
                                auto val = ptr[j];
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
                        output[g][i + start] = tmp[g];
                    }
                }
            }, dim, sopt.num_threads);

        } else {
            tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
                std::vector<sums::RunningDense<Output_, Value_, Index_> > runners;
                runners.reserve(num_groups);
                std::vector<LocalOutputBuffer<Output_> > local_output;
                local_output.reserve(num_groups);

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    local_output.emplace_back(thread, start, len, output[g]);
                    runners.emplace_back(len, local_output.back().data(), sopt.skip_nan);
                }

                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(len);
                auto ext = tatami::consecutive_extractor<false>(mat, !row, static_cast<Index_>(0), otherdim, start, len);

                for (Index_ i = 0; i < otherdim; ++i) {
                    auto ptr = ext->fetch(xbuffer.data());
                    runners[group[i]].add(ptr);
                }

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    local_output[g].transfer();
                }
            }, dim, sopt.num_threads);
        }
    }
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Value_, typename Index_, typename Group_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, const Group_* group, std::size_t num_groups, Output_** output, const Options& sopt) {
    apply(row, *p, group, num_groups, output, sopt);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for row-wise grouped sums.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns.
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param sopt Summation options.
 *
 * @return Vector of length equal to the number of groups.
 * Each entry is a vector of length equal to the number of rows, containing the row-wise sums for the corresponding group.
 */
template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>& mat, const Group_* group, const Options& sopt) {
    auto mydim = mat.nrow();
    auto ngroup = total_groups(group, mat.ncol());

    auto output = sanisizer::create<std::vector<std::vector<Output_> > >(ngroup);
    std::vector<Output_*> ptrs;
    ptrs.reserve(output.size());
    for (auto& o : output) {
        o.resize(mydim);
        ptrs.push_back(o.data());
    }

    apply(true, mat, group, ngroup, ptrs.data(), sopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>* p, const Group_* group, const Options& sopt) {
    return by_row<Output_>(*p, group, sopt);
}

template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>& mat, const Group_* group) {
    return by_row<Output_>(mat, group, Options());
}

template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>* p, const Group_* group) {
    return by_row<Output_>(*p, group);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column-wise grouped sums.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of rows.
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param sopt Summation options.
 *
 * @return Vector of length equal to the number of groups.
 * Each entry is a vector of length equal to the number of columns, containing the column-wise sums for the corresponding group.
 */
template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>& mat, const Group_* group, const Options& sopt) {
    auto mydim = mat.ncol();
    auto ngroup = total_groups(group, mat.nrow());

    auto output = sanisizer::create<std::vector<std::vector<Output_> > >(ngroup);
    std::vector<Output_*> ptrs;
    ptrs.reserve(output.size());
    for (auto& o : output) {
        o.resize(mydim);
        ptrs.push_back(o.data());
    }

    apply(false, mat, group, ngroup, ptrs.data(), sopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>* p, const Group_* group, const Options& sopt) {
    return by_column<Output_>(*p, group, sopt);
}

template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>& mat, const Group_* group) {
    return by_column<Output_>(mat, group, Options());
}

template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>* p, const Group_* group) {
    return by_column<Output_>(*p, group);
}
/**
 * @endcond
 */

}

}

#endif
