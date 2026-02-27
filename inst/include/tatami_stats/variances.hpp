#ifndef TATAMI_STATS_VARS_HPP
#define TATAMI_STATS_VARS_HPP

#include "utils.hpp"

#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file variances.hpp
 *
 * @brief Compute row and column variances from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Functions for computing dimension-wise variances.
 * @namespace tatami_stats::variances
 */
namespace variances {

/**
 * @brief Variance calculation options.
 */
struct Options {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the variance calculation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing variances across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<typename Output_ = double, typename Value_, typename Index_ >
void add_welford(Output_& mean, Output_& sumsq, Value_ value, Index_ count) {
    Output_ delta = value - mean;
    mean += delta / count;
    sumsq += delta * (value - mean);
}

template<typename Output_ = double, typename Index_ >
void add_welford_zeros(Output_& mean, Output_& sumsq, Index_ num_nonzero, Index_ num_all) {
    auto ratio = static_cast<Output_>(num_nonzero) / static_cast<Output_>(num_all);
    sumsq += mean * mean * ratio * (num_all - num_nonzero);
    mean *= ratio;
}

// Avoid problems from interactions between constexpr/lambda/std::conditional. 
template<typename Index_>
struct MockVector {
    MockVector(std::size_t) {}
    Index_& operator[](std::size_t) { return out; }
    std::size_t size() { return 0; }
    Index_ out = 0;
};

}
/**
 * @endcond
 */

/**
 * Compute the mean and variance from a sparse objective vector.
 * This uses the standard two-pass algorithm with naive accumulation of the sum of squared differences;
 * thus, it is best used with a sufficiently high-precision `Output_` like `double`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param[in] value Pointer to an array of length `num`, containing the values of the structural non-zeros.
 * @param num_nonzero Length of the array pointed to by `value`.
 * @param num_all Length of the objective vector, including the structural zeros not in `value`.
 * This should be greater than or equal to `num_nonzero`.
 * @param skip_nan See `Options::skip_nan`.
 *
 * @return The sample mean and variance of values in the sparse array.
 * This may be NaN if there are not enough (non-NaN) values in `value`.
 */
template<typename Output_ = double, typename Value_, typename Index_ >
std::pair<Output_, Output_> direct(const Value_* value, Index_ num_nonzero, Index_ num_all, bool skip_nan) {
    Output_ mean = 0;
    Index_ lost = 0;

    ::tatami_stats::internal::nanable_ifelse<Value_>(
        skip_nan,
        [&]() -> void {
            auto copy = value;
            for (Index_ i = 0; i < num_nonzero; ++i, ++copy) {
                auto val = *copy;
                if (std::isnan(val)) {
                    ++lost;
                } else {
                    mean += val;
                }
            }
        },
        [&]() -> void {
            auto copy = value;
            for (Index_ i = 0; i < num_nonzero; ++i, ++copy) {
                mean += *copy;
            }
        }
    );

    auto count = num_all - lost;
    mean /= count;

    Output_ var = 0;
    ::tatami_stats::internal::nanable_ifelse<Value_>(
        skip_nan,
        [&]() -> void {
            for (Index_ i = 0; i < num_nonzero; ++i) {
                auto val = value[i];
                if (!std::isnan(val)) {
                    auto delta = static_cast<Output_>(val) - mean;
                    var += delta * delta;
                }
            }
        },
        [&]() -> void {
            for (Index_ i = 0; i < num_nonzero; ++i) {
                auto delta = static_cast<Output_>(value[i]) - mean;
                var += delta * delta;
            }
        }
    );

    if (num_nonzero < num_all) {
        var += static_cast<Output_>(num_all - num_nonzero) * mean * mean;
    }

    if (count == 0) {
        return std::make_pair(std::numeric_limits<Output_>::quiet_NaN(), std::numeric_limits<Output_>::quiet_NaN());
    } else if (count == 1) {
        return std::make_pair(mean, std::numeric_limits<Output_>::quiet_NaN());
    } else {
        return std::make_pair(mean, var / (count - 1));
    }
}

/**
 * Compute the mean and variance from a dense objective vector.
 * This uses the standard two-pass algorithm with naive accumulation of the sum of squared differences;
 * thus, it is best used with a sufficiently high-precision `Output_` like `double`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param[in] ptr Pointer to an array of length `num`, containing the values of the objective vector.
 * @param num Length of the objective vector, i.e., length of the array at `ptr`.
 * @param skip_nan See `Options::skip_nan`.
 *
 * @return The sample mean and variance of values in `[ptr, ptr + num)`.
 * This may be NaN if there are not enough (non-NaN) values in `ptr`.
 */
template<typename Output_ = double, typename Value_, typename Index_ >
std::pair<Output_, Output_> direct(const Value_* ptr, Index_ num, bool skip_nan) {
    return direct<Output_>(ptr, num, num, skip_nan);
}

/**
 * @brief Running variances from dense data.
 *
 * Compute running means and variances from dense data using Welford's method.
 * This considers a scenario with a set of equilength "objective" vectors \f$[v_1, v_2, v_3, ..., v_n]\f$,
 * but data are only available for "observed" vectors \f$[p_1, p_2, p_3, ..., p_m]\f$,
 * where the \f$j\f$-th element of \f$p_i\f$ is the \f$i\f$-th element of \f$v_j\f$.
 * The idea is to repeatedly call `add()` for `ptr` corresponding to observed vectors from 0 to \f$m - 1\f$,
 * and then finally call `finish()` to obtain the mean and variance for each objective vector.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 */
template<typename Output_, typename Value_, typename Index_>
class RunningDense {
public:
    /**
     * @param num Number of objective vectors, i.e., \f$n\f$.
     * @param[out] mean Pointer to an output array of length `num`.
     * This should be zeroed on input; after `finish()` is called, this will contain the means for each objective vector.
     * @param[out] variance Pointer to an output array of length `num`, containing the variances for each objective vector.
     * This should be zeroed on input; after `finish()` is called, this will contain the sample variance for each objective vector.
     * @param skip_nan See `Options::skip_nan` for details.
     */
    RunningDense(Index_ num, Output_* mean, Output_* variance, bool skip_nan) : 
        my_num(num),
        my_mean(mean),
        my_variance(variance),
        my_skip_nan(skip_nan),
        my_ok_count(skip_nan ? tatami::can_cast_Index_to_container_size<I<decltype(my_ok_count)> >(num) : static_cast<Index_>(0))
    {}

    /**
     * Add the next observed vector to the variance calculation.
     * @param[in] ptr Pointer to an array of values of length `num`, corresponding to an observed vector.
     */
    void add(const Value_* ptr) {
        // my_count is the upper bound of all my_ok_count, so we check it regardless of my_skip_nan.
        my_count = sanisizer::sum<Index_>(my_count, 1);

        ::tatami_stats::internal::nanable_ifelse<Value_>(
            my_skip_nan,
            [&]() -> void {
                for (Index_ i = 0; i < my_num; ++i, ++ptr) {
                    auto val = *ptr;
                    if (!std::isnan(val)) {
                        internal::add_welford(my_mean[i], my_variance[i], val, ++(my_ok_count[i]));
                    }
                }
            },
            [&]() -> void {
                for (Index_ i = 0; i < my_num; ++i, ++ptr) {
                    internal::add_welford(my_mean[i], my_variance[i], *ptr, my_count);
                }
            }
        );
    }

    /**
     * Finish the variance calculation once all observed vectors have been passed to `add()`. 
     */
    void finish() {
        ::tatami_stats::internal::nanable_ifelse<Value_>(
            my_skip_nan,
            [&]() -> void {
                for (Index_ i = 0; i < my_num; ++i) {
                    auto ct = my_ok_count[i];
                    if (ct < 2) {
                        my_variance[i] = std::numeric_limits<Output_>::quiet_NaN();
                        if (ct == 0) {
                            my_mean[i] = std::numeric_limits<Output_>::quiet_NaN();
                        }
                    } else {
                        my_variance[i] /= ct - 1;
                    }
                }
            },
            [&]() -> void {
                if (my_count < 2) {
                    std::fill_n(my_variance, my_num, std::numeric_limits<Output_>::quiet_NaN());
                    if (my_count == 0) {
                        std::fill_n(my_mean, my_num, std::numeric_limits<Output_>::quiet_NaN());
                    }
                } else {
                    for (Index_ i = 0; i < my_num; ++i) {
                        my_variance[i] /= my_count - 1;
                    }
                }
            }
        );
    }

private:
    Index_ my_num;
    Output_* my_mean;
    Output_* my_variance;
    bool my_skip_nan;
    Index_ my_count = 0;
    typename std::conditional<std::numeric_limits<Value_>::has_quiet_NaN, std::vector<Index_>, internal::MockVector<Index_> >::type my_ok_count;
};

/**
 * @brief Running variances from sparse data.
 *
 * Compute running means and variances from sparse data using Welford's method.
 * This does the same as `RunningDense` but for sparse observed vectors.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 */
template<typename Output_, typename Value_, typename Index_>
class RunningSparse {
public:
    /**
     * @param num Number of objective vectors.
     * @param[out] mean Pointer to an output array of length `num`, containing the means for each objective vector.
     * This should be zeroed on input; after `finish()` is called, this will contain the mean for each objective vector.
     * @param[out] variance Pointer to an output array of length `num`, containing the variances for each objective vector.
     * This should be zeroed on input; after `finish()` is called, this will contain the sample variance for each objective vector.
     * @param skip_nan See `Options::skip_nan` for details.
     * @param subtract Offset to subtract from each element of `index` before using it to index into `mean` and friends.
     * Only relevant if `mean` and friends hold statistics for a contiguous subset of objective vectors,
     * e.g., during task allocation for parallelization.
     */
    RunningSparse(Index_ num, Output_* mean, Output_* variance, bool skip_nan, Index_ subtract = 0) : 
        my_num(num),
        my_mean(mean),
        my_variance(variance),
        my_nonzero(tatami::can_cast_Index_to_container_size<I<decltype(my_nonzero)> >(num)),
        my_skip_nan(skip_nan),
        my_subtract(subtract),
        my_nan(skip_nan ? tatami::can_cast_Index_to_container_size<I<decltype(my_nan)> >(num) : static_cast<Index_>(0))
    {}

    /**
     * Add the next observed vector to the variance calculation.
     * @param[in] value Value of structural non-zero elements.
     * @param[in] index Index of structural non-zero elements.
     * @param number Number of non-zero elements in `value` and `index`.
     */
    void add(const Value_* value, const Index_* index, Index_ number) {
        // my_count is the upper bound of all my_nonzero, so no need to check individual increments.
        my_count = sanisizer::sum<Index_>(my_count, 1);

        ::tatami_stats::internal::nanable_ifelse<Value_>(
            my_skip_nan,
            [&]() -> void {
                for (Index_ i = 0; i < number; ++i) {
                    auto val = value[i];
                    auto ri = index[i] - my_subtract;
                    if (std::isnan(val)) {
                        ++my_nan[ri];
                    } else {
                        internal::add_welford(my_mean[ri], my_variance[ri], val, ++(my_nonzero[ri]));
                    }
                }
            },
            [&]() -> void {
                for (Index_ i = 0; i < number; ++i) {
                    auto ri = index[i] - my_subtract;
                    internal::add_welford(my_mean[ri], my_variance[ri], value[i], ++(my_nonzero[ri]));
                }
            }
        );
    }

    /**
     * Finish the variance calculation once all observed vectors have been passed to `add()`. 
     */
    void finish() {
        ::tatami_stats::internal::nanable_ifelse<Value_>(
            my_skip_nan,
            [&]() -> void {
                for (Index_ i = 0; i < my_num; ++i) {
                    auto& curM = my_mean[i];
                    auto& curV = my_variance[i];
                    Index_ ct = my_count - my_nan[i];

                    if (ct < 2) {
                        curV = std::numeric_limits<Output_>::quiet_NaN();
                        if (ct == 0) {
                            curM = std::numeric_limits<Output_>::quiet_NaN();
                        }
                    } else {
                        internal::add_welford_zeros(curM, curV, my_nonzero[i], ct);
                        curV /= ct - 1;
                    }
                }
            },
            [&]() -> void {
                if (my_count < 2) {
                    std::fill_n(my_variance, my_num, std::numeric_limits<Output_>::quiet_NaN());
                    if (my_count == 0) {
                        std::fill_n(my_mean, my_num, std::numeric_limits<Output_>::quiet_NaN());
                    }
                } else {
                    for (Index_ i = 0; i < my_num; ++i) {
                        auto& var = my_variance[i];
                        internal::add_welford_zeros(my_mean[i], var, my_nonzero[i], my_count);
                        var /= my_count - 1;
                    }
                }
            }
        );
    }

private:
    Index_ my_num;
    Output_* my_mean;
    Output_* my_variance;
    std::vector<Index_> my_nonzero;
    bool my_skip_nan;
    Index_ my_subtract;
    Index_ my_count = 0;
    typename std::conditional<std::numeric_limits<Value_>::has_quiet_NaN, std::vector<Index_>, internal::MockVector<Index_> >::type my_nan;
};

/**
 * Compute variances for each element of a chosen dimension of a `tatami::Matrix`.
 * This may use either Welford's method or the standard two-pass method,
 * depending on the dimension in `row` and the preferred access dimension of `p`.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 *
 * @param row Whether to compute the variance for each row.
 * If false, the variance is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column variances.
 * @param vopt Variance calculation options.
 */
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const Options& vopt) {
    auto dim = (row ? mat.nrow() : mat.ncol());
    auto otherdim = (row ? mat.ncol() : mat.nrow());
    const bool direct = mat.prefer_rows() == row;

    if (mat.sparse()) {
        if (direct) {
            tatami::Options opt;
            opt.sparse_extract_index = false;
            tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, row, s, l);
                auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                for (Index_ x = 0; x < l; ++x) {
                    auto out = ext->fetch(vbuffer.data(), NULL);
                    output[x + s] = variances::direct<Output_>(out.value, out.number, otherdim, vopt.skip_nan).second;
                }
            }, dim, vopt.num_threads);

        } else {
            tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, !row, static_cast<Index_>(0), otherdim, s, l);
                auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(l);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(l);

                auto running_means = tatami::create_container_of_Index_size<std::vector<Output_> >(l);
                LocalOutputBuffer<Output_> local_output(thread, s, l, output);
                variances::RunningSparse<Output_, Value_, Index_> runner(l, running_means.data(), local_output.data(), vopt.skip_nan, s);

                for (Index_ x = 0; x < otherdim; ++x) {
                    auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                    runner.add(out.value, out.index, out.number);
                }
                runner.finish();

                local_output.transfer(); 
            }, dim, vopt.num_threads);
        }

    } else {
        if (direct) {
            tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                for (Index_ x = 0; x < l; ++x) {
                    auto out = ext->fetch(buffer.data());
                    output[x + s] = variances::direct<Output_>(out, otherdim, vopt.skip_nan).second;
                }
            }, dim, vopt.num_threads);

        } else {
            tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, !row, static_cast<Index_>(0), otherdim, s, l);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(l);

                auto running_means = tatami::create_container_of_Index_size<std::vector<Output_> >(l);
                LocalOutputBuffer<Output_> local_output(thread, s, l, output);
                variances::RunningDense<Output_, Value_, Index_> runner(l, running_means.data(), local_output.data(), vopt.skip_nan);

                for (Index_ x = 0; x < otherdim; ++x) {
                    runner.add(ext->fetch(buffer.data()));
                }
                runner.finish();

                local_output.transfer(); 
            }, dim, vopt.num_threads);
        }
    }
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, Output_* output, const Options& vopt) {
    apply(row, *p, output, vopt);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column variances.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param vopt Variance calculation options.
 *
 * @return A vector of length equal to the number of columns, containing the column variances.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat, const Options& vopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.ncol());
    apply(false, mat, output.data(), vopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p, const Options& vopt) {
    return by_column<Output_>(*p, vopt);
}

template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat) {
    return by_column<Output_>(mat, {});
}

template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p) {
    return by_column<Output_>(*p);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column variances.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param vopt Variance calculation options.
 *
 * @return A vector of length equal to the number of rows, containing the row variances.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat, const Options& vopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.nrow());
    apply(true, mat, output.data(), vopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p, const Options& vopt) {
    return by_row<Output_>(*p, vopt);
}

template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat) {
    return by_row<Output_>(mat, {});
}

template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p) {
    return by_row<Output_>(*p);
}
/**
 * @endcond
 */

}

}

#endif
