#ifndef TATAMI_STATS__SUMS_HPP
#define TATAMI_STATS__SUMS_HPP

#include "utils.hpp"

#include <vector>
#include <numeric>
#include <algorithm>

#include "tatami/tatami.hpp"

/**
 * @file sums.hpp
 *
 * @brief Compute row and column sums from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Functions for computing dimension-wise sums.
 * @namespace tatami_stats::sums
 */
namespace sums {

/**
 * @brief Summation options.
 */
struct Options {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of summation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing sums across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * Compute the sum across an objective vector using naive accumulation.
 * This is best used with a sufficiently high-precision `Output_`, hence the default of `double`.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param[in] ptr Pointer to an array of length `num`, containing the values of the objective vector.
 * @param num Size of the array at `ptr`.
 * This may be less than the length of the objective vector for sparse data.
 * @param skip_nan See `Options::skip_nan`.
 * @return The sum.
 */
template<typename Output_ = double, typename Value_, typename Index_>
Output_ direct(const Value_* ptr, Index_ num, bool skip_nan) {
    return internal::nanable_ifelse_with_value<Value_>(
        skip_nan,
        [&]() -> Output_ {
            Output_ sum = 0;
            for (Index_ i = 0; i < num; ++i) {
                auto val = ptr[i];
                if (!std::isnan(val)) {
                    sum += val;
                }
            }
            return sum;
        },
        [&]() -> Output_ {
            return std::accumulate(ptr, ptr + num, static_cast<Output_>(0));
        }
    );
}

/**
 * @brief Running sums from dense data.
 *
 * This considers a scenario with a set of equilength "objective" vectors \f$[v_1, v_2, v_3, ..., v_n]\f$,
 * but data are only available for "observed" vectors \f$[p_1, p_2, p_3, ..., p_m]\f$,
 * where the \f$j\f$-th element of \f$p_i\f$ is the \f$i\f$-th element of \f$v_j\f$.
 * The idea is to repeatedly call `add()` for `ptr` corresponding to observed vectors from 0 to \f$m - 1\f$,
 * and then finally call `finish()` to obtain the sum for each objective vector.
 *
 * This class uses naive accumulation to obtain the sum for each objective vector.
 * Callers should use a sufficiently high-precision `Output_` such as `double` to mitigate round-off errors.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 */
template<typename Output_, typename Value_, typename Index_>
class RunningDense {
public:
    /**
     * @param num Number of objective vectors, i.e., \f$n\f$.
     * @param[out] sum Pointer to an output array of length `num`.
     * This should be zeroed on input, and will store the running sums after each `add()`.
     * @param skip_nan See `Options::skip_nan` for details.
     */
    RunningDense(Index_ num, Output_* sum, bool skip_nan) : my_num(num), my_sum(sum), my_skip_nan(skip_nan) {}

    /**
     * Add the next observed vector to the running sums.
     * @param[in] ptr Pointer to an array of values of length `my_num`, corresponding to an observed vector.
     */
    void add(const Value_* ptr) {
        internal::nanable_ifelse<Value_>(
            my_skip_nan,
            [&]() -> void {
                for (Index_ i = 0; i < my_num; ++i) {
                    auto val = ptr[i];
                    if (!std::isnan(val)) {
                        my_sum[i] += val;
                    }
                }
            },
            [&]() -> void {
                for (Index_ i = 0; i < my_num; ++i) {
                    my_sum[i] += ptr[i];
                }
            }
        );
    }

private:
    Index_ my_num;
    Output_* my_sum;
    bool my_skip_nan;
};

/**
 * @brief Running sums from sparse data.
 *
 * Compute running sums from sparse data. 
 * This is the counterpart to `RunningDense`, but for sparse observed vectors.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 */
template<typename Output_, typename Value_, typename Index_>
class RunningSparse {
public:
    /**
     * @param[out] sum Pointer to an output array of length equal to the number of objective vectors.
     * This should be zeroed on input, and will store the running sums after each `add()`.
     * @param skip_nan See `Options::skip_nan` for details.
     * @param subtract Offset to subtract from each element of `index` before using it to index into `mean` and friends.
     * Only relevant if `mean` and friends hold statistics for a contiguous subset of objective vectors,
     * e.g., during task allocation for parallelization.
     */
    RunningSparse(Output_* sum, bool skip_nan, Index_ subtract = 0) : 
        my_sum(sum), my_skip_nan(skip_nan), my_subtract(subtract) {}

    /**
     * Add the next observed vector to the running sums.
     * @param[in] value Value of structural non-zero elements.
     * @param[in] index Index of structural non-zero elements.
     * This does not have to be sorted.
     * @param number Number of non-zero elements in `value` and `index`.
     */
    void add(const Value_* value, const Index_* index, Index_ number) {
        internal::nanable_ifelse<Value_>(
            my_skip_nan,
            [&]() -> void {
                for (Index_ i = 0; i < number; ++i) {
                    auto val = value[i];
                    if (!std::isnan(val)) {
                        my_sum[index[i] - my_subtract] += val;
                    }
                }
            },
            [&]() -> void {
                for (Index_ i = 0; i < number; ++i) {
                    my_sum[index[i] - my_subtract] += value[i];
                }
            }
        );
    }

private:
    Output_* my_sum;
    bool my_skip_nan;
    Index_ my_subtract;
};

/**
 * Compute sums for each element of a chosen dimension of a `tatami::Matrix`.
 * This is done using `direct()`, `RunningDense` or `RunningSparse`, 
 * depending on the requested dimension in `row` and the preferred dimension for access in `p`.
 * Note that all sums are obtained using naive accumulation,
 * so it is best to use a sufficiently high-precision `Output_` to mitigate round-off errors.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 *
 * @param row Whether to compute the sum for each row.
 * If false, the sum is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column variances.
 * @param sopt Summation options.
 */
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const Options& sopt) {
    auto dim = (row ? mat.nrow() : mat.ncol());
    auto otherdim = (row ? mat.ncol() : mat.nrow());
    const bool direct = mat.prefer_rows() == row;

    if (mat.sparse()) {
        if (direct) {
            tatami::Options opt;
            opt.sparse_extract_index = false;

            tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, opt);
                auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                for (Index_ x = 0; x < l; ++x) {
                    auto out = ext->fetch(vbuffer.data(), NULL);
                    output[x + s] = sums::direct(out.value, out.number, sopt.skip_nan);
                }
            }, dim, sopt.num_threads);

        } else {
            tatami::Options opt;
            opt.sparse_ordered_index = false;

            tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, !row, static_cast<Index_>(0), otherdim, s, l, opt);
                auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(l);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(l);

                LocalOutputBuffer<Output_> local_output(thread, s, l, output);
                sums::RunningSparse<Output_, Value_, Index_> runner(local_output.data(), sopt.skip_nan, s);

                for (Index_ x = 0; x < otherdim; ++x) {
                    auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                    runner.add(out.value, out.index, out.number);
                }

                local_output.transfer();
            }, dim, sopt.num_threads);
        }

    } else {
        if (direct) {
            tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                for (Index_ x = 0; x < l; ++x) {
                    auto out = ext->fetch(buffer.data());
                    output[x + s] = sums::direct(out, otherdim, sopt.skip_nan);
                }
            }, dim, sopt.num_threads);

        } else {
            tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, !row, static_cast<Index_>(0), otherdim, s, l);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(l);

                LocalOutputBuffer<Output_> local_output(thread, s, l, output);
                sums::RunningDense<Output_, Value_, Index_> runner(l, local_output.data(), sopt.skip_nan);

                for (Index_ x = 0; x < otherdim; ++x) {
                    auto out = ext->fetch(buffer.data());
                    runner.add(out);
                }

                local_output.transfer();
            }, dim, sopt.num_threads);
        }
    }

    return;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, Output_* output, const Options& sopt) {
    apply(row, *p, output, sopt);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column sums.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param sopt Summation options.
 * @return Vector containing the sum for each column.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat, const Options& sopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.ncol());
    apply(false, mat, output.data(), sopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p, const Options& sopt) {
    return by_column<Output_>(*p, sopt);
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
 * Wrapper around `apply()` for row sums.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param sopt Summation options.
 * @return Vector containing the sum of each row.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat, const Options& sopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.nrow());
    apply(true, mat, output.data(), sopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p, const Options& sopt) {
    return by_row<Output_>(*p, sopt);
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
