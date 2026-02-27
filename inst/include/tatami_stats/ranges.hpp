#ifndef TATAMI_STATS_RANGES_HPP
#define TATAMI_STATS_RANGES_HPP

#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <type_traits>

#include "tatami/tatami.hpp"

/**
 * @file ranges.hpp
 *
 * @brief Compute row and column ranges from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Functions for computing dimension-wise ranges.
 * @namespace tatami_stats::ranges
 */
namespace ranges {

/**
 * @brief Range calculation options.
 */
struct Options {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the range calculation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing ranges across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<typename Value_>
constexpr Value_ choose_minimum_placeholder() {
    // Placeholder value 'x' is such that 'x >= y' is always true for any non-NaN 'y'.
    if constexpr(std::numeric_limits<Value_>::has_infinity) {
        return std::numeric_limits<Value_>::infinity();
    } else {
        return std::numeric_limits<Value_>::max();
    }
}

template<typename Value_>
constexpr Value_ choose_maximum_placeholder() {
    // Placeholder value 'x' is such that 'x <= y' is always true for any non-NaN 'y'.
    if constexpr(std::numeric_limits<Value_>::has_infinity) {
        return -std::numeric_limits<Value_>::infinity();
    } else {
        return std::numeric_limits<Value_>::lowest();
    }
}

}
/**
 * @endcond
 */

/**
 * Directly compute the minimum or maximum of a dense objective vector.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param[in] ptr Pointer to an array of length `num`, containing the values of the objective vector.
 * @param num Length of the objective vector, i.e., length of the array at `ptr`.
 * @param minimum Whether to compute the minimum.
 * If false, the maximum is computed instead.
 * @param skip_nan See `Options::skip_nan` for details.
 *
 * @return The minimum or maximum value, depending on `minimum`.
 * If `num = 0` or (if `skip_nan = true`) there are no non-NaN values, a placeholder value is returned instead
 * that is never less than (if `minimum true`) or greater than (otherwise) any non-NaN value of type `Value_`.
 */
template<typename Value_, typename Index_>
Value_ direct(const Value_* ptr, Index_ num, bool minimum, bool skip_nan) {
    return ::tatami_stats::internal::nanable_ifelse_with_value<Value_>(
        skip_nan,

        [&]() -> Value_ {
            if (minimum) {
                auto current = internal::choose_minimum_placeholder<Value_>(); 
                for (Index_ i = 0; i < num; ++i) {
                    auto val = ptr[i];
                    if (val < current) { // no need to explicitly handle NaNs, as any comparison with NaNs is always false.
                        current = val;
                    }
                }
                return current;
            } else {
                auto current = internal::choose_maximum_placeholder<Value_>(); 
                for (Index_ i = 0; i < num; ++i) {
                    auto val = ptr[i];
                    if (val > current) { // again, no need to explicitly handle NaNs, as any comparison with NaNs is always false.
                        current = val;
                    }
                }
                return current;
            }
        },

        [&]() -> Value_ {
            if (num) {
                if (minimum) {
                    return *std::min_element(ptr, ptr + num);
                } else {
                    return *std::max_element(ptr, ptr + num);
                }
            } else {
                if (minimum) {
                    return internal::choose_minimum_placeholder<Value_>();
                } else {
                    return internal::choose_maximum_placeholder<Value_>();
                }
            }
        }
    );
}

/**
 * Compute the extremes of a sparse objective vector.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param[in] value Pointer to an array of length `num_nonzero`, containing the values of the structural non-zeros.
 * @param num_nonzero Number of structural non-zeros in the objective vector.
 * @param num_all Length of the objective vector, including the structural zeros not in `value`.
 * This should be greater than or equal to `num_nonzero`.
 * @param minimum Whether to compute the minimum.
 * If false, the maximum is computed instead.
 * @param skip_nan See `Options::skip_nan` for details.
 *
 * @return The minimum or maximum value, depending on `minimum`.
 * If `num_all = 0` or (if `skip_nan = true`) there are no non-NaN values, a placeholder value is returned instead
 * that is never less than (if `minimum true`) or greater than (otherwise) any non-NaN value of type `Value_`.
 */
template<typename Value_, typename Index_>
Value_ direct(const Value_* value, Index_ num_nonzero, Index_ num_all, bool minimum, bool skip_nan) {
    if (num_nonzero) {
        auto candidate = direct(value, num_nonzero, minimum, skip_nan);
        if (num_nonzero < num_all) {
            if (minimum) {
                if (candidate > 0) {
                    candidate = 0;
                }
            } else {
                if (candidate < 0) {
                    candidate = 0;
                }
            }
        }
        return candidate;

    } else if (num_all) {
        return 0;

    } else {
        if (minimum) {
            return internal::choose_minimum_placeholder<Value_>();
        } else {
            return internal::choose_maximum_placeholder<Value_>();
        }
    }
}

/**
 * @brief Running minima/maxima from dense data.
 *
 * This considers a scenario with a set of equilength "objective" vectors \f$[v_1, v_2, v_3, ..., v_n]\f$,
 * but data are only available for "observed" vectors \f$[p_1, p_2, p_3, ..., p_m]\f$,
 * where the \f$j\f$-th element of \f$p_i\f$ is the \f$i\f$-th element of \f$v_j\f$.
 * The idea is to repeatedly call `add()` for `ptr` corresponding to observed vectors from 0 to \f$m - 1\f$,
 * which computes the running minimum/maximum for each objective vector at each invocation.
 *
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 */
template<typename Output_, typename Value_, typename Index_>
class RunningDense {
public:
    /**
     * @param num Number of objective vectors, i.e., \f$n\f$.
     * @param[out] store_min Pointer to an output array of length `num`.
     * After `finish()` is called, this will contain the minimum for each objective vector.
     * If `NULL`, no minimum is reported.
     * @param[out] store_max Pointer to an output array of length `num`.
     * After `finish()` is called, this will contain the maximum for each objective vector.
     * If `NULL`, no maximum is reported.
     * @param skip_nan See `Options::skip_nan` for details.
     */
    RunningDense(Index_ num, Output_* store_min, Output_* store_max, bool skip_nan) :
        my_num(num), my_store_min(store_min), my_store_max(store_max), my_skip_nan(skip_nan) {}

    /**
     * Add the next observed vector to the running min/max calculation.
     * @param[in] ptr Pointer to an array of values of length `num`, corresponding to an observed vector.
     */
    void add(const Value_* ptr) {
        if (my_init) {
            my_init = false;
            ::tatami_stats::internal::nanable_ifelse<Value_>(
                my_skip_nan,

                [&]() -> void {
                    if (my_store_min) {
                        for (Index_ i = 0; i < my_num; ++i) {
                            auto val = ptr[i];
                            if (std::isnan(val)) {
                                my_store_min[i] = internal::choose_minimum_placeholder<Value_>();
                            } else {
                                my_store_min[i] = val;
                            }
                        }
                    }
                    if (my_store_max) {
                        for (Index_ i = 0; i < my_num; ++i) {
                            auto val = ptr[i];
                            if (std::isnan(val)) {
                                my_store_max[i] = internal::choose_maximum_placeholder<Value_>();
                            } else {
                                my_store_max[i] = val;
                            }
                        }
                    }
                },

                [&]() -> void {
                    if (my_store_min) {
                        std::copy_n(ptr, my_num, my_store_min);
                    }
                    if (my_store_max) {
                        std::copy_n(ptr, my_num, my_store_max);
                    }
                }
            );

        } else {
            if (my_store_min) {
                for (Index_ i = 0; i < my_num; ++i) {
                    auto val = ptr[i];
                    auto& current = my_store_min[i];
                    if (val < current) { // this should implicitly skip val=NaNs, as any NaN comparison will be false.
                        current = val;
                    }
                }
            }
            if (my_store_max) {
                for (Index_ i = 0; i < my_num; ++i) {
                    auto val = ptr[i];
                    auto& current = my_store_max[i];
                    if (val > current) { // this should implicitly skip val=NaNs, as any NaN comparison will be false.
                        current = val;
                    }
                }
            }
        }
    }

    /**
     * Finish the running calculation once all observed vectors have been passed to `add()`. 
     */
    void finish() {
        if (my_init) {
            if (my_store_min) {
                std::fill_n(my_store_min, my_num, internal::choose_minimum_placeholder<Value_>());
            }
            if (my_store_max) {
                std::fill_n(my_store_max, my_num, internal::choose_maximum_placeholder<Value_>());
            }
        }
    }

private:
    bool my_init = true;
    Index_ my_num;
    Output_* my_store_min;
    Output_* my_store_max;
    bool my_skip_nan;
};

/**
 * @brief Running minima/maxima from sparse data.
 *
 * Compute running minima and maximuma from sparse data. 
 * This does the same as `RunningDense` but for sparse observed vectors.
 *
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 */
template<typename Output_, typename Value_, typename Index_>
class RunningSparse {
public:
    /**
     * @param num Number of objective vectors.
     * @param[out] store_min Pointer to an output array of length `num`.
     * After `finish()` is called, this will contain the minimum for each objective vector.
     * If `NULL`, no minimum is reported.
     * @param[out] store_max Pointer to an output array of length `num`.
     * After `finish()` is called, this will contain the maximum for each objective vector.
     * If `NULL`, no maximum is reported.
     * @param skip_nan See `Options::skip_nan` for details.
     * @param subtract Offset to subtract from each element of `index` before using it to index into `store`.
     * Only relevant if `store` holds statistics for a contiguous subset of objective vectors,
     * e.g., during task allocation for parallelization.
     */
    RunningSparse(Index_ num, Output_* store_min, Output_* store_max, bool skip_nan, Index_ subtract = 0) : 
        my_num(num), my_store_min(store_min), my_store_max(store_max), my_skip_nan(skip_nan), my_subtract(subtract) {}

    /**
     * Add the next observed vector to the min/max calculation.
     * @param[in] value Value of structural non-zero elements.
     * @param[in] index Index of structural non-zero elements.
     * @param number Number of non-zero elements in `value` and `index`.
     */
    void add(const Value_* value, const Index_* index, Index_ number) {
        if (my_count == 0) {
            tatami::resize_container_to_Index_size(my_nonzero, my_num);

            if (my_store_min) {
                std::fill_n(my_store_min, my_num, internal::choose_minimum_placeholder<Value_>());
            }
            if (my_store_max) {
                std::fill_n(my_store_max, my_num, internal::choose_maximum_placeholder<Value_>());
            }

            if (!my_skip_nan) {
                for (Index_ i = 0; i < number; ++i) {
                    auto val = value[i];
                    auto idx = index[i] - my_subtract;
                    if (my_store_min) {
                        my_store_min[idx] = val;
                    }
                    if (my_store_max) {
                        my_store_max[idx] = val;
                    }
                    ++my_nonzero[idx];
                }
                my_count = 1;
                return;
            }
        }

        for (Index_ i = 0; i < number; ++i) {
            auto val = value[i];
            auto idx = index[i] - my_subtract;
            if (my_store_min) { // this should implicitly skip NaNs, any NaN comparison will be false.
                auto& current = my_store_min[idx];
                if (current > val) {
                    current = val;
                }
            }
            if (my_store_max) {
                auto& current = my_store_max[idx];
                if (current < val) {
                    current = val;
                }
            }
            ++my_nonzero[idx];
        }

        ++my_count;
    }

    /**
     * Finish the min/max calculation once all observed vectors have been passed to `add()`. 
     */
    void finish() {
        if (my_count) {
            for (Index_ i = 0; i < my_num; ++i) {
                if (my_count > my_nonzero[i]) {
                    if (my_store_min) {
                        auto& current = my_store_min[i];
                        if (current > 0) {
                            current = 0;
                        }
                    }
                    if (my_store_max) {
                        auto& current = my_store_max[i];
                        if (current < 0) {
                            current = 0;
                        }
                    }
                }
            }
        } else {
            if (my_store_min) {
                std::fill_n(my_store_min, my_num, internal::choose_minimum_placeholder<Value_>());
            }
            if (my_store_max) {
                std::fill_n(my_store_max, my_num, internal::choose_maximum_placeholder<Value_>());
            }
        }
    }

private:
    Index_ my_num;
    Output_* my_store_min;
    Output_* my_store_max;
    bool my_skip_nan;
    Index_ my_subtract;
    Index_ my_count = 0;
    std::vector<Index_> my_nonzero;
};

/**
 * Compute ranges for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 *
 * @param row Whether to compute the range for each row.
 * If false, the range is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] min_out Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the minimum of each row/column.
 * Alternatively, this may be NULL, in which case the minima are not computed.
 * @param[out] max_out Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the maximum of each row/column.
 * Alternatively, this may be NULL, in which case the maxima are not computed.
 * @param ropt Range calculation options.
 */
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* min_out, Output_* max_out, const Options& ropt) {
    auto dim = (row ? mat.nrow() : mat.ncol());
    auto otherdim = (row ? mat.ncol() : mat.nrow());
    const bool direct = mat.prefer_rows() == row;

    bool store_min = min_out != NULL;
    bool store_max = max_out != NULL;

    if (mat.sparse()) {
        tatami::Options opt;
        opt.sparse_ordered_index = false;

        if (direct) {
            opt.sparse_extract_index = false;
            tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, opt);
                auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                for (Index_ x = 0; x < l; ++x) {
                    auto out = ext->fetch(vbuffer.data(), NULL);
                    if (store_min) {
                        min_out[x + s] = ranges::direct(out.value, out.number, otherdim, true, ropt.skip_nan);
                    }
                    if (store_max) {
                        max_out[x + s] = ranges::direct(out.value, out.number, otherdim, false, ropt.skip_nan);
                    }
                }
            }, dim, ropt.num_threads);

        } else {
            tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<true>(mat, !row, static_cast<Index_>(0), otherdim, s, l, opt);
                auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(l);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(l);

                auto local_min = (store_min ? LocalOutputBuffer<Output_>(thread, s, l, min_out) : LocalOutputBuffer<Output_>());
                auto local_max = (store_max ? LocalOutputBuffer<Output_>(thread, s, l, max_out) : LocalOutputBuffer<Output_>());
                ranges::RunningSparse<Output_, Value_, Index_> runner(l, local_min.data(), local_max.data(), ropt.skip_nan, s);

                for (Index_ x = 0; x < otherdim; ++x) {
                    auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                    runner.add(out.value, out.index, out.number);
                }

                runner.finish();
                if (store_min) {
                    local_min.transfer();
                }
                if (store_max) {
                    local_max.transfer();
                }
            }, dim, ropt.num_threads);
        }

    } else {
        if (direct) {
            tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                for (Index_ x = 0; x < l; ++x) {
                    auto ptr = ext->fetch(buffer.data());
                    if (store_min) {
                        min_out[x + s] = ranges::direct(ptr, otherdim, true, ropt.skip_nan);
                    }
                    if (store_max) {
                        max_out[x + s] = ranges::direct(ptr, otherdim, false, ropt.skip_nan);
                    }
                }
            }, dim, ropt.num_threads);

        } else {
            tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
                auto ext = tatami::consecutive_extractor<false>(mat, !row, static_cast<Index_>(0), otherdim, s, l);
                auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(l);

                auto local_min = (store_min ? LocalOutputBuffer<Output_>(thread, s, l, min_out) : LocalOutputBuffer<Output_>());
                auto local_max = (store_max ? LocalOutputBuffer<Output_>(thread, s, l, max_out) : LocalOutputBuffer<Output_>());
                ranges::RunningDense<Output_, Value_, Index_> runner(l, local_min.data(), local_max.data(), ropt.skip_nan);

                for (Index_ x = 0; x < otherdim; ++x) {
                    auto ptr = ext->fetch(buffer.data());
                    runner.add(ptr);
                }

                runner.finish();
                if (store_min) {
                    local_min.transfer();
                }
                if (store_max) {
                    local_max.transfer();
                }
            }, dim, ropt.num_threads);
        }
    }

    return;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, Output_* min_out, Output_* max_out, const Options& ropt) {
    apply(row, *p, min_out, max_out, ropt); 
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column ranges.
 *
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param ropt Range calculation options.
 *
 * @return A pair of vectors, each of length equal to the number of columns.
 * The first and second vector contains the minimum and maximum value per column, respectively.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>& mat, const Options& ropt) {
    auto mins = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.ncol());
    auto maxs = mins;
    apply(false, mat, mins.data(), maxs.data(), ropt);
    return std::make_pair(std::move(mins), std::move(maxs));
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>* p, const Options& ropt) {
    return by_column<Output_>(*p, ropt);
}

template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>& mat) {
    return by_column<Output_>(mat, {});
}

template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_column(const tatami::Matrix<Value_, Index_>* p) {
    return by_column<Output_>(*p);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for row ranges.
 *
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param ropt Range calculation options.
 *
 * @return A pair of vectors, each of length equal to the number of rows.
 * The first and second vector contains the minimum and maximum value per row, respectively.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>& mat, const Options& ropt) {
    auto mins = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.nrow());
    auto maxs = mins;
    apply(true, mat, mins.data(), maxs.data(), ropt);
    return std::make_pair(std::move(mins), std::move(maxs));
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>* p, const Options& ropt) {
    return by_row<Output_>(*p, ropt);
}

template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>& mat) {
    return by_row<Output_>(mat, {}); 
}

template<typename Output_ = double, typename Value_, typename Index_>
std::pair<std::vector<Output_>, std::vector<Output_> > by_row(const tatami::Matrix<Value_, Index_>* p) {
    return by_row<Output_>(*p);
}
/**
 * @endcond
 */

}

}

#endif
