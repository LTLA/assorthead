#ifndef QUICKSTATS_RSS_HPP
#define QUICKSTATS_RSS_HPP

#include <cassert>
#include <limits>
#include <cstddef>
#include <algorithm>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"
#include "pairwise_sum.hpp"

/**
 * @file rss.hpp
 * @brief Compute residual sum of squares.
 */

namespace quickstats {

/** 
 * @brief Result of `rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct RssResult { 
    /**
     * Sample mean.
     * This is set to `RssOptions::mean_placeholder` if there are no input values.
     */
    Output_ mean = 0;

    /**
     * Residual sum of squares from the mean.
     * This is set to 0 if there are no input values.
     */
    Output_ rss = 0;
};

/** 
 * @brief Re-usable workspace for `rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct RssWorkspace {
    /**
     * @cond
     */
    PairwiseSumWorkspace<Output_> pswork;
    /**
     * @endcond
     */
};

/** 
 * @brief Options for `rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct RssOptions {
    /**
     * Maximum number of elements to sum in sequence, see `PairwiseSumOptions::max_sum_length` for details.
     */
    std::size_t max_sum_length = 128;

    /**
     * Placeholder value for `RssResult::mean` when `num_total == 0` in `rss()`. 
     */
    Output_ mean_placeholder = nan_if_available_else_zero<Output_>();
};

/**
 * Compute the residual sum of squares from a sparse vector.
 * This uses the standard two-pass algorithm with naive accumulation of the sum of squared differences;
 * thus, it is best used with a sufficiently high-precision `Output_` like `double`.
 *
 * No consideration is given to special values like NaNs in the values of the structural non-zeros.
 * If these are to be skipped, consider using `skip_values()` before calling this method.
 *
 * @tparam accumulators_ Number of accumulators, see `pairwise_sum()` for details.
 * @tparam Input_ Numeric type of the input values.
 * @tparam Output_ Floating-point type of the output data.
 *
 * @param num_total Total number of elements in the sparse vector.
 * @param num_non_zero Number of structural non-zeros in the sparse vector.
 * This should be no greater than `num_total`.
 * `num_total - num_non_zero` is the number of structural zeros.
 * @param[in] ptr Pointer to an array of length `num_non_zero`, containing the values of the structural non-zeros in the sparse vector.
 * @param work Workspace that can be re-used across multiple `rss()` calls.
 * @param options Further options.
 *
 * @return The sample mean and residual sum of squares of the sparse vector.
 */
template<std::size_t accumulators_ = 4, typename Input_, typename Output_>
RssResult<Output_> rss(const std::size_t num_total, const std::size_t num_non_zero, const Input_* const ptr, RssWorkspace<Output_>& work, const RssOptions<Output_>& options) {
    static_assert(std::is_floating_point<Output_>::value);

    RssResult<Output_> output;
    if (num_total == 0) {
        output.mean = options.mean_placeholder;
        return output;
    }

    Output_& mean = output.mean;
    PairwiseSumOptions psopt;
    psopt.max_sum_length = options.max_sum_length;
    mean = pairwise_sum<accumulators_>(num_non_zero, ptr, work.pswork, psopt);
    mean /= num_total;

    Output_& ssd = output.rss;
    ssd = pairwise_sum_abstract<accumulators_>(
        num_non_zero,
        [&](const std::size_t i) -> Output_ {
            const auto delta = static_cast<Output_>(ptr[i]) - mean;
            return delta * delta;
        },
        work.pswork,
        psopt
    );

    assert(num_non_zero <= num_total);
    if (num_non_zero < num_total) {
        ssd += static_cast<Output_>(num_total - num_non_zero) * mean * mean;
    }

    return output;
}

/**
 * Compute the residual sum of squares from the mean from a dense array.
 * This uses the standard two-pass algorithm with naive accumulation of the sum of squared differences;
 * thus, it is best used with a sufficiently high-precision `Output_` like `double`.
 *
 * No consideration is given to special values like NaNs in the dense array.
 * If these are to be skipped, consider using `skip_values()` before calling this method.
 *
 * @tparam accumulators_ Number of accumulators, see `pairwise_sum()` for details.
 * @tparam Input_ Numeric type of the input values.
 * @tparam Output_ Floating-point type of the output data.
 *
 * @param num_total Total number of elements in the array.
 * @param[in] ptr Pointer to an array of length `num_total`.
 * @param work Workspace that can be re-used across multiple `rss()` calls.
 * @param options Further options.
 *
 * @return The sample mean and residual sum of squares of the array.
 */
template<std::size_t accumulators_ = 4, typename Input_, typename Output_>
RssResult<Output_> rss(const std::size_t num_total, const Input_* const ptr, RssWorkspace<Output_>& work, const RssOptions<Output_>& options) {
    return rss<accumulators_>(num_total, num_total, ptr, work, options);
}

/**
 * @cond
 */
// For back-compatibility.
template<std::size_t limit_ = 128, std::size_t accumulators_ = 4, typename Input_, typename Output_>
RssResult<Output_> rss(const std::size_t num_total, const std::size_t num_non_zero, const Input_* const ptr, RssWorkspace<Output_>& work) {
    return rss<accumulators_>(
        num_total,
        num_non_zero,
        ptr,
        work,
        [&]{
            RssOptions<Output_> options;
            options.max_sum_length = limit_;
            return options;
        }()
    );
}

template<std::size_t limit_ = 128, std::size_t accumulators_ = 4, typename Input_, typename Output_>
RssResult<Output_> rss(const std::size_t num_total, const Input_* const ptr, RssWorkspace<Output_>& work) {
    return rss<accumulators_>(
        num_total,
        ptr,
        work,
        [&]{
            RssOptions<Output_> options;
            options.max_sum_length = limit_;
            return options;
        }()
    );
}
/**
 * @endcond
 */

/**
 * Update the mean and RSS by adding a new value using Welford's method.
 *
 * @param mean On input, the mean of previous values.
 * If no previous values were provided, this should be set to zero. 
 * On output, the updated mean after including the latest value.
 * @param rss On input, the RSS of previous values.
 * If no previous values were provided, this should be set to zero. 
 * On output, the updated RSS after including the latest value.
 * @param value New value to update the mean/RSS.
 * @param num_after Number of values used to compute the updated mean/RSS, after adding the latest `value`.
 * This should always be positive.
 */
template<typename Output_ = double, typename Input_, typename Count_>
void update_rss(Output_& mean, Output_& rss, const Input_ value, const Count_ num_total) {
    assert(num_total > 0);
    Output_ delta = static_cast<Output_>(value) - mean;
    mean += delta / num_total;
    rss += delta * (static_cast<Output_>(value) - mean);
}

/**
 * Update the mean and RSS by adding any number of zeros using Welford's method.
 * This assumes that `num_total > 0`; if this cannot be guaranteed, use `update_rss_with_zeros()` instead.
 *
 * @param mean On input, the mean of previous values.
 * If no previous values were provided, this should be set to zero. 
 * On output, the updated mean after including the zeros.
 * @param rss On input, the RSS of previous values.
 * If no previous values were provided, this should be set to zero. 
 * On output, the updated RSS after including the zeros.
 * @param num_zeros Number of zeros to be added.
 * This may be zero.
 * @param num_total Number of values used to compute the updated mean/RSS, after adding any number of zeros.
 * This should be positive and no less than `num_total`.
 */
template<typename Output_ = double, typename Count_>
void update_rss_with_zeros_unsafe(Output_& mean, Output_& rss, const Count_ num_zeros, const Count_ num_total) {
    assert(num_total > 0);
    assert(num_total >= num_zeros);
    const auto ratio = static_cast<Output_>(num_total - num_zeros) / static_cast<Output_>(num_total);
    rss += mean * mean * ratio * num_zeros;
    mean *= ratio;
}

/**
 * Update the mean and RSS with any number of zeros using Welford's method.
 * This is a slightly slower version of `update_rss_with_zeros_unsafe()` that avoids problems if `num_total == 0`.
 *
 * @param mean On input, the mean of previous values.
 * If no previous values were provided, this should be set to zero. 
 * On output, the updated mean after including the zeros.
 * @param rss On input, the RSS of previous values.
 * If no previous values were provided, this should be set to zero. 
 * On output, the updated RSS after including the zeros.
 * @param num_before Number of values used to compute the input values of `mean` and `rss`. 
 * This may be zero.
 * @param num_total Number of values used to compute the updated mean/RSS, after adding any number of zeros.
 * This may be zero but should be no less than `num_total`.
 */
template<typename Output_ = double, typename Count_>
void update_rss_with_zeros(Output_& mean, Output_& rss, const Count_ num_zeros, const Count_ num_total) {
    if (num_total) { 
        update_rss_with_zeros_unsafe(mean, rss, num_zeros, num_total);
    }
}

/**
 * @cond
 */
template<typename Input_, typename Output_ = double>
class RssRunningDense {
public:
    RssRunningDense(const std::size_t num_obj, Output_* const mean, Output_* const rss) :
        my_num_obj(num_obj),
        my_mean(mean),
        my_rss(rss)
    {
        assert(check_zeroed(num_obj, mean));
        assert(check_zeroed(num_obj, rss));
    }

public:
    void add(const Input_* const ptr) {
        my_count = sanisizer::sum<std::size_t>(my_count, 1);
        for (std::size_t i = 0; i < my_num_obj; ++i) {
            update_rss(my_mean[i], my_rss[i], ptr[i], my_count);
        }
    }

    void finish() {
        finish(nan_if_available_else_zero<Output_>());
    }

    void finish(const Output_ mean_placeholder) {
        if (my_count == 0 && mean_placeholder != 0) { // my_mean should already be zeroed, so no need to fill if our placeholder is also zero.
            std::fill_n(my_mean, my_num_obj, mean_placeholder);
        }
    }

    std::size_t num_obs() const {
        return my_count;
    }

private:
    std::size_t my_num_obj;
    Output_* my_mean;
    Output_* my_rss;
    std::size_t my_count = 0;

    static_assert(std::is_floating_point<Output_>::value);
};

template<typename Count_, typename Input_, typename Output_ = double>
class RssRunningDenseSkip {
public:
    RssRunningDenseSkip(const std::size_t num_obj, Output_* mean, Output_* rss, Count_* num_unskipped) :
        my_num_obj(num_obj),
        my_mean(mean),
        my_rss(rss),
        my_num_unskipped(num_unskipped)
    {
        assert(check_zeroed(num_obj, mean));
        assert(check_zeroed(num_obj, rss));
        assert(check_zeroed(num_obj, num_unskipped));
    }

public:
    template<class Skip_>
    void add(const Input_* ptr, Skip_ skip) {
        // my_count is the upper bound of all my_num_unskipped, so we check it once here to avoid having to check it in the loop.
        my_count = sanisizer::sum<Count_>(my_count, 1);

        for (std::size_t i = 0; i < my_num_obj; ++i) {
            const auto val = ptr[i];
            if (!skip(i, val)) {
                update_rss(my_mean[i], my_rss[i], val, ++(my_num_unskipped[i]));
            }
        }
    }

    void finish() {
        finish(nan_if_available_else_zero<Output_>());
    }

    void finish(const Output_ mean_placeholder) {
        if (mean_placeholder != 0) { // my_mean should already be zeroed, so no need to fill if our placeholder is also zero.
            if (my_count == 0) {
                std::fill_n(my_mean, my_num_obj, mean_placeholder);
            } else {
                for (std::size_t i = 0; i < my_num_obj; ++i) {
                    if (my_num_unskipped[i] == 0) {
                        my_mean[i] = mean_placeholder;
                    }
                }
            }
        }
    }

    Count_ num_obs() const {
        return my_count;
    }

private:
    std::size_t my_num_obj;
    Output_* my_mean;
    Output_* my_rss;
    Count_ my_count = 0;
    Count_* my_num_unskipped;

    static_assert(std::is_integral<Count_>::value);
    static_assert(std::is_floating_point<Output_>::value);
};

template<typename Count_, typename Input_, typename Output_ = double>
class RssRunningSparse {
public:
    RssRunningSparse(const std::size_t num_obj, Output_* const mean, Output_* const rss, Count_* const num_non_zero) : 
        my_num_obj(num_obj),
        my_mean(mean),
        my_rss(rss),
        my_num_non_zero(num_non_zero)
    {
        assert(check_zeroed(num_obj, mean));
        assert(check_zeroed(num_obj, rss));
        assert(check_zeroed(num_obj, num_non_zero));
    }

    template<typename Index_>
    void add(const std::size_t num_non_zero_obs, const Input_* const value, const Index_* const index) {
        static_assert(std::is_integral<Index_>::value);

        // my_count is the upper bound of all my_num_non_zero, so no need to check individual increments.
        my_count = sanisizer::sum<Count_>(my_count, 1);

        for (std::size_t i = 0; i < num_non_zero_obs; ++i) {
            const auto ri = index[i]; 
            update_rss(my_mean[ri], my_rss[ri], value[i], ++(my_num_non_zero[ri]));
        }
    }

    void finish() {
        finish(nan_if_available_else_zero<Output_>());
    }

    void finish(const Output_ mean_placeholder) {
        if (my_count == 0) {
            if (mean_placeholder != 0) { // my_mean should already be zeroed, so no need to fill again if the placeholder is also zero.
                std::fill_n(my_mean, my_num_obj, mean_placeholder);
            }
        } else {
            for (std::size_t i = 0; i < my_num_obj; ++i) {
                update_rss_with_zeros_unsafe(my_mean[i], my_rss[i], static_cast<Count_>(my_count - my_num_non_zero[i]), my_count);
            }
        }
    }

    Count_ num_obs() const {
        return my_count;
    }

private:
    std::size_t my_num_obj;
    Output_* my_mean;
    Output_* my_rss;
    Count_* my_num_non_zero;
    Count_ my_count = 0;

    static_assert(std::is_integral<Count_>::value);
    static_assert(std::is_floating_point<Output_>::value);
};

template<typename Count_, typename Input_, typename Output_ = double>
class RssRunningSparseSkip {
public:
    RssRunningSparseSkip(const std::size_t num_obj, Output_* const mean, Output_* const rss, Count_* const num_non_zero, Count_* const num_unskipped) : 
        my_num_obj(num_obj),
        my_mean(mean),
        my_rss(rss),
        my_num_non_zero(num_non_zero),
        my_num_unskipped(num_unskipped)
    {
        assert(check_zeroed(num_obj, mean));
        assert(check_zeroed(num_obj, rss));
        assert(check_zeroed(num_obj, num_non_zero));
        assert(check_zeroed(num_obj, num_unskipped));
    }

    template<typename Index_, class Skip_>
    void add(const std::size_t num_non_zero_obs, const Input_* value, const Index_* index, Skip_ skip) {
        static_assert(std::is_integral<Index_>::value);

        // my_count is the upper bound of all my_num_non_zero, so no need to check individual increments.
        my_count = sanisizer::sum<Count_>(my_count, 1);

        for (std::size_t i = 0; i < num_non_zero_obs; ++i) {
            const auto val = value[i];
            const auto ri = index[i]; 
            if (skip(ri, val)) {
                ++my_num_unskipped[ri]; // storing the number that was skipped so we don't have to add the zeros later.
            } else {
                update_rss(my_mean[ri], my_rss[ri], val, ++(my_num_non_zero[ri]));
            }
        }
    }

    void finish() {
        finish(nan_if_available_else_zero<Output_>());
    }

    void finish(const Output_ mean_placeholder) {
        for (std::size_t i = 0; i < my_num_obj; ++i) {
            my_num_unskipped[i] = my_count - my_num_unskipped[i];
        }

        if (my_count == 0) {
            if (mean_placeholder != 0) { // my_mean should already be zeroed, so no need to do a fill if the placeholder is zero.
                std::fill_n(my_mean, my_num_obj, mean_placeholder);
            }
        } else {
            for (std::size_t i = 0; i < my_num_obj; ++i) {
                if (my_num_unskipped[i] == 0) {
                    my_mean[i] = mean_placeholder;
                } else {
                    const auto num_unskipped = my_num_unskipped[i];
                    update_rss_with_zeros_unsafe(my_mean[i], my_rss[i], static_cast<Count_>(num_unskipped - my_num_non_zero[i]), num_unskipped);
                }
            }
        }
    }

    Count_ num_obs() const {
        return my_count;
    }

private:
    std::size_t my_num_obj;
    Output_* my_mean;
    Output_* my_rss;
    Count_* my_num_non_zero;
    Count_* my_num_unskipped;
    Count_ my_count = 0;

    static_assert(std::is_integral<Count_>::value);
    static_assert(std::is_floating_point<Output_>::value);
};
/**
 * @endcond
 */

/**
 * Recenter the residual sum of squares, i.e., sum of squares from a different mean.
 * This is typically used to combine RSS values from different subsets of the data, e.g., when splitting calculations across cores.
 * In such cases, the global mean across the entire dataset should be used as the new mean,
 * and the sum of the recentered RSS values will be the RSS of the entire dataset.
 * (This approach is more numerically stable than computing the sum of squared observations and then computing the difference with the squared mean.)
 *
 * This function is considered "unsafe" as it assumes that `old_mean` is zero when `num_total == 0`.
 * In many cases, `old_mean` will be NaN when `num_total == 0` due to division by zero.
 * If `num_total > 0` or `old_mean == 0` cannot be guaranteed, consider using `recenter_rss()` instead.
 *
 * @param num_total Total number of elements used to compute the RSS.
 * @param old_rss The old value of the RSS.
 * @param old_mean The old mean used to compute the RSS.
 * @param new_mean The new mean. 
 *
 * @tparam Float_ Floating-point type of the various statistics.
 *
 * @return The recentered RSS.
 */
template<typename Float_>
Float_ recenter_rss_unsafe(const std::size_t num_total, const Float_ old_rss, const Float_ old_mean, const Float_ new_mean) {
    assert(num_total > 0 || old_mean == 0);
    const Float_ delta = old_mean - new_mean;
    return old_rss + num_total * delta * delta;
}

/**
 * Recenter the residual sum of squares, i.e., sum of squares from a different mean.
 * This is a safer version of `recenter_rss_unsafe()` that correctly handles `num_total == 0`, at the cost of some performance.
 *
 * @param num_total Total number of elements used to compute the RSS.
 * This should be non-negative.
 * @param old_rss The old value of the RSS.
 * @param old_mean The old mean used to compute the RSS.
 * This is ignored if `num_total == 0`.
 * @param new_mean The new mean. 
 *
 * @tparam Float_ Floating-point type of the various statistics.
 *
 * @return The recentered RSS, or `old_rss` (which should be zero) if `num_total == 0`.
 */
template<typename Float_>
Float_ recenter_rss(const std::size_t num_total, const Float_ old_rss, const Float_ old_mean, const Float_ new_mean) {
    if (num_total == 0) {
        return old_rss;
    } else {
        return recenter_rss_unsafe(num_total, old_rss, old_mean, new_mean);
    }
}

/**
 * @cond
 */
template<typename Float_>
Float_ rss_to_variance(const std::size_t num_total, const Float_ rss) {
    if (num_total <= 1) {
        return std::numeric_limits<Float_>::quiet_NaN();
    } else {
        return rss / (num_total - 1);
    }
}

template<typename Float_>
void rss_to_variance(const std::size_t num_obj, const std::size_t num_total, Float_* const rss) {
    if (num_total <= 1) {
        std::fill_n(rss, num_obj, std::numeric_limits<Float_>::quiet_NaN());
    } else {
        // For consistency with the other overloads, we won't do the '* (1/denom)' trick.
        // It shouldn't have much effect on throughput anyway as the bottleneck should be reading from memory.
        for (std::size_t i = 0; i < num_obj; ++i) {
            rss[i] /= num_total - 1;
        }
    }
}

template<typename Count_, typename Float_>
void rss_to_variance(const std::size_t num_obj, const Count_* const num_total, Float_* const rss) {
    for (std::size_t i = 0; i < num_obj; ++i) {
        rss[i] = rss_to_variance(num_total[i], rss[i]);
    }
}
/**
 * @endcond
 */

}

#endif
