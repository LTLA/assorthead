#ifndef QUICKSTATS_MAD_HPP
#define QUICKSTATS_MAD_HPP

#include <cstddef>
#include <cmath>
#include <limits>

#include "median.hpp"

/**
 * @file mad.hpp
 * @brief Compute the median absolute deviation.
 */

namespace quickstats {

/**
 * @brief Options for `mad()` and `mad_with_infinities()`.
 * @tparam Output_ Floating-point type of the output value.
 */
template<typename Output_ = double>
struct MadOptions {
    /**
     * Placeholder value returned by `mad()` when `num_total == 0`.
     * This is also returned when `median` is not finite and `MadOptions::difference_between_infinities_is_zero = false`.
     */
    Output_ placeholder = nan_if_available_else_zero<Output_>();

    /**
     * Assume that all occurrences of infinity of the same sign refer to the same arbitrarily large number,
     * such that a value of infinity has a deviation of zero from a median of infinity (instead of NaN via usual IEEE arithmetic).
     * This ensures that the MAD will be well-defined for a non-empty array, i.e., either 0 or infinity.
     */
    bool difference_between_infinities_is_zero = false;
};

/**
 * Compute the median absolute deviation (MAD) of an array of elements, given its median.
 *
 * No consideration is given to special values like NaNs in the array.
 * If these are to be skipped, consider using `skip_values()` before calling this function.
 *
 * See also `scale_mad_to_sd()` if the MAD is to be used as an estimate of the standard deviation.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Input_ Numeric type of the input values and median.
 * This is generally expected to be floating-point, though it is also possible to use signed integers as long as their differences do not overflow.
 *
 * @param num_total Total number of elements from which to compute a MAD.
 * @param[in] ptr Pointer to an array of length `num_total`.
 * On output, the contents will contain the (possibly reordered) absolute deviations from the median.
 * @param median Median of the array at `ptr`, typically computed with `median()`.
 * For medians that might be infinite, consider using `mad_with_infinities()` instead.
 * @param options Further options.
 *
 * @return MAD of the array.
 * If `num_total == 0`, `MadOptions::placeholder` is returned.
 * If `median` is infinite and `MadOptions::difference_between_infinites_is_zero == false`, `MadOptions::placeholder` is also returned.
 */
template<typename Output_ = double, typename Input_>
Output_ mad(const std::size_t num_total, Input_* const ptr, const Input_ median, const MadOptions<Output_>& options) {
    // Check for infinities to avoid shenanigans from trying to order NaN deviations from Inf - Inf.
    // This entire conditional should be optimized out if Input_ has no infinities.
    if (std::isinf(median)) {
        if (!options.difference_between_infinities_is_zero) {
            return options.placeholder;
        } 

        // We use inf_if_available_else_max() just to get it to compile if Input_ doesn't have infs.
        // At this point, Input_ must support median otherwise we wouldn't have gotten here.
        for (std::size_t i = 0; i < num_total; ++i) {
            ptr[i] = (median == ptr[i] ? 0 : inf_if_available_else_max<Input_>());
        }

    } else {
        for (std::size_t i = 0; i < num_total; ++i) {
            ptr[i] = std::abs(ptr[i] - median);
        }
    }

    MedianOptions<Output_> medopt;
    medopt.placeholder = options.placeholder;
    return ::quickstats::median<Output_>(num_total, ptr, medopt);
}

/**
 * Compute the median absolute deviation (MAD) of a sparse vector, given its median.
 * This vector is assumed to have `num_non_zero` structural non-zeros and `num_total - num_non_zero` zeros.
 *
 * No consideration is given to special values like NaNs in the array.
 * If these are to be skipped, consider using `skip_values()` before calling this function.
 *
 * See also `scale_mad_to_sd()` if the MAD is to be used as an estimate of the standard deviation.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Input_ Numeric type of the input values and median.
 * This is generally expected to be floating-point, though it is also possible to use signed integers as long as their differences do not overflow.
 *
 * @param num_total Total number of elements in the sparse vector.
 * @param num_non_zero Number of structural non-zeros in the sparse vector.
 * This should be no greater than `num_total`.
 * `num_total - num_non_zero` is the number of structural zeros.
 * @param[in] values Pointer to the start of an array of length `num_non_zero`, containing the values of the structural non-zeros of the sparse vector.
 * On output, the contents will contain the (possibly reordered) absolute deviations from the median.
 * @param median Median of the sparse vector at `values`, typically computed with `median()`.
 * For medians that might be infinite, consider using `mad_with_infinities()` instead.
 * @param options Further options.
 *
 * @return MAD of the sparse vector.
 * If `num_total == 0`, `MadOptions::placeholder` is returned.
 * If `median` is infinite and `MadOptions::difference_between_infinites_is_zero == false`, `MadOptions::placeholder` is also returned.
 */
template<typename Output_ = double, typename Input_>
Output_ mad(const std::size_t num_total, const std::size_t num_non_zero, Input_* const values, const Input_ median, const MadOptions<Output_>& options) {
    // Check for infinities to avoid shenanigans from trying to order NaN deviations from Inf - Inf.
    // This entire conditional should be optimized out if Input_ has no infinities.
    if (std::isinf(median)) {
        if (!options.difference_between_infinities_is_zero) {
            return options.placeholder;
        }

        // We use inf_if_available_else_max() just to get it to compile if Input_ doesn't have infs.
        // At this point, Input_ must support median otherwise we wouldn't have gotten here.
        for (std::size_t i = 0; i < num_non_zero; ++i) {
            values[i] = (median == values[i] ? 0 : inf_if_available_else_max<Input_>());
        }

    } else {
        for (std::size_t i = 0; i < num_non_zero; ++i) {
            values[i] = std::abs(values[i] - median);
        }
    }

    // It is also possible to implement the sparse MAD by subtracting 'abs(median)' from the absolute deviations,
    // computing the sparse median of the difference, and then adding 'abs(median)' back to the result.
    // We don't do this as the subtraction and addition introduces some numerical error,
    // which isn't that consequential in practice but interferes with exact comparisons to the dense results in our tests.

    MedianOptions<Output_> medopt;
    medopt.placeholder = options.placeholder;
    return median_internal<Output_>(num_total, num_non_zero, values, std::abs(median), medopt);
}

/**
 * Scale the median absolute deviation (MAD) so that its expected value for a normal distribution is equal to the standard deviation.
 *
 * @tparam Float_ Floating-point type of the MAD.
 *
 * @param x MAD, typically computed from `mad()`.
 *
 * @return Scaled value of `x`.
 */
template<typename Float_>
Float_ scale_mad_to_sd(const Float_ x) {
    return x * 1.4826;
}

/**
 * @cond
 */
// Backwards compatibility.
template<typename Output_ = double, typename Input_>
Output_ mad(const std::size_t num_total, Input_* const ptr, const Input_ median) {
    return mad(num_total, ptr, median, MadOptions<Output_>());
}

template<typename Output_ = double, typename Input_>
Output_ mad(const std::size_t num_total, const std::size_t num_non_zero, Input_* const values, const Input_ median) {
    return mad(num_total, num_non_zero, values, median, MadOptions<Output_>());
}

template<typename Output_ = double, typename Input_>
Output_ mad_with_infinities(const std::size_t num_total, Input_* const ptr, const Input_ median) {
    MadOptions<Output_> opt;
    opt.difference_between_infinities_is_zero = true;
    return mad(num_total, ptr, median, opt);
}

template<typename Output_ = double, typename Input_>
Output_ mad_with_infinities(const std::size_t num_total, const std::size_t num_non_zero, Input_* const values, const Input_ median) {
    MadOptions<Output_> opt;
    opt.difference_between_infinities_is_zero = true;
    return mad(num_total, num_non_zero, values, median, opt);
}
/**
 * @endcond
 */

}

#endif
