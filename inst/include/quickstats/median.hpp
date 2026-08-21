#ifndef QUICKSTATS_MEDIAN_HPP
#define QUICKSTATS_MEDIAN_HPP

#include <algorithm>
#include <type_traits>
#include <cassert>
#include <cstddef>

#include "utils.hpp"

/**
 * @file median.hpp
 * @brief Compute a median.
 */

namespace quickstats {

/**
 * @brief Options for `median()`.
 * @tparam Output_ Floating-point type of the output value.
 */
template<typename Output_ = double>
struct MedianOptions {
    /**
     * Placeholder value returned by `median()` when `num_total == 0`. 
     */
    Output_ placeholder = nan_if_available_else_zero<Output_>();
};

/**
 * Compute the median of an array of elements.
 *
 * No consideration is given to special values like NaNs in the array.
 * If these are to be skipped, consider using `skip_values()` before calling this function.
 *
 * @param num_total Total number of elements from which to compute a median.
 * @param[in] ptr Pointer to an array of length `num_total`.
 * On output, the contents may be reordered.
 * @param options Further options.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Input_ Numeric type of the input values.
 *
 * @return Median of values in `[ptr, ptr + num_total)`, or `MedianOptions::placeholder` if `num_total == 0`.
 */
template<typename Output_ = double, typename Input_>
Output_ median(const std::size_t num_total, Input_* const ptr, const MedianOptions<Output_>& options) {
    static_assert(std::is_floating_point<Output_>::value);

    if (num_total == 0) {
        return options.placeholder;
    }

    const std::size_t halfway = num_total / 2;
    const bool is_even = (num_total % 2 == 0);

    std::nth_element(ptr, ptr + halfway, ptr + num_total);
    const Output_ medtmp = *(ptr + halfway);
    if (!is_even) {
        return medtmp;
    }

    // 'nth_element()' reorganizes 'ptr' so that everything below 'halfway' is
    // less than or equal to 'ptr[halfway]', while everything above 'halfway'
    // is greater than or equal to 'ptr[halfway]'. Thus, to get the element
    // immediately before 'halfway' in the sort order, we just need to find the
    // maximum from '[0, halfway)'.
    const Output_ other = *std::max_element(ptr, ptr + halfway);

    return interpolate<Output_>(medtmp, other, 0.5);
}

/**
 * @cond
 */
// We support an arbitrary zero_value so that the MAD calculation can re-use this for the sparse calculations, see mad.hpp for details.
template<typename Output_, typename Input_>
Output_ median_internal(const std::size_t num_total, const std::size_t num_non_zero, Input_* const non_zero_values, const Input_ zero_value, const MedianOptions<Output_>& options) {
    assert(num_total >= num_non_zero);

    // Fallback to the dense code if there are no structural zeros. This is not
    // just for efficiency as the downstream averaging code assumes that there
    // is at least one structural zero when considering its scenarios.
    if (num_non_zero == num_total) {
        return median<Output_>(num_total, non_zero_values, options);
    }

    // Is the number of non-zeros less than the number of zeros?
    // If so, the median must be equal to the zero value.
    // Note that we calculate it in this way to avoid overflow from 'num_non_zero * 2'.
    if (num_non_zero < num_total - num_non_zero) {
        return zero_value;
    } 

    const std::size_t halfway = num_total / 2;
    const bool is_even = (num_total % 2 == 0);

    const std::size_t num_zero = num_total - num_non_zero;
    std::size_t num_below = 0;
    for (std::size_t i = 0; i < num_non_zero; ++i) {
        num_below += (non_zero_values[i] < zero_value);
    }

    if (!is_even) {
        if (num_below > halfway) {
            std::nth_element(non_zero_values, non_zero_values + halfway, non_zero_values + num_non_zero);
            return non_zero_values[halfway];

        } else if (halfway >= num_below + num_zero) {
            const std::size_t skip_zeros = halfway - num_zero;
            std::nth_element(non_zero_values, non_zero_values + skip_zeros, non_zero_values + num_non_zero);
            return non_zero_values[skip_zeros];

        } else {
            return zero_value;
        }
    }

    Output_ baseline = zero_value, other = zero_value;
    if (num_below > halfway) { // both halves of the median are below the zero value.
        std::nth_element(non_zero_values, non_zero_values + halfway, non_zero_values + num_non_zero);
        baseline = non_zero_values[halfway];
        other = *(std::max_element(non_zero_values, non_zero_values + halfway)); // max_element gets the sorted value at halfway - 1, see explanation for the dense case.

    } else if (num_below == halfway) { // the upper half is guaranteed to be the zero value.
        const std::size_t below_halfway = halfway - 1;
        std::nth_element(non_zero_values, non_zero_values + below_halfway, non_zero_values + num_non_zero);
        other = non_zero_values[below_halfway]; // set to other so that, in the common case of zero_value = 0, addition/subtraction of baseline = 0 has no effect on precision. 

    } else if (num_below < halfway && num_below + num_zero > halfway) { // both halves are the zero value, so the zero value is the median.
        ;

    } else if (num_below + num_zero == halfway) { // the lower half is guaranteed to be the zero value.
        const std::size_t skip_zeros = halfway - num_zero;
        std::nth_element(non_zero_values, non_zero_values + skip_zeros, non_zero_values + num_non_zero);
        other = non_zero_values[skip_zeros]; // set to other so that, in the common case of zero_value = 0, addition/subtraction of baseline = 0 has no effect on precision. 

    } else { // both halves of the median are above the zero value.
        const std::size_t skip_zeros = halfway - num_zero;
        std::nth_element(non_zero_values, non_zero_values + skip_zeros, non_zero_values + num_non_zero);
        baseline = non_zero_values[skip_zeros];
        other = *(std::max_element(non_zero_values, non_zero_values + skip_zeros)); // max_element gets the sorted value at skip_zeros - 1, see explanation for the dense case.
    }

    return interpolate<Output_>(baseline, other, 0.5);
}
/**
 * @endcond
 */

/**
 * Compute the median of a sparse vector.
 * This vector is assumed to have `num_non_zero` structural non-zeros and `num_total - num_non_zero` zeros.
 *
 * No consideration is given to special values like NaNs in the array.
 * If these are to be skipped, consider using `skip_values()` before calling this function.
 *
 * @param num_total Total number of elements in the sparse vector.
 * @param num_non_zero Number of structural non-zeros in the sparse vector.
 * This should be no greater than `num_total`.
 * `num_total - num_non_zero` is the number of structural zeros.
 * @param[in] values Pointer to the start of an array of length `num_non_zero`, containing the values of the structural non-zeros of the sparse vector.
 * On output, the contents may be reordered.
 * @param options Further options.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Input_ Numeric type of the input values.
 *
 * @return Median of the sparse vector, or `MedianOptions::placeholder` if `num_total == 0`.
 */
template<typename Output_ = double, typename Input_>
Output_ median(const std::size_t num_total, const std::size_t num_non_zero, Input_* const values, const MedianOptions<Output_>& options) {
    return median_internal<Output_>(num_total, num_non_zero, values, static_cast<Input_>(0), options);
}

/**
 * @cond
 */
// Backwards compatibility.
template<typename Output_ = double, typename Input_>
Output_ median(const std::size_t num_total, Input_* const ptr) {
    return median(num_total, ptr, MedianOptions<Output_>());
}

template<typename Output_ = double, typename Input_>
Output_ median(const std::size_t num_total, const std::size_t num_non_zero, Input_* const values) {
    return median(num_total, num_non_zero, values, MedianOptions<Output_>());
}
/**
 * @endcond
 */

}

#endif
