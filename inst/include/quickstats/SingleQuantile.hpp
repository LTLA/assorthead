#ifndef QUICKSTATS_SINGLE_QUANTILE_HPP
#define QUICKSTATS_SINGLE_QUANTILE_HPP

#include <cmath>
#include <cassert>
#include <type_traits>
#include <stdexcept>
#include <optional>
#include <limits>
#include <cstddef>
#include <vector>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file SingleQuantile.hpp
 * @brief Compute a single quantile.
 */

namespace quickstats {

/**
 * @cond
 */
template<typename Output_>
void configure_single_quantile(
    const Output_ quantile,
    const std::size_t num_m1,
    std::size_t& upper_index,
    Output_& upper_fraction,
    bool& skip_lower
) {
    if (quantile < 0 || quantile > 1) {
        throw std::out_of_range("'quantile' should lie in [0, 1]");
    }

    const Output_ raw_index = static_cast<Output_>(num_m1) * quantile;
    const Output_ raw_upper_index = std::ceil(raw_index);
    const Output_ raw_lower_index = std::floor(raw_index);

    // Protect the cast from conversion imprecision between size_t and Float_,
    // e.g., if it rounds up or if it converts it to an Inf.
    const auto converted_upper_index = sanisizer::from_float<std::size_t>(raw_upper_index);
    if (converted_upper_index <= num_m1) {
        upper_index = converted_upper_index;
        upper_fraction = raw_index - raw_lower_index;
        skip_lower = (raw_upper_index == raw_lower_index);
    } else {
        // Just gently cap it at the maximum value.
        upper_index = num_m1;
        upper_fraction = 0;
        skip_lower = true;
    }
}
/**
 * @endcond
 */

/**
 * @brief Calculate a single quantile from a fixed number of elements.
 *
 * @tparam Output_ Floating-point type of the output quantile.
 *
 * A `SingleQuantileFixedNumber` instance computes a type 7 quantile, consistent with the default in R's `quantile` function.
 */
template<typename Output_ = double>
class SingleQuantileFixedNumber {
public:
    /**
     * @param num_total Total number of elements from which to compute a quantile.
     * This should be positive.
     * @param quantile Probability of the quantile to compute, in \f$[0, 1]\f$.
     */
    SingleQuantileFixedNumber(const std::size_t num_total, const Output_ quantile) : my_num_total(num_total) {
        static_assert(std::is_floating_point<Output_>::value);

        if (num_total <= 0) {
            throw std::out_of_range("'num' should be positive");
        }
        const std::size_t num_m1 = num_total - 1;

        configure_single_quantile(quantile, num_m1, my_upper_index, my_upper_fraction, my_skip_lower);
    }

private:
    std::size_t my_num_total, my_upper_index;
    Output_ my_upper_fraction;
    bool my_skip_lower;

public:
    /**
     * Compute the quantile of interest from a dense array.
     *
     * No consideration is given to special values like NaNs in the dense array.
     * If these are to be skipped, consider using `skip_values()` before calling this method.
     *
     * @tparam Input_ Numeric type of the input values.
     *
     * @param[in] ptr Pointer to the start of an array of length `num_total`.
     * This should not contain any NaN values.
     * On output, the elements may be reordered.
     *
     * @return Quantile of the specified probability. 
     */
    template<typename Input_>
    Output_ operator()(Input_* const ptr) const {
        const auto target = ptr + my_upper_index;
        std::nth_element(ptr, target, ptr + my_num_total);

        // Avoid an extra memory access to get the lower index if we don't need it; this would also fail if upper_index = 0.
        const Output_ upper = *target;
        if (my_skip_lower) {
            return upper;
        }

        const Output_ lower = *std::max_element(ptr, target); 
        return interpolate(lower, upper, my_upper_fraction);
    }

    /**
     * Overload to compute the desired quantile from a sparse vector of length `num_total`.
     * This vector should have `num_non_zero` structural non-zeros and `num_total - num_non_zero` structural zeros.
     *
     * No consideration is given to special values like NaNs in the structural non-zeros.
     * If these are to be skipped, consider using `skip_values()` before calling this method.
     *
     * @tparam Input_ Numeric type of the input values.
     *
     * @param num_non_zero Number of structural non-zeros in the sparse vector.
     * This should be no greater than `num_total`.
     * @param[in] ptr Pointer to the start of an array of length `num_non_zero`, containing the values of the structural non-zeros of the sparse vector.
     * It should not contain any NaN values.
     * On output, the elements may be reordered.
     *
     * @return Quantile of the specified probability.
     */
    template<typename Input_>
    Output_ operator()(const std::size_t num_non_zero, Input_* const ptr) const {
        assert(num_non_zero <= my_num_total);
        if (num_non_zero == my_num_total) {
            return operator()(ptr);
        } else if (num_non_zero == 0) {
            return 0;
        }

        std::size_t num_negative = 0;
        for (std::size_t i = 0; i < num_non_zero; ++i) {
            num_negative += (ptr[i] < 0);
        }

        if (my_upper_index < num_negative) {
            const auto target = ptr + my_upper_index;
            std::nth_element(ptr, target, ptr + num_non_zero);

            const Output_ upper = *target;
            if (my_skip_lower) {
                return upper;
            }

            const Output_ lower = *std::max_element(ptr, target); 
            return interpolate(lower, upper, my_upper_fraction);
        }

        if (num_negative && my_upper_index == num_negative) {
            // The upper value is zero. It can't be positive, as that would
            // imply that there are no structural zeros and num_non_zero == my_num_total.
            if (my_skip_lower) {
                return 0;
            }

            const auto target = ptr + (my_upper_index - 1);
            std::nth_element(ptr, target, ptr + num_non_zero);
            return static_cast<Output_>(*target) * (1 - my_upper_fraction);
        }

        const std::size_t num_zeros = my_num_total - num_non_zero;
        const std::size_t num_not_positive = num_zeros + num_negative;
        if (my_upper_index < num_not_positive) {
            return 0;
        }

        const auto target = ptr + (my_upper_index - num_zeros);
        std::nth_element(ptr, target, ptr + num_non_zero);
        const Output_ upper = *target;
        if (my_skip_lower) {
            return upper;
        }

        if (my_upper_index == num_not_positive) {
            return upper * my_upper_fraction;
        }

        const Output_ lower = *std::max_element(ptr, target); 
        return interpolate(lower, upper, my_upper_fraction);
    }
};

/**
 * @brief Options for `SingleQuantileVariableNumber`.
 * @tparam Output_ Floating-point type of the output quantile.
 */
template<typename Output_ = double>
struct SingleQuantileVariableNumberOptions {
    /**
     * Placeholder value to return when `num_total == 0` in the `SingleQuantileVariableNumber::operator()()` method.
     */
    Output_ placeholder = nan_if_available_else_zero<Output_>();
};

/**
 * @brief Calculate a quantile from a variable number of elements.
 *
 * @tparam Output_ Floating-point type of the output quantile.
 *
 * This uses the same logic as the `SingleQuantileFixedNumber` class but supports any number of elements, from zero up to a specified maximum.
 */
template<typename Output_>
class SingleQuantileVariableNumber {
public:
    /**
     * @param max_num_total Maximum of the total number of elements. 
     * Unlike `SingleQuantileFixedNumber`, this may also be zero.
     * @param quantile Probability of the quantile to compute, in \f$[0, 1]\f$.
     * @param options Further options.
     */
    SingleQuantileVariableNumber(const std::size_t max_num_total, const Output_ quantile, const SingleQuantileVariableNumberOptions<Output_>& options) :
        my_quantile(quantile),
        my_placeholder(options.placeholder)
    {
        if (max_num_total >= 2) {
            sanisizer::resize(my_choices, max_num_total - 1);
        }
    }

    /**
     * @cond
     */
    SingleQuantileVariableNumber(const std::size_t max_num_total, const Output_ quantile) : SingleQuantileVariableNumber(max_num_total, quantile, {}) {}
    /**
     * @endcond
     */

private:
    std::vector<std::optional<SingleQuantileFixedNumber<Output_> > > my_choices;
    Output_ my_quantile;
    Output_ my_placeholder;

public:
    /**
     * Compute a single quantile from a dense array of length equal to `num_total`.
     * 
     * No consideration is given to special values like NaNs in the array.
     * If these are to be skipped, consider using `skip_values()` before calling this method.
     *
     * This method is not thread-safe.
     *
     * @tparam Input_ Numeric type of the input values.
     *
     * @param num_total Total number of the elements from which to compute the quantile.
     * This should be no greater than `max_num_total`.
     * @param[in] ptr Pointer to an array of length `num_total`.
     * This should not contain NaN values.
     * On output, the elements may be reordered.
     *
     * @return Quantile of the specified probability. 
     * If `num_total == 0`, `SingleQuantileVariableNumberOptions::placeholder` is returned instead.
     */
    template<typename Input_>
    Output_ operator()(const std::size_t num_total, Input_* ptr) {
        if (num_total == 0) {
            return my_placeholder;
        } else if (num_total == 1) {
            return *ptr;
        } else {
            assert(sanisizer::is_less_than_or_equal(num_total - 1, my_choices.size())); 
            auto& ocalc = my_choices[num_total - 2];
            if (!ocalc.has_value()) { // Instantiating them on demand.
                ocalc.emplace(num_total, my_quantile);
            }
            return (*ocalc)(ptr);
        }
    }

    /**
     * Overload to compute the desired quantile from a sparse vector of length `num_total`.
     * This vector is assumed to have `num_non_zero` structural non-zeros and `num_total - num_non_zero` zeros.
     *
     * No consideration is given to special values like NaNs in the values of the structural non-zeros.
     * If these are to be skipped, consider using `skip_values()` before calling this method.
     *
     * This method is not thread-safe.
     *
     * @tparam Input_ Numeric type of the input values.
     *
     * @param num_total Total number of elements from which to compute the quantile.
     * This should be no greater than `max_num_total`.
     * @param num_non_zero Number of elements that are structural non-zero elements.
     * This should be no greater than `num_total`.
     * @param[in] values Pointer to the start of an array of length `num_total`.
     * This is expected to contain the values of the structural non-zero elements of the sparse vector.
     * It should not contain any NaN values.
     * On output, the elements may be reordered.
     *
     * @return Quantile of the specified probability. 
     * If `num_total == 0`, `SingleQuantileVariableNumberOptions::placeholder` is returned instead.
     */
    template<typename Input_>
    Output_ operator()(const std::size_t num_total, const std::size_t num_non_zero, Input_* const values) {
        if (num_total == 0) {
            return my_placeholder;
        } else if (num_total == 1) {
            return (num_non_zero ? *values : 0);
        } else {
            assert(sanisizer::is_less_than_or_equal(num_total - 1, my_choices.size())); 
            auto& ocalc = my_choices[num_total - 2];
            if (!ocalc.has_value()) { // Instantiating them on demand.
                ocalc.emplace(num_total, my_quantile);
            }
            return (*ocalc)(num_non_zero, values);
        }
    }
};

}

#endif
