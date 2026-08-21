#ifndef QUICKSTATS_UTILS_HPP
#define QUICKSTATS_UTILS_HPP

#include <type_traits>
#include <cmath>
#include <cstddef>
#include <limits>

/**
 * @file utils.hpp
 * @brief Miscellaneous utilities.
 */

namespace quickstats {

/**
 * @cond
 */
template<typename Value_>
using I = std::remove_cv_t<std::remove_reference_t<Value_> >;

template<typename Value_>
Value_ interpolate(const Value_ left, const Value_ right, const Value_ right_frac) {
    if (std::isinf(left)) {
        // If only left is +/-Inf, the sum will also be the same Inf.
        // If both are Inf of the same sign, the sum will be the same Inf.
        // If they are Inf of different signs, the sum will be a NaN.
        return left + right;
    } else {
        // If only right is +/-Inf, this will also yield the same Inf.
        // This assumes that right_frac > 0 otherwise there wouldn't be any need for interpolation.
        return left + (right - left) * right_frac; 
    }
}

template<typename Output_>
bool check_zeroed(const std::size_t len, const Output_* const ptr) {
    for (std::size_t i = 0; i < len; ++i) {
        if (ptr[i] != 0) {
            return false;
        }
    }
    return true;
}

template<typename Value_>
constexpr Value_ inf_if_available_else_max() {
    if constexpr(std::numeric_limits<Value_>::has_infinity) {
        return std::numeric_limits<Value_>::infinity();
    } else {
        return std::numeric_limits<Value_>::max();
    }
}
/**
 * @endcond
 */

/**
 * @tparam Value_ Some numeric type.
 * @return NaN if supported by `Value_`, otherwise zero. 
 */
template<typename Value_>
constexpr Value_ nan_if_available_else_zero() {
    if constexpr(std::numeric_limits<Value_>::has_quiet_NaN) {
        return std::numeric_limits<Value_>::quiet_NaN();
    } else {
        return 0;
    }
}

}

#endif
