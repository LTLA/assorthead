#ifndef SANISIZER_CAP_HPP
#define SANISIZER_CAP_HPP

#include <limits>

#include "comparisons.hpp"

/**
 * @file cap.hpp
 * @brief Cap a value at the largest size.
 */

namespace sanisizer {

/**
 * Cap an integer at the largest value of a size type.
 * This is primarily intended for setting appropriate default values of function arguments and class variables.
 *
 * @tparam Size_ Integer type to return, typically representing some concept of size for an array/container.
 * @tparam Input_ Integer type of the input size.
 *
 * @param x Non-negative value to be capped.
 *
 * @return `x` if it can be represented in `Size_`, otherwise the maximum value of `Size_`.
 */
template<typename Size_, typename Input_>
constexpr Size_ cap(Input_ x) {
    constexpr Size_ maxed = std::numeric_limits<Size_>::max();
    if (is_greater_than(x, maxed)) {
        return maxed;
    } else {
        return x;
    }
}

}

#endif
