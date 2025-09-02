#ifndef SANISIZER_CAST_HPP
#define SANISIZER_CAST_HPP

#include <limits>

#include "comparisons.hpp"
#include "utils.hpp"

/**
 * @file cast.hpp
 * @brief Safe casts of integer size.
 */

namespace sanisizer {

/**
 * Cast `x` to the type of the size of a C-style array or STL container.
 * This avoids accidental overflow from an implicit cast when `x` is used in `new` or the container's constructor.
 * 
 * @tparam Size_ Integer type to cast to, typically representing some concept of size for an array/container.
 * @tparam Input_ Integer type of the input size.
 *
 * @param x Non-negative value to be casted, typically representing the size of an array/container.
 *
 * @return `x` as a `Size_`.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<typename Size_, typename Input_>
Size_ cast(Input_ x) {
    constexpr Size_ maxed = std::numeric_limits<Size_>::max();
    if constexpr(is_greater_than(std::numeric_limits<Input_>::max(), maxed)) {
        if (is_greater_than(x, maxed)) {
            throw OverflowError("overflow detected in sanisize::cast");
        }
    }
    return x;
}

/**
 * Check that `x` can be cast to the type of the size of a C-style array or STL container. 
 * This is useful for chaining together checks without actually doing the cast itself.
 * 
 * @tparam Size_ Integer type to cast to, typically representing some concept of size for an array/container.
 * @tparam Input_ Integer type of the input size.
 *
 * @param x Non-negative value to be casted, typically representing the size of an array/container.
 *
 * @return `x` as its input type.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<typename Size_, typename Input_>
Input_ can_cast(Input_ x) {
    constexpr Size_ maxed = std::numeric_limits<Size_>::max();
    if constexpr(is_greater_than(std::numeric_limits<Input_>::max(), maxed)) {
        if (is_greater_than(x, maxed)) {
            throw OverflowError("overflow detected in sanisize::cast");
        }
    }
    return x;
}

}

#endif
