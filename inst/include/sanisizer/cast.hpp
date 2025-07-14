#ifndef SANISIZER_CAST_HPP
#define SANISIZER_CAST_HPP

#include "comparisons.hpp"

#include <stdexcept>
#include <utility>

/**
 * @file cast.hpp
 * @brief Safe casts of integer size.
 */

namespace sanisizer {

/**
 * @brief Error from an integer overflow.
 *
 * This allow callers of various **sanisizer** functions to catch overflows and handle them accordingly.
 */
class OverflowError final : public std::runtime_error {
public:
    /**
     * @cond
     */
    template<typename ... Args_>
    OverflowError(Args_&&... args) : std::runtime_error(std::forward<Args_>(args)...) {}
    /**
     * @endcond
     */
};

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

/**
 * Create a new container of a specified size.
 * This protects against overflow when casting the integer size to the container's size type.
 *
 * @tparam Container_ Container class that accepts the size in the first argument of its constructor.
 * @tparam Input_ Integer type of the input size.
 * @tparam Args_ Further arguments to pass to the container's constructor.
 *
 * @param x Non-negative value representing the desired container size.
 * @param args Additional arguments to pass to the `Container_` constructor after the size.
 *
 * @return An instance of the container, constructed to be of size `x`.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<class Container_, typename Input_, typename ... Args_>
Container_ create(Input_ x, Args_&&... args) {
    return Container_(cast<decltype(std::declval<Container_>().size())>(x), std::forward<Args_>(args)...);
}

}

#endif
