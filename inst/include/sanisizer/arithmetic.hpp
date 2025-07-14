#ifndef SANISIZER_ARITHMETIC_HPP
#define SANISIZER_ARITHMETIC_HPP

#include <limits>

#include "cast.hpp"
#include "comparisons.hpp"

/**
 * @file arithmetic.hpp
 * @brief Safe arithmetic on integer sizes.
 */

namespace sanisizer {

/**
 * @cond
 */
template<typename Size_, typename Left_, typename Right_>
constexpr bool needs_sum_check() {
    constexpr Size_ smaxed = std::numeric_limits<Size_>::max();
    constexpr Left_ lmaxed = std::numeric_limits<Left_>::max();
    if constexpr(is_greater_than_or_equal(lmaxed, smaxed)) {
        return true;
    } else {
        constexpr Size_ delta = smaxed - static_cast<Size_>(lmaxed);
        constexpr Right_ rmaxed = std::numeric_limits<Right_>::max();
        return is_greater_than(rmaxed, delta);
    }
}
/**
 * @endcond
 */

/**
 * Add two values, checking for overflow in the output type.
 * This is typically used to compute the size of an array that is a concatenation of smaller arrays.
 *
 * @tparam Size_ Integer type representing some concept of size for an array/container.
 * @tparam Left_ Integer type of the left hand side value.
 * @tparam Right_ Integer type of the right hand side value.
 *
 * @param left Non-negative value to add.
 * @param right Non-negative value to add.
 *
 * @return Sum of `left` and `right` as a `Size_`.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<typename Size_, typename Left_, typename Right_>
Size_ sum(Left_ left, Right_ right) {
    Size_ lout = cast<Size_>(left);
    Size_ rout = cast<Size_>(right);

    if constexpr(needs_sum_check<Size_, Left_, Right_>()) {
        constexpr Size_ smaxed = std::numeric_limits<Size_>::max();
        if (smaxed - lout < rout) {
            throw OverflowError("overflow detected in sanisize::sum");
        }
    }

    return lout + rout;
}

/**
 * Unsafe version of `sum()` that casts its arguments to `Size_` but does not check for overflow.
 * This is more efficent if it is known that the sum will not overflow, e.g., from previous calls to `sum()` with larger values.
 *
 * @tparam Size_ Integer type representing some concept of size for an array/container.
 * @tparam Left_ Integer type of the left hand side value.
 * @tparam Right_ Integer type of the right hand side value.
 * 
 * @param left Non-negative value to add.
 * @param right Non-negative value to add.
 *
 * @return Sum of `left` and `right` as a `Size_`.
 */
template<typename Size_, typename Left_, typename Right_>
Size_ sum_unsafe(Left_ left, Right_ right) {
    return static_cast<Size_>(left) + static_cast<Size_>(right);
}

/**
 * @cond
 */
template<typename Size_, typename Left_, typename Right_>
constexpr bool needs_product_check() {
    constexpr Size_ smaxed = std::numeric_limits<Size_>::max();
    constexpr Left_ lmaxed = std::numeric_limits<Left_>::max();
    if constexpr(is_greater_than_or_equal(lmaxed, smaxed)) {
        return true;
    } else {
        constexpr Size_ ratio = smaxed / static_cast<Size_>(lmaxed);
        constexpr Right_ rmaxed = std::numeric_limits<Right_>::max();
        return is_greater_than(rmaxed, ratio);
    }
}
/**
 * @endcond
 */

/**
 * Multiply two values, checking for overflow in the output type.
 * This is typically used to compute the size of a flattened N-dimensional array as the product of its dimension extents.
 * 
 * For consistency, this function will also check that each of `left` and `right` can be cast to `Size_`.
 * This ensures that per-dimension indices/extents can be safely represented as `Size_` in later steps (e.g., `nd_offset()`). 
 * These checls are necessary as the product may fit in `Size_` but not each of `left` and `right` if either is equal to zero.
 *
 * @tparam Size_ Integer type representing some concept of size for an array/container.
 * @tparam Left_ Integer type of the left hand side value.
 * @tparam Right_ Integer type of the right hand side value.
 *
 * @param left Non-negative value to multiply.
 * @param right Non-negative value to multiply.
 *
 * @return Product of `left` and `right` as a `Size_`.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<typename Size_, typename Left_, typename Right_>
Size_ product(Left_ left, Right_ right) {
    Size_ lout = cast<Size_>(left);
    Size_ rout = cast<Size_>(right);

    if constexpr(needs_product_check<Size_, Left_, Right_>()) {
        constexpr Size_ smaxed = std::numeric_limits<Size_>::max();
        if (lout && smaxed / lout < rout) {
            throw OverflowError("overflow detected in sanisize::product");
        }
    }

    return lout * rout;
}

/**
 * Unsafe version of `product()` that casts its arguments to `Size_` but does not check for overflow.
 * This is more efficent if it is known that the product will not overflow, e.g., from previous calls to `product()` with larger values.
 *
 * @tparam Size_ Integer type representing some concept of size for an array/container.
 * @tparam Left_ Integer type of the left hand side value.
 * @tparam Right_ Integer type of the right hand side value.
 * 
 * @param left Non-negative value to multiply.
 * @param right Non-negative value to multiply.
 *
 * @return Product of `left` and `right` as a `Size_`.
 */
template<typename Size_, typename Left_, typename Right_>
Size_ product_unsafe(Left_ left, Right_ right) {
    return static_cast<Size_>(left) * static_cast<Size_>(right);
}

}

#endif
