#ifndef SANISIZER_COMPARISONS_HPP
#define SANISIZER_COMPARISONS_HPP

#include <type_traits>

/**
 * @file comparisons.hpp
 * @brief Signedness-safe integer comparisons.
 */

namespace sanisizer {

/**
 * @tparam Left_ Integer type on the left hand side of the comparison.
 * @tparam Right_ Integer type on the right hand side of the comparison.
 *
 * @param l Non-negative value on the left hand side of the comparison.
 * @param r Non-negative value on the right hand side of the comparison.
 *
 * @return Whether `l` is equal to `r`.
 */
template<typename Left_, typename Right_>
constexpr bool is_equal(Left_ l, Right_ r) {
    return static_cast<typename std::make_unsigned<Left_>::type>(l) == static_cast<typename std::make_unsigned<Right_>::type>(r);
}

/**
 * @tparam Left_ Integer type on the left hand side of the comparison.
 * @tparam Right_ Integer type on the right hand side of the comparison.
 *
 * @param l Non-negative value on the left hand side of the comparison.
 * @param r Non-negative value on the right hand side of the comparison.
 *
 * @return Whether `l` is less than `r`.
 */
template<typename Left_, typename Right_>
constexpr bool is_less_than(Left_ l, Right_ r) {
    return static_cast<typename std::make_unsigned<Left_>::type>(l) < static_cast<typename std::make_unsigned<Right_>::type>(r);
}

/**
 * @tparam Left_ Integer type on the left hand side of the comparison.
 * @tparam Right_ Integer type on the right hand side of the comparison.
 *
 * @param l Non-negative value on the left hand side of the comparison.
 * @param r Non-negative value on the right hand side of the comparison.
 *
 * @return Whether `l` is greater than or equal to `r`.
 */
template<typename Left_, typename Right_>
constexpr bool is_greater_than_or_equal(Left_ l, Right_ r) {
    return !is_less_than(l, r);
}

/**
 * @tparam Left_ Integer type on the left hand side of the comparison.
 * @tparam Right_ Integer type on the right hand side of the comparison.
 *
 * @param l Non-negative value on the left hand side of the comparison.
 * @param r Non-negative value on the right hand side of the comparison.
 *
 * @return Whether `l` is greater than `r`.
 */
template<typename Left_, typename Right_>
constexpr bool is_greater_than(Left_ l, Right_ r) {
    return static_cast<typename std::make_unsigned<Left_>::type>(l) > static_cast<typename std::make_unsigned<Right_>::type>(r);
}

/**
 * @tparam Left_ Integer type on the left hand side of the comparison.
 * @tparam Right_ Integer type on the right hand side of the comparison.
 *
 * @param l Non-negative value on the left hand side of the comparison.
 * @param r Non-negative value on the right hand side of the comparison.
 *
 * @return Whether `l` is less than or equal to `r`.
 */
template<typename Left_, typename Right_>
constexpr bool is_less_than_or_equal(Left_ l, Right_ r) {
    return !is_greater_than(l, r);
}

}

#endif
