#ifndef SANISIZER_COMPARISONS_HPP
#define SANISIZER_COMPARISONS_HPP

#include <type_traits>
#include <limits>

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

/**
 * @tparam First_ First integer type. 
 * @tparam Second_ Second integer type.
 *
 * @param first First non-negative value. 
 * @param second Second non-negative value. 
 *
 * @return The smaller of `first` and `second`, in the smaller integer type of `First_` and `Second_`.
 */
template<typename First_, typename Second_>
constexpr auto min(First_ first, Second_ second) {
    if constexpr(std::numeric_limits<First_>::max() > std::numeric_limits<Second_>::max()) {
        if (is_greater_than(first, second)) {
            return second;
        } else {
            return static_cast<Second_>(first);
        }
    } else {
        if (is_greater_than(first, second)) {
            return static_cast<First_>(second);
        } else {
            return first;
        }
    }
}

/**
 * @tparam First_ First integer type. 
 * @tparam Second_ Second integer type.
 *
 * @param first First non-negative value. 
 * @param second Second non-negative value. 
 *
 * @return The larger of `first` and `second`, in the larger integer type of `First_` and `Second_`.
 */
template<typename First_, typename Second_>
constexpr auto max(First_ first, Second_ second) {
    if constexpr(std::numeric_limits<First_>::max() > std::numeric_limits<Second_>::max()) {
        if (is_greater_than(first, second)) {
            return first;
        } else {
            return static_cast<First_>(second);
        }
    } else {
        if (is_greater_than(first, second)) {
            return static_cast<Second_>(first);
        } else {
            return second;
        }
    }
}

}

#endif
