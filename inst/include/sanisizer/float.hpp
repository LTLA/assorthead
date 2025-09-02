#ifndef SANISIZER_FLOAT_HPP
#define SANISIZER_FLOAT_HPP

#include <limits>
#include <cmath>
#include <stdexcept>

/**
 * @file float.hpp
 * @brief Safely convert floats to integer sizes.
 */

namespace sanisizer {

/**
 * @cond
 */
template<typename Integer_, typename Float_>
bool float_to_int_overflows(Float_ floored_x) {
    constexpr auto output_precision = std::numeric_limits<Integer_>::digits;
#ifndef SANISIZER_FLOAT_FORCE_FREXP
    if constexpr(std::numeric_limits<Float_>::radix == 2) {
        // ilogb returns an 'exp' such that 2^exp <= floored_x < 2^(exp + 1).
        return floored_x != 0 && std::ilogb(floored_x) >= output_precision;
    } else {
        // Ensure we're covered for weird float types where the radix is not 2.
        // This is pretty unusual so we need to use a macro to force test coverage.
#endif
        int exp;
        std::frexp(floored_x, &exp); 
        // frexp guarantees that 2^(exp - 1) <= floored_x < 2^exp.
        return exp > output_precision;
#ifndef SANISIZER_FLOAT_FORCE_FREXP
    }
#endif
}
/**
 * @endcond
 */

/**
 * Safely convert a non-negative floating-point number to an integer with truncation.
 * This is occasionally necessary when the size of a container or number of loop iterations is determined by floating-point calculations.
 *
 * We do not have a `to_float()` function as we assume that the implementation's floats are compliant with the IEEE-754 specifiation,
 * such that very large integers will be safely converted to positive infinity.
 *
 * @tparam Integer_ Integer type.
 * @tparam Float_ Floating-point type.
 *
 * @param x Non-negative floating-point number, usually holding some kind of size. 
 *
 * @return The value of `x` as an integer, after truncation.
 * An exception is raised if the input is non-finite or overflow would occur.
 */
template<typename Integer_, typename Float_>
Integer_ from_float(Float_ x) {
    if (!std::isfinite(x)) {
        throw std::runtime_error("invalid conversion in sanisizer::from_float");
    }
    x = std::trunc(x);
    if (float_to_int_overflows<Integer_>(x)) {
        throw std::runtime_error("invalid conversion in sanisizer::from_float");
    }
    return x;
}

}

#endif
