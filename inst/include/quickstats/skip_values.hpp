#ifndef QUICKSTATS_SKIP_VALUES_HPP
#define QUICKSTATS_SKIP_VALUES_HPP

#include <cstddef>

/**
 * @file skip_values.hpp
 * @brief Skip uninteresting values.
 */

namespace quickstats {

/**
 * @tparam Input_ Numeric type of the input.
 * @tparam Skip_ Function defining what elements to skip.
 *
 * @param num_total Total number of elements.
 * @param[in,out] ptr Pointer to an array of length `num_total`, containing the input elements.
 * On output, the first \f$n\f$ entries will contain the unskipped elements, in the same order as provided in the input.
 * @param skip Function that accepts the index of an element in `ptr` (as a `std::size_t`) and the value of the element (as an `Input_`),
 * and returns a boolean indicating whether this element should be skipped.
 *
 * @return The number of unskipped elements \f$n\f$.
 */
template<typename Input_, class Skip_>
std::size_t skip_values(const std::size_t num_total, Input_* const ptr, Skip_ skip) {
    std::size_t used = 0;
    for (std::size_t i = 0; i < num_total; ++i) {
        const auto val = ptr[i];
        if (!skip(i, val)) {
            ptr[used] = val;
            ++used;
        }
    }
    return used;
}

}

#endif
