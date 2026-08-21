#ifndef TATAMI_MULT_SPARSE_DOT_PRODUCT_H
#define TATAMI_MULT_SPARSE_DOT_PRODUCT_H

#include <cstddef>
#include <array>
#include <numeric>

#include "utils.hpp"

namespace tatami_mult {

// See comments in dense_dot_product.hpp for implementation decisions.

template<std::size_t accumulators_, class ValueIterator_, class IndexIterator_, typename Dense_, typename Output_>
Output_ sparse_dot_product(const std::size_t num_non_zeros, ValueIterator_ vptr, IndexIterator_ iptr, Dense_ dense, Output_ initial) {
    if constexpr(accumulators_ == 1) {
        Output_ dot = initial;
        for (std::size_t i = 0; i < num_non_zeros; ++i) {
            dot += static_cast<Output_>(dense[*(iptr + i)]) * static_cast<Output_>(*(vptr + i));
        }
        return dot;

    } else {
        std::array<Output_, accumulators_> dots{};
        const std::size_t cycles = num_non_zeros / accumulators_;
        const std::size_t remainder = num_non_zeros % accumulators_;

        for (std::size_t c = 0; c < cycles; ++c) {
            for (std::size_t a = 0; a < accumulators_; ++a) {
                const std::size_t idx = c * accumulators_ + a;
                dots[a] += static_cast<Output_>(dense[*(iptr + idx)]) * static_cast<Output_>(*(vptr + idx));
            }
        }

        for (std::size_t i = 0; i < remainder; ++i) {
            const auto idx = cycles * accumulators_ + i;
            initial += dense[*(iptr + idx)] * *(vptr + idx);
        }

        return initial + recursive_sum(dots);
    }
}

}

#endif
