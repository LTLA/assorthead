#ifndef TATAMI_MULT_DENSE_DOT_PRODUCT_H
#define TATAMI_MULT_DENSE_DOT_PRODUCT_H

#include <cstddef>
#include <array>
#include <numeric>

#include "utils.hpp"

namespace tatami_mult {

// Check out https://github.com/tatami-inc/test-multiplication/blob/master/other/accumulators for implementation choices.
//
// In particular, we trust compiler to the appropriate amount of unrolling.
// Note that this doesn't happen at -O2 with older GCCs, but I still don't want to manually unroll;
// I want the compiler to be able to decide to do partial unrolling if there are too many accumulators.
//
// We don't do any peeling or vectorization of the epilogue loop
// as these don't seem to provide any benefit and are sometimes worse at higher numbers of accumulators.

template<std::size_t accumulators_, typename Iterator1_, typename Iterator2_, typename Output_>
Output_ dense_dot_product(const std::size_t len, Iterator1_ start1, Iterator2_ start2, Output_ initial) {
    if constexpr(accumulators_ == 1) {
        Output_ dot = initial;
        for (std::size_t i = 0; i < len; ++i) {
            dot += static_cast<Output_>(*(start1 + i)) * static_cast<Output_>(*(start2 + i));
        }
        return dot;

    } else {
        const std::size_t cycles = len / accumulators_;
        const std::size_t remainder = len % accumulators_;
        std::array<Output_, accumulators_> dots{};

        for (std::size_t c = 0; c < cycles; ++c) {
            for (std::size_t a = 0; a < accumulators_; ++a) {
                const std::size_t idx = c * accumulators_ + a;;
                dots[a] += static_cast<Output_>(*(start1 + idx)) * static_cast<Output_>(*(start2 + idx));
            }
        }

        for (std::size_t i = 0; i < remainder; ++i) {
            const auto idx = cycles * accumulators_ + i;
            initial += static_cast<Output_>(*(start1 + idx)) * static_cast<Output_>(*(start2 + idx));
        }

        return initial + recursive_sum(dots);
    }
}

}

#endif
