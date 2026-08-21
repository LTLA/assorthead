#ifndef TATAMI_MULT_DENSE_MATRIX_UTILS_HPP
#define TATAMI_MULT_DENSE_MATRIX_UTILS_HPP

#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"

namespace tatami_mult {

template<typename Value_, typename Index_>
void populate_dense_buffers(
    const bool row,
    const Index_ primary,
    const Index_ secondary,
    const tatami::Matrix<Value_, Index_>& matrix,
    std::vector<std::vector<Value_> >& all_buffers, // pass by reference so that a move won't invalidate the pointers.
    std::vector<const Value_*>& all_ptrs,
    int num_threads
) {
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        std::vector<Value_> tmp;
        sanisizer::reserve(tmp, secondary);
        auto ext = tatami::consecutive_extractor<false>(matrix, row, start, length);
        for (Index_ i = start, end = start + length; i < end; ++i) {
            tmp.resize(secondary); // reserve is safe, so should resize.
            auto ptr = ext->fetch(tmp.data());
            if (ptr == tmp.data()) {
                all_buffers[i].swap(tmp);
                all_ptrs[i] = all_buffers[i].data(); // only do this after the swap to ensure pointer validity.
            } else {
                all_ptrs[i] = ptr;
            }
        }
    }, primary, num_threads);
}

}

#endif
