#ifndef TATAMI_MULT_UTILS_HPP
#define TATAMI_MULT_UTILS_HPP

#include <limits>
#include <cmath>
#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_>
constexpr bool supports_special_values() {
    return (std::numeric_limits<Value_>::has_infinity || std::numeric_limits<Value_>::has_quiet_NaN || std::numeric_limits<Value_>::has_signaling_NaN);
}

template<typename Value_>
bool is_special(Value_ x) {
    return !std::isfinite(x);
}

template<typename Index_, typename Value_>
void fill_special_index(Index_ len, const Value_* ptr, std::vector<Index_>& specials) {
    for (Index_ i = 0; i < len; ++i) {
        if (is_special(ptr[i])) {
            specials.push_back(i);
        }
    }
}

template<typename Output_, typename DenseValue_, typename Value_, typename Index_>
Output_ dense_sparse_multiply(const DenseValue_* ptr, const tatami::SparseRange<Value_, Index_>& range) {
    Output_ out = 0;
    for (Index_ k = 0; k < range.number; ++k) {
        out += range.value[k] * ptr[range.index[k]];
    }
    return out;
}

template<typename Output_, typename SpecialIndex_, typename SpecialValue_, typename Value_, typename Index_>
Output_ special_dense_sparse_multiply(const std::vector<SpecialIndex_>& specials, const SpecialValue_* ptr, const tatami::SparseRange<Value_, Index_>& range) {
    Output_ out = 0;
    auto sIt = specials.begin(), sEnd = specials.end();
    Index_ k = 0;

    if (k < range.number && sIt != sEnd) {
        Index_ spec = *sIt;
        Index_ ridx = range.index[k];
        while (1) {
            if (ridx < spec) {
                out += ptr[ridx] * range.value[k]; // need to multiply by zero in case the range.value is special.
                if (++k == range.number) {
                    break;
                }
                ridx = range.index[k];
            } else if (ridx > spec) {
                out += ptr[spec] * static_cast<Value_>(0); // it's special, so we can't assume the product would be zero.
                if (++sIt == sEnd) {
                    break;
                }
                spec = *sIt;
            } else {
                out += ptr[spec] * range.value[k];
                ++k;
                ++sIt;
                if (k == range.number || sIt == sEnd) {
                    break;
                }
                ridx = range.index[k];
                spec = *sIt;
            }
        }
    }

    for (; k < range.number; ++k) {
        out += ptr[range.index[k]] * range.value[k];
    }
    for (; sIt != sEnd; ++sIt) {
        out += ptr[*sIt] * static_cast<Value_>(0);
    }

    return out;
}

// Row-major output matrices should have `col_shift = 1`, otherwise it shoud have `row_shift =  1`.
template<typename Output_, class GetOutput_, typename Index_, typename RightIndex_>
void non_contiguous_transfer(const tatami_stats::LocalOutputBuffers<Output_, GetOutput_>& stores, Index_ start, Index_ length, Output_* output, RightIndex_ row_shift, Index_ col_shift) {
    auto rhs_col = stores.size();
    for (decltype(rhs_col) j = 0; j < rhs_col; ++j) {
        auto optr = stores.data(j);
        auto start_offset = sanisizer::product_unsafe<std::size_t>(j, col_shift);
        for (Index_ r = 0; r < length; ++r) {
            // Keeping it simple and just computing the offset within the loop.
            // This is more amenable to vectorization and the compiler can just
            // easily optimize it out if it wants to.
            output[start_offset + sanisizer::product_unsafe<std::size_t>(start + r, row_shift)] = optr[r];
        }
    }
}

}

}

#endif
