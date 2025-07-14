#ifndef TATAMI_MULT_SPARSE_ROW_HPP
#define TATAMI_MULT_SPARSE_ROW_HPP

#include "utils.hpp"

#include <vector>
#include <cstddef>
#include <type_traits>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_, typename Index_>
void expand_sparse_range(const tatami::SparseRange<Value_, Index_>& range, std::vector<Value_>& expanded) {
    for (Index_ k = 0; k < range.number; ++k) {
        expanded[range.index[k]] = range.value[k];
    }
}

template<typename Value_, typename Index_>
void reset_expanded_sparse_range(const tatami::SparseRange<Value_, Index_>& range, std::vector<Value_>& expanded) {
    for (Index_ k = 0; k < range.number; ++k) {
        expanded[range.index[k]] = 0;
    }
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    // Check if the RHS has any special values.
    constexpr bool supports_specials = supports_special_values<Right_>();
    typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;
    if constexpr(supports_specials) {
        fill_special_index(NC, rhs, specials);
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            if constexpr(supports_specials) {
                if (specials.size()) {
                    output[r] = special_dense_sparse_multiply<Output_>(specials, rhs, range);
                    continue;
                }
            }

            output[r] = dense_sparse_multiply<Output_>(rhs, range);
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void sparse_row_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs, const std::vector<Output_*>& output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    auto num_rhs = rhs.size();

    // Check if the RHS has any special values.
    constexpr bool supports_specials = supports_special_values<Right_>();
    typename std::conditional<supports_specials, std::vector<std::vector<Index_> >, bool>::type specials;
    if constexpr(supports_specials) {
        specials.resize(sanisizer::cast<decltype(specials.size())>(num_rhs));
        for (decltype(num_rhs) j = 0; j < num_rhs; ++j) {
            fill_special_index(NC, rhs[j], specials[j]);
        }
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            for (decltype(num_rhs) j = 0; j < num_rhs; ++j) {
                auto& out = output[j][r];
                if constexpr(supports_specials) {
                    if (specials[j].size()) {
                        out = special_dense_sparse_multiply<Output_>(specials[j], rhs[j], range);
                        continue;
                    }
                }
                out = dense_sparse_multiply<Output_>(rhs[j], range);
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_dense(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads)
{
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    // Do one pass through the other matrix to see which columns have special values.
    // We can't afford to hold the indices here as rhs_col may be arbitrarily large
    // and the matrix might be full of special values.
    constexpr bool supports_specials = supports_special_values<RightValue_>();
    typename std::conditional<supports_specials, std::vector<unsigned char>, bool>::type has_special;
    bool any_special = false;
    if constexpr(supports_specials) {
        tatami::resize_container_to_Index_size(has_special, rhs_col);

        tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, start, length);
            auto buffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC); // remember, NC == right.nrow() here.
            for (RightIndex_ j = start, end = start + length; j < end; ++j) {
                auto rptr = rext->fetch(buffer.data());
                for (RightIndex_ r = 0; r < NC; ++r) {
                    if (is_special(rptr[r])) {
                        has_special[j] = true;
                        break;
                    }
                }
            }
        }, rhs_col, num_threads);

        for (auto is : has_special) {
            if (is) {
                any_special = true;
                break;
            }
        }
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC);

        // The idea is that if we have special values in the RHS, we expand the sparse
        // values into a dense array for a regular inner product. This avoids having to
        // keep track of the individual indices of the special values.
        typename std::conditional<supports_specials, std::vector<Value_>, bool>::type expanded;
        if constexpr(supports_specials) {
            if (any_special) {
                tatami::resize_container_to_Index_size(expanded, NC);
            }
        }

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, static_cast<RightIndex_>(0), rhs_col);
            auto start_offset = sanisizer::product_unsafe<std::size_t>(r, row_shift); // offsets must fit in size_t if output is correctly sized.

            if constexpr(supports_specials) {
                if (any_special) {
                    // Expanding the range for easier full multiplication with a dense vector.
                    expand_sparse_range(range, expanded);

                    for (RightIndex_ j = 0; j < rhs_col; ++j) {
                        auto rptr = rext->fetch(rbuffer.data());
                        auto cur_offset = start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift);
                        if (has_special[j]) {
                            output[cur_offset] = std::inner_product(expanded.begin(), expanded.end(), rptr, static_cast<Output_>(0));
                        } else {
                            output[cur_offset] = dense_sparse_multiply<Output_>(rptr, range);
                        }
                    }

                    reset_expanded_sparse_range(range, expanded);
                    continue;
                }
            }

            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto rptr = rext->fetch(rbuffer.data());
                auto cur_offset = start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift);
                output[cur_offset] = dense_sparse_multiply<Output_>(rptr, range);
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void sparse_row_tatami_sparse(
    const tatami::Matrix<Value_, Index_>& matrix,
    const tatami::Matrix<RightValue_, RightIndex_>& rhs,
    Output_* output,
    RightIndex_ row_shift,
    Index_ col_shift,
    int num_threads)
{
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    RightIndex_ rhs_col = rhs.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(&matrix, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        auto rvbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC);
        auto ribuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(NC);
        auto expanded = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        constexpr bool supports_specials = supports_special_values<Value_>();
        typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;
        if constexpr(supports_specials) {
            tatami::resize_container_to_Index_size(specials, NC);
        }

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            auto rext = tatami::consecutive_extractor<true>(&rhs, false, static_cast<RightIndex_>(0), rhs_col);

            // Expanding the sparse vector into a dense format for easier mapping by the RHS's sparse vector.
            expand_sparse_range(range, expanded);

            if constexpr(supports_specials) {
                specials.clear();
                for (Index_ i = 0; i < range.number; ++i) {
                    if (is_special(range.value[i])) {
                        specials.push_back(range.index[i]);
                    }
                }
            }

            auto start_offset = sanisizer::product_unsafe<std::size_t>(r, row_shift); // offsets must fit in size_t if output is correctly sized.
            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto rrange = rext->fetch(rvbuffer.data(), ribuffer.data());
                auto cur_offset = start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift);

                if constexpr(supports_specials) {
                    if (specials.size()) {
                        output[cur_offset] = special_dense_sparse_multiply<Output_>(specials, expanded.data(), rrange);
                        continue;
                    }
                }

                output[cur_offset] = dense_sparse_multiply<Output_>(expanded.data(), rrange);
            }

            reset_expanded_sparse_range(range, expanded);
        }
    }, NR, num_threads);
}

}

}

#endif
