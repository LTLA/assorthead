#ifndef TATAMI_MULT_DENSE_ROW_HPP
#define TATAMI_MULT_DENSE_ROW_HPP

#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_row_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());
            output[r] = std::inner_product(ptr, ptr + NC, rhs, static_cast<Output_>(0));
        }

    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_row_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs, const std::vector<Output_*>& output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    auto num_rhs = rhs.size();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());
            for (decltype(num_rhs) j = 0; j < num_rhs; ++j) {
                output[j][r] = std::inner_product(ptr, ptr + NC, rhs[j], static_cast<Output_>(0));
            }
        }

    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_row_tatami_dense(
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
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC);

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());
            auto rext = tatami::consecutive_extractor<false>(&rhs, false, static_cast<RightIndex_>(0), rhs_col);
            auto start_offset = sanisizer::product_unsafe<std::size_t>(r, row_shift); // offsets must fit in size_t if output is correctly sized.

            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto rptr = rext->fetch(rbuffer.data());
                output[start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift)] = std::inner_product(ptr, ptr + NC, rptr, static_cast<Output_>(0));
            }
        }

    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_row_tatami_sparse(
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
        auto ext = tatami::consecutive_extractor<false>(&matrix, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(NC);

        constexpr bool supports_specials = supports_special_values<Value_>();
        typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());
            auto rext = tatami::consecutive_extractor<true>(&rhs, false, static_cast<RightIndex_>(0), rhs_col);
            auto start_offset = sanisizer::product_unsafe<std::size_t>(r, row_shift); // offsets must fit in size_t if output is correctly sized.

            if constexpr(supports_specials) {
                specials.clear();
                fill_special_index(NC, ptr, specials);
            }

            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto range = rext->fetch(vbuffer.data(), ibuffer.data());
                auto cur_offset = start_offset + sanisizer::product_unsafe<std::size_t>(j, col_shift);

                if constexpr(supports_specials) {
                    if (specials.size()) {
                        output[cur_offset] = special_dense_sparse_multiply<Output_>(specials, ptr, range);
                        continue;
                    }
                }

                output[cur_offset] = dense_sparse_multiply<Output_>(ptr, range);
            }
        }

    }, NR, num_threads);
}

}

}

#endif
