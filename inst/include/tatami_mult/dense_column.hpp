#ifndef TATAMI_MULT_DENSE_COLUMN_HPP
#define TATAMI_MULT_DENSE_COLUMN_HPP

#include "utils.hpp"

#include <vector>
#include <cstddef>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_mult {

namespace internal {

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_column_vector(const tatami::Matrix<Value_, Index_>& matrix, const Right_* rhs, Output_* output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();

    tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, static_cast<Index_>(0), NC, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        tatami_stats::LocalOutputBuffer<Value_> store(t, start, length, output);
        auto optr = store.data();

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());
            Output_ mult = rhs[c];
            for (Index_ r = 0; r < length; ++r) {
                optr[r] += mult * ptr[r];
            }
        }

        store.transfer();
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Right_, typename Output_>
void dense_column_vectors(const tatami::Matrix<Value_, Index_>& matrix, const std::vector<Right_*>& rhs, const std::vector<Output_*>& output, int num_threads) {
    Index_ NR = matrix.nrow();
    Index_ NC = matrix.ncol();
    auto num_rhs = rhs.size();

    tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, static_cast<Index_>(0), NC, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);

        auto getter = [&](Index_ i) -> Output_* { return output[i]; };
        tatami_stats::LocalOutputBuffers<Output_, decltype(getter)> stores(
            t,
            sanisizer::cast<std::size_t>(output.size()),
            start,
            length,
            std::move(getter)
        );

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());

            for (decltype(num_rhs) j = 0; j < num_rhs; ++j) {
                auto optr = stores.data(j);
                Output_ mult = rhs[j][c];
                for (Index_ r = 0; r < length; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }
   
        stores.transfer();
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_column_tatami_dense(
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

    tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, static_cast<Index_>(0), NC, start, length);
        auto rext = tatami::consecutive_extractor<false>(&rhs, true, static_cast<RightIndex_>(0), static_cast<RightIndex_>(NC)); // remember, NC == rhs.nrow().
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(rhs_col);

        bool contiguous_output = (row_shift == 1);
        auto getter = [&](RightIndex_ j) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(j, col_shift); }; // product must fit if output is correctly sized.
        tatami_stats::LocalOutputBuffers<Output_, decltype(getter)> stores(
            (contiguous_output ? t : num_threads), // avoid a direct write at t = 0 if it's not contiguous.
            rhs_col, // cast to size_t is safe as the tatami contract guarantees that RightIndex_ fits in a size_t.
            start,
            length,
            std::move(getter)
        );

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());
            auto rptr = rext->fetch(rbuffer.data());

            for (RightIndex_ j = 0; j < rhs_col; ++j) {
                auto optr = stores.data(j);
                Output_ mult = rptr[j];
                for (Index_ r = 0; r < length; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }

        if (contiguous_output) {
            stores.transfer();
        } else {
            non_contiguous_transfer(stores, start, length, output, row_shift, col_shift);
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename RightValue_, typename RightIndex_, typename Output_>
void dense_column_tatami_sparse(
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

    tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(&matrix, false, static_cast<Index_>(0), NC, start, length);
        auto rext = tatami::consecutive_extractor<true>(&rhs, true, static_cast<RightIndex_>(0), static_cast<RightIndex_>(NC)); // remember, NC == rhs.nrow().
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(rhs_col);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(rhs_col);

        bool contiguous_output = (row_shift == 1);
        auto getter = [&](RightIndex_ j) -> Output_* { return output + sanisizer::product_unsafe<std::size_t>(j, col_shift); }; // product must fit if output is correctly sized.
        tatami_stats::LocalOutputBuffers<Output_, decltype(getter)> stores(
            (contiguous_output ? t : num_threads), // avoid a direct write at t = 0 if it's not contiguous.
            rhs_col, // cast to size_t is safe as the tatami contract guarantees that RightIndex_ fits in a size_t.
            start,
            length,
            std::move(getter)
        );

        constexpr bool supports_specials = supports_special_values<Value_>();
        typename std::conditional<supports_specials, std::vector<Index_>, bool>::type specials;

        for (Index_ c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(buffer.data());
            auto range = rext->fetch(vbuffer.data(), ibuffer.data());

            if constexpr(supports_specials) { // need separate multiplication to preserve the specials.
                specials.clear();
                fill_special_index(length, ptr, specials);

                if (specials.size()) {
                    RightIndex_ k = 0; 
                    for (RightIndex_ j = 0; j < rhs_col; ++j) {
                        auto optr = stores.data(j);
                        if (k < range.number && j == range.index[k]) {
                            Output_ mult = range.value[k];
                            for (Index_ r = 0; r < length; ++r) {
                                optr[r] += mult * ptr[r];
                            }
                            ++k;
                        } else {
                            for (auto s : specials) {
                                optr[s] += ptr[s] * static_cast<Output_>(0);
                            }
                        }
                    }
                    continue;
                }
            }

            for (RightIndex_ k = 0; k < range.number; ++k) {
                auto optr = stores.data(range.index[k]);
                Output_ mult = range.value[k];
                for (Index_ r = 0; r < length; ++r) {
                    optr[r] += mult * ptr[r];
                }
            }
        }

        if (contiguous_output) {
            stores.transfer();
        } else {
            non_contiguous_transfer(stores, start, length, output, row_shift, col_shift);
        }
    }, NR, num_threads);
}

}

}

#endif
