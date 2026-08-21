#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_ROW_TO_ROW_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Dense row-major LHS, sparse row-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_sparse_row_matrix_to_row_output()`.
 */
struct MultiplyDenseRowWithSparseRowMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to be loaded at once.
     * See the \f$C\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 1;
};

/**
 * This function will iterate over `left`, realizing rows into memory as needed.
 * It will also realize all of `right` into memory for fast repeated accesses.
 *
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_sparse_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithSparseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(common_dim);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(common_dim);
    populate_sparse_buffers(true, common_dim, right_NC, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    // If there are any empty RHS rows, we only iterate over the non-empty ones in the loop for each LHS row.
    auto right_non_empty = filter_non_empty_sparse(
        right_ranges,
        [&](RightIndex_) -> void {}
    );

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);
    }

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            std::optional<std::vector<Output_> > tmp_row;
            if (do_parallel) {
                tmp_row.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(right_NC));
            }

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = ext->fetch(dbuffer.data());
                const auto optr = output + sanisizer::product_unsafe<std::size_t>(start + lr, right_NC);
                const auto tmp_optr = (do_parallel ? tmp_row->data() : optr);

                auto loop_body = [&](LeftIndex_ cd) -> void {
                    const auto rrange = right_ranges[cd];
                    const Output_ mult = lptr[cd];
                    for (RightIndex_ x = 0; x < rrange.number; ++x) {
                        tmp_optr[rrange.index[x]] += mult * static_cast<Output_>(rrange.value[x]);
                    }
                };

                if (right_non_empty.has_value()) {
                    for (const auto cd : *right_non_empty) {
                        loop_body(cd);
                    }
                } else {
                    for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                        loop_body(cd);
                    }
                }

                if (do_parallel) {
                    std::copy_n(tmp_optr, right_NC, optr);

                    // Technically, we only have to reset the positions at which there is at least one non-zero across all RHS rows.
                    // However, the union of all non-zero positions across all RHS rows is probably quite dense.
                    // It'll likely be faster to just zero the entire buffer rather than trying to zero specific positions;
                    // for example, one 64-byte cache line contains 8 doubles, so you'd need a density below ~10% to even avoid loading every cache line.
                    // And that's not even considering further optimizations in the memset call.
                    std::fill_n(tmp_optr, right_NC, 0);
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            std::vector<std::vector<LeftValue_> > lbuffers;
            lbuffers.reserve(max_block_rows);
            for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
                lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
            }
            auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);

            std::optional<std::vector<Output_> > tmp_rows;
            if (do_parallel) {
                tmp_rows.emplace(sanisizer::product<I<decltype(tmp_rows->size())> >(max_block_rows, right_NC));
            }

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    lptrs[lr_counter] = ext->fetch(lbuffers[lr_counter].data());
                }
                const auto optr = output + sanisizer::product_unsafe<std::size_t>(start + lr, right_NC);
                const auto tmp_optr = (do_parallel ? tmp_rows->data() : optr);

                auto loop_body = [&](LeftIndex_ cd) -> void {
                    const auto rrange = right_ranges[cd];
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const Output_ mult = lptrs[lr_counter][cd];
                        for (RightIndex_ x = 0; x < rrange.number; ++x) {
                            tmp_optr[sanisizer::nd_offset<std::size_t>(rrange.index[x], right_NC, lr_counter)] += mult * static_cast<Output_>(rrange.value[x]);
                        }
                    }
                };

                if (right_non_empty.has_value()) {
                    for (const auto cd : *right_non_empty) {
                        loop_body(cd);
                    }
                } else {
                    for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                        loop_body(cd);
                    }
                }

                if (do_parallel) {
                    const auto output_size = sanisizer::product_unsafe<std::size_t>(lr_num, right_NC);
                    std::copy_n(tmp_optr, output_size, optr);
                    std::fill_n(tmp_optr, output_size, 0);
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
