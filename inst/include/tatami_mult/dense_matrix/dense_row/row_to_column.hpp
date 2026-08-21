#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP

#include <vector>
#include <cstddef>
#include <optional>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_column.hpp
 * @brief Dense row-major LHS, dense row-major matrix RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_dense_row_matrix_to_column_output()`.
 */
struct MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Primary block size, i.e., the number of LHS rows to be loaded at once.
     * This is also used to define the number of LHS columns in each block.
     * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size, i.e., the number of RHS columns to be processed in each block.
     * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     * Different secondary block sizes will not change the results.
     */
    int secondary_block_size = 64;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightRow_ Functor that accepts a `LeftIndex_` and returns a pointer to an RHS row.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_row Function that accepts a `LeftIndex_` in `[0, left.ncol())` and returns a pointer to an array of length `right_columns`.
 * The array referenced by `get_right_row(i)` represents the `i`-th row of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this stores the matrix product in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightRow_, typename Output_>
void multiply_dense_row_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightRow_ get_right_row,
    Output_* const output,
    const MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto buffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            // Use a temporary buffer to mimic an output row.
            // This gives us contiguous writes in the innermost loop while mitigating false sharing.
            auto tmp_output = tatami::create_container_of_Index_size<std::vector<Output_> >(right_columns);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto left_ptr = ext->fetch(buffer.data());
                std::fill(tmp_output.begin(), tmp_output.end(), 0);
                for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                    const Output_ mult = left_ptr[cd];
                    const auto rightrow = get_right_row(cd);
                    for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                        tmp_output[rc] += static_cast<Output_>(rightrow[rc]) * mult;
                    }
                }
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = tmp_output[rc];
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto left_ext = tatami::consecutive_extractor<false>(left, true, start, length);
            std::vector<std::vector<LeftValue_> > left_buffers;
            std::vector<const LeftValue_*> left_ptrs;

            std::vector<Output_> tmp_output;
            {
                const LeftIndex_ max_block_rows = sanisizer::min(length, options.primary_block_size);
                left_buffers.reserve(max_block_rows);
                for (LeftIndex_  b = 0; b < max_block_rows; ++b) {
                    left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                }
                sanisizer::resize(left_ptrs, max_block_rows);

                // Creating a block to hold the output during the updates over all 'c'.
                // This enables contiguous writes in the innermost loop while also avoiding false sharing.
                // Hopefully std::vector can store this - I'd be very surprised if size_type != size_t, but we check anyway.
                sanisizer::resize(tmp_output, sanisizer::product_unsafe<std::size_t>(max_block_rows, right_columns));
            }

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lr_num = sanisizer::min(options.primary_block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    left_ptrs[lr_counter] = left_ext->fetch(left_buffers[lr_counter].data());
                }

                const auto out_space = sanisizer::product_unsafe<std::size_t>(lr_num, right_columns);
                std::fill_n(tmp_output.data(), out_space, 0);

                LeftIndex_ cd = 0;
                while (cd < common_dim) { 
                    const LeftIndex_ cd_end = cd + sanisizer::min(options.primary_block_size, common_dim - cd);
                    RightColumns_ rc = 0;
                    while (rc < right_columns) {
                        const RightColumns_ rc_end = rc + sanisizer::min(options.secondary_block_size, right_columns - rc);

                        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            const auto matrow = left_ptrs[lr_counter];
                            const auto prod = tmp_output.data() + sanisizer::product_unsafe<std::size_t>(lr_counter, right_columns);
                            for (auto cd_copy = cd; cd_copy < cd_end; ++cd_copy) {
                                const auto mult = matrow[cd_copy];
                                const auto rightrow = get_right_row(cd_copy);
                                for (auto rc_copy = rc; rc_copy < rc_end; ++rc_copy) {
                                    prod[rc_copy] += mult * rightrow[rc_copy];
                                }
                            }
                        }

                        rc = rc_end;
                    }
                    cd = cd_end;
                }

                // Transposition using square blocks of the smaller (primary) block size.
                RightColumns_ rct = 0;
                while (rct < right_columns) {
                    const RightColumns_ rct_end = rct + sanisizer::min(options.primary_block_size, right_columns - rct);
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        for (auto rct_copy = rct; rct_copy < rct_end; ++rct_copy) {
                            const auto val = tmp_output[sanisizer::nd_offset<std::size_t>(rct_copy, right_columns, lr_counter)];
                            output[sanisizer::nd_offset<std::size_t>(start + lr + lr_counter, left_NR, rct_copy)] = val;
                        }
                    }
                    rct = rct_end;
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

/**
 * Overload of `multiply_dense_row_with_dense_row_matrix_to_column_output()` for a RHS `tatami::Matrix`.
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
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto common_dim = left.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(common_dim);
    const auto right_NC = right.ncol();
    populate_dense_buffers(true, common_dim, right_NC, right, right_buffers, right_ptrs, options.num_threads);

    multiply_dense_row_with_dense_row_matrix_to_column_output(
        left,
        right_NC,
        [&](const LeftIndex_ cd) -> const RightValue_* {
            return right_ptrs[cd];
        },
        output,
        options
    ); 
}

}

#endif
