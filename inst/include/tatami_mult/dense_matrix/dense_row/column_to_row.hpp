#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP

#include <vector>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../dense_dot_product.hpp"
#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file column_to_row.hpp
 * @brief Dense row-major LHS, dense column-major matrix RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_dense_column_matrix_to_row_output()`.
 */
struct MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Primary block size, i.e., the number of LHS rows to be loaded at once.
     * This is also used to define the number of RHS columns in each block.
     * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size, i.e., the number of LHS columns to be processed in each block.
     * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     * Different secondary block sizes may slightly change the results due to differences in floating-point round-off error.
     */
    int secondary_block_size = 64;
};

/**
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightColumn_ Functor that accepts a `RightColumns_` and returns a pointer to an RHS column.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_column Function that accepts a `RightColumns_` in `[0, right_columns)` and returns a pointer to an array of length `left.ncol()`.
 * The array referenced by `get_right_column(i)` represents the `i`-th column of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this stores the matrix product in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightColumns_, typename GetRightColumn_, typename Output_>
void multiply_dense_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightColumn_ get_right_column,
    Output_* const output,
    const MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto lext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto lbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = lext->fetch(lbuffer.data());
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    // cast of common_dim to size_t is safe due to tatami's contract.
                    const auto res = dense_dot_product<accumulators_>(common_dim, lptr, get_right_column(rc), static_cast<Output_>(0));
                    output[sanisizer::nd_offset<std::size_t>(rc, right_columns, start + lr)] = res;
                }
            }
        }, left_NR, options.num_threads);
        return;
    } 

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        // Product must fit in a size_t in order for output to have been allocated correctly in the first place.
        // Technically, right_columns could be larger than a size_t if left_NR == 0, but the product after wraparound would still be zero, so it's fine.
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_columns), 0);
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

        const LeftIndex_ max_block_rows = sanisizer::min(length, options.primary_block_size);
        std::vector<std::vector<LeftValue_> > lbuffers;
        lbuffers.reserve(max_block_rows);
        for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
            lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
        }
        auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* optr;
        if (do_parallel) {
            // For the multi-threaded case, we create some temporary buffers to hold the partial dot products for the current set of submatrices.
            // This aims to mitigate false sharing as we update each block's partial dot products in the loop over the common dimension.
            // There is still some potential for false sharing when we transfer the results to the output buffers,
            // but this is the same as the unblocked case so we won't worry about it.
            const auto max_block_cols = sanisizer::min(right_columns, options.primary_block_size);
            const auto max_block_rows = sanisizer::min(length, options.primary_block_size);
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(max_block_cols, max_block_rows));
            optr = tmp_output->data();
        } else {
            optr = output; // no need to add 'start' as this is zero in the single-threaded case.
        }

        LeftIndex_ lr = 0;
        while (lr < length) {
            const LeftIndex_ lr_num = sanisizer::min(options.primary_block_size, length - lr);
            for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                lptrs[lr_counter] = ext->fetch(lbuffers[lr_counter].data());
            }

            RightColumns_ rc = 0;
            while (rc < right_columns) {
                const RightColumns_ rc_num = sanisizer::min(options.primary_block_size, right_columns - rc);

                LeftIndex_ out_row_offset;
                RightColumns_ out_col_offset, out_stride;
                if (do_parallel) {
                    std::fill_n(optr, sanisizer::product_unsafe<std::size_t>(rc_num, lr_num), 0);
                    out_row_offset = 0;
                    out_col_offset = 0;
                    out_stride = rc_num;
                } else {
                    out_row_offset = lr; // start == 0 in serial code, so no need to add start here.
                    out_col_offset = rc;
                    out_stride = right_columns;
                }

                LeftIndex_ cd = 0;
                while (cd < common_dim) {
                    const LeftIndex_ cd_num = sanisizer::min(options.secondary_block_size, common_dim - cd);
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto leftrow = lptrs[lr_counter];
                        for (RightColumns_ rc_counter = 0; rc_counter < rc_num; ++rc_counter) {
                            auto& dest = optr[sanisizer::nd_offset<std::size_t>(out_col_offset + rc_counter, out_stride, out_row_offset + lr_counter)]; 
                            dest = dense_dot_product<accumulators_>(
                                cd_num, // cast to size_t is safe due to tatami's contract.
                                get_right_column(rc + rc_counter) + cd,
                                leftrow + cd,
                                dest
                            );
                        }
                    }
                    cd += cd_num;
                }

                if (do_parallel) {
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        std::copy_n(
                            optr + sanisizer::product_unsafe<std::size_t>(rc_num, lr_counter),
                            rc_num,
                            output + sanisizer::nd_offset<std::size_t>(rc, right_columns, start + lr + lr_counter)
                        );
                    }
                }

                rc += rc_num;
            }
            lr += lr_num;
        }
    }, left_NR, options.num_threads);
}

/**
 * Overload of `multiply_dense_row_with_dense_column_matrix_to_row_output()` for a RHS `tatami::Matrix`.
 * This function will iterate over `left`, realizing rows into memory as needed.
 * It will also realize all of `right` into memory for fast repeated accesses.
 *
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions& options
) {
    const auto right_NC = right.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    const auto common_dim = left.ncol();
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    multiply_dense_row_with_dense_column_matrix_to_row_output(
        left,
        right_NC,
        [&](const RightIndex_ rc) -> const RightValue_* {
            return right_ptrs[rc];
        },
        output,
        options
    );
}

}

#endif
