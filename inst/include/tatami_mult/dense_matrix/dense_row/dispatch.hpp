#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_DISPATCH_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_ROW_DISPATCH_HPP

#include "row_to_row.hpp"
#include "row_to_column.hpp"
#include "column_to_row.hpp"
#include "column_to_column.hpp"

#include "../../utils.hpp"

/**
 * @file dispatch.hpp
 * @brief Dense row-major LHS, dense matrix RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_dense_row_with_dense_matrix()`.
 */
struct MultiplyDenseRowWithDenseMatrixOptions {
    /**
     * Options to pass to `multiply_dense_row_with_dense_column_matrix_to_column_output()`,
     * if `right` is a column-major matrix and `output_row_major == false`.
     */
    MultiplyDenseRowWithDenseColumnMatrixToColumnOutputOptions column_to_column;

    /**
     * Options to pass to `multiply_dense_row_with_dense_column_matrix_to_row_output()`,
     * if `right` is a column-major matrix and `output_row_major == true`.
     */
    MultiplyDenseRowWithDenseColumnMatrixToRowOutputOptions column_to_row;

    /**
     * Options to pass to `multiply_dense_row_with_dense_row_matrix_to_column_output()`,
     * if `right` is a row-major matrix and `output_row_major == false`.
     */
    MultiplyDenseRowWithDenseRowMatrixToColumnOutputOptions row_to_column;

    /**
     * Options to pass to `multiply_dense_row_with_dense_row_matrix_to_row_output()`,
     * if `right` is a row-major matrix and `output_row_major == true`.
     */
    MultiplyDenseRowWithDenseRowMatrixToRowOutputOptions row_to_row;
};

/**
 * Set the number of threads to use in all multiplication functions involving a dense row-major LHS and a dense matrix RHS.
 * Different numbers of threads will not change the results. 
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyDenseRowWithDenseMatrixOptions& options, int num_threads) {
    options.column_to_column.num_threads = num_threads;
    options.column_to_row.num_threads = num_threads;
    options.row_to_column.num_threads = num_threads;
    options.row_to_row.num_threads = num_threads;
}

/**
 * Set the primary block size to use in all multiplication functions involving a dense row-major LHS and a dense matrix RHS.
 * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 *
 * @param options Options to be set.
 * @param primary_block_size Primary block size.
 */
inline void set_dense_primary_block_size(MultiplyDenseRowWithDenseMatrixOptions& options, int primary_block_size) {
    options.column_to_column.primary_block_size = primary_block_size;
    options.column_to_row.primary_block_size = primary_block_size;
    options.row_to_column.primary_block_size = primary_block_size;
    options.row_to_row.primary_block_size = primary_block_size;
}

/**
 * Set the secondary block size to use in all multiplication functions involving a dense row-major LHS and a dense matrix RHS.
 * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 * Different secondary block sizes may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param secondary_block_size Secondary block size.
 */
inline void set_dense_secondary_block_size(MultiplyDenseRowWithDenseMatrixOptions& options, int secondary_block_size) {
    options.column_to_column.secondary_block_size = secondary_block_size;
    options.column_to_row.secondary_block_size = secondary_block_size;
    options.row_to_column.secondary_block_size = secondary_block_size;
    options.row_to_row.secondary_block_size = secondary_block_size;
}

/**
 * This function delegates to `multiply_dense_row_with_dense_row_matrix_to_row_output()`,
 * `multiply_dense_row_with_dense_row_matrix_to_column_output()`,
 * `multiply_dense_row_with_dense_column_matrix_to_row_output()`, or
 * `multiply_dense_row_with_dense_column_matrix_to_column_output()`,
 * depending on the properties of `right` and the choice of `output_row_major`.
 *
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
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * This function is optimized for dense matrices, but will work with all matrices.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in either row- or column-major format depending on `output_row_major`.
 * @param output_row_major Whether to store the matrix product in row-major format in `output`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_dense_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const bool output_row_major,
    const MultiplyDenseRowWithDenseMatrixOptions& options
) {
    if (right.prefer_rows()) {
        if (output_row_major) {
            multiply_dense_row_with_dense_row_matrix_to_row_output(left, right, output, options.row_to_row);
        } else {
            multiply_dense_row_with_dense_row_matrix_to_column_output(left, right, output, options.row_to_column);
        }

    } else {
        if (output_row_major) {
            multiply_dense_row_with_dense_column_matrix_to_row_output<accumulators_>(left, right, output, options.column_to_row);
        } else {
            multiply_dense_row_with_dense_column_matrix_to_column_output<accumulators_>(left, right, output, options.column_to_column);
        }
    }
}

}

#endif
