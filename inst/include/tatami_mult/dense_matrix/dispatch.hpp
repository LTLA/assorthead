#ifndef TATAMI_MULT_DENSE_MATRIX_DISPATCH_HPP
#define TATAMI_MULT_DENSE_MATRIX_DISPATCH_HPP

#include "dense_row/dispatch.hpp"
#include "dense_column/dispatch.hpp"
#include "sparse_row/dispatch.hpp"
#include "sparse_column/dispatch.hpp"

/**
 * @file dispatch.hpp
 * @brief Any matrix LHS, dense matrix RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_with_dense_matrix()`.
 */
struct MultiplyWithDenseMatrixOptions {
    /**
     * Options to pass to `multiply_dense_row_with_dense_matrix()`, if `left` is a dense matrix that prefers row access.
     */
    MultiplyDenseRowWithDenseMatrixOptions dense_row;

    /**
     * Options to pass to `multiply_dense_column_with_dense_matrix()`, if `left` is a dense matrix that prefers column access.
     */
    MultiplyDenseColumnWithDenseMatrixOptions dense_column;

    /**
     * Options to pass to `multiply_sparse_row_with_dense_matrix()`, if `left` is a sparse matrix that prefers row access.
     */
    MultiplySparseRowWithDenseMatrixOptions sparse_row;

    /**
     * Options to pass to `multiply_sparse_column_with_dense_matrix()`, if `left` is a sparse matrix that prefers column access.
     */
    MultiplySparseColumnWithDenseMatrixOptions sparse_column;
};

/**
 * Set the number of threads to use in all multiplication functions involving a dense matrix RHS.
 * Different numbers of threads may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyWithDenseMatrixOptions& options, int num_threads) {
    set_num_threads(options.dense_row, num_threads);
    set_num_threads(options.dense_column, num_threads);
    set_num_threads(options.sparse_row, num_threads);
    set_num_threads(options.sparse_column, num_threads);
}

/**
 * Set the primary block size to use in all multiplication functions involving a dense matrix LHS and a dense matrix RHS.
 * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 *
 * @param options Options to be set.
 * @param primary_block_size Primary block size.
 */
inline void set_dense_primary_block_size(MultiplyWithDenseMatrixOptions& options, int primary_block_size) {
    set_dense_primary_block_size(options.dense_row, primary_block_size);
    set_dense_primary_block_size(options.dense_column, primary_block_size);
}

/**
 * Set the secondary block size to use in all multiplication functions involving a dense matrix LHS and a dense matrix RHS.
 * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 * Different secondary block sizes may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param secondary_block_size Secondary block size.
 */
inline void set_dense_secondary_block_size(MultiplyWithDenseMatrixOptions& options, int secondary_block_size) {
    set_dense_secondary_block_size(options.dense_row, secondary_block_size);
    set_dense_secondary_block_size(options.dense_column, secondary_block_size);
}

/**
 * Set the block size to use in all multiplication functions involving a sparse matrix LHS and a dense matrix RHS.
 * See the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
 *
 * @param options Options to be set.
 * @param block_size Block size.
 */
inline void set_sparse_block_size(MultiplyWithDenseMatrixOptions& options, int block_size) {
    set_sparse_block_size(options.sparse_row, block_size);
    set_sparse_block_size(options.sparse_column, block_size);
}

/**
 * This function delegates to `multiply_sparse_row_with_dense_matrix()`,
 * `multiply_sparse_column_with_dense_matrix()`,
 * `multiply_dense_row_with_dense_matrix()`, or
 * `multiply_dense_column_with_dense_matrix()`,
 * depending on the properties of `left`.
 *
 * This function will iterate over `left`, realizing rows/columns into memory as needed.
 * It may either simultaneously iterate over `right` or realize all of `right` into memory for fast repeated accesses,
 * depending on the choice of delegated function.
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
 * @param right RHS matrix to be multiplied.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * This function is optimized for dense matrices, but will work with all matrices.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in either row- or column-major format depending on `output_row_major`.
 * @param output_row_major Whether to store the matrix product in row-major format in `output`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_with_dense_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const bool output_row_major,
    const MultiplyWithDenseMatrixOptions& options
) {
    if (left.is_sparse()) {
        if (left.prefer_rows()) {
            multiply_sparse_row_with_dense_matrix<accumulators_>(left, right, output, output_row_major, options.sparse_row);
        } else {
            multiply_sparse_column_with_dense_matrix(left, right, output, output_row_major, options.sparse_column);
        }
    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_dense_matrix<accumulators_>(left, right, output, output_row_major, options.dense_row);
        } else {
            multiply_dense_column_with_dense_matrix(left, right, output, output_row_major, options.dense_column);
        }
    }
}

}

#endif
