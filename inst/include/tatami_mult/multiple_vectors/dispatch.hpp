#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DISPATCH_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DISPATCH_HPP

#include "dense_row.hpp"
#include "dense_column.hpp"
#include "sparse_row.hpp"
#include "sparse_column.hpp"

/**
 * @file dispatch.hpp
 * @brief Any matrix LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_with_multiple_vectors()`.
 */
struct MultiplyWithMultipleVectorsOptions {
    /**
     * Options to pass to `multiply_dense_row_with_multiple_vectors()`, if `left` is a dense matrix that prefers row access.
     */
    MultiplyDenseRowWithMultipleVectorsOptions dense_row;

    /**
     * Options to pass to `multiply_dense_column_with_multiple_vectors()`, if `left` is a dense matrix that prefers column access.
     */
    MultiplyDenseColumnWithMultipleVectorsOptions dense_column;

    /**
     * Options to pass to `multiply_sparse_row_with_multiple_vectors()`, if `left` is a sparse matrix that prefers row access.
     */
    MultiplySparseRowWithMultipleVectorsOptions sparse_row;

    /**
     * Options to pass to `multiply_sparse_column_with_multiple_vectors()`, if `left` is a sparse matrix that prefers column access.
     */
    MultiplySparseColumnWithMultipleVectorsOptions sparse_column;
};

/**
 * Set the number of threads to use in all multiplication functions involving multiple vectors RHS.
 * Different numbers of threads may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyWithMultipleVectorsOptions& options, int num_threads) {
    options.dense_row.num_threads = num_threads;
    options.dense_column.num_threads = num_threads;
    options.sparse_row.num_threads = num_threads;
    options.sparse_column.num_threads = num_threads;
}

/**
 * Set the primary block size to use in all multiplication functions involving a dense matrix LHS and multiple vectors RHS.
 * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 *
 * @param options Options to be set.
 * @param primary_block_size Primary block size.
 */
inline void set_dense_primary_block_size(MultiplyWithMultipleVectorsOptions& options, int primary_block_size) {
    options.dense_row.primary_block_size = primary_block_size;
    options.dense_column.primary_block_size = primary_block_size;
}

/**
 * Set the secondary block size to use in all multiplication functions involving a dense matrix LHS and multiple vectors RHS.
 * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 * Different secondary block sizes may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param secondary_block_size Secondary block size.
 */
inline void set_dense_secondary_block_size(MultiplyWithMultipleVectorsOptions& options, int secondary_block_size) {
    options.dense_row.secondary_block_size = secondary_block_size;
    options.dense_column.secondary_block_size = secondary_block_size;
}

/**
 * Set the block size to use in all multiplication functions involving a sparse matrix LHS and multiple vectors RHS.
 * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
 *
 * @param options Options to be set.
 * @param block_size Block size.
 */
inline void set_sparse_block_size(MultiplyWithMultipleVectorsOptions& options, int block_size) {
    options.sparse_row.block_size = block_size;
    options.sparse_column.block_size = block_size;
}

/**
 * This function delegates to `multiply_sparse_row_with_multiple_vectors()`,
 * `multiply_sparse_column_with_multiple_vectors()`,
 * `multiply_dense_row_with_multiple_vectors()`, or
 * `multiply_dense_column_with_multiple_vectors()`,
 * depending on the properties of `left`.
 *
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam Value_ Numeric type of the LHS matrix value.
 * @tparam Index_ Integer type of the LHS matrix index.
 * @tparam Right_ Numeric type of the RHS vectors.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains a RHS vector with which to multiply `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_with_multiple_vectors(
    const tatami::Matrix<Value_, Index_>& left,
    const std::vector<Right_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyWithMultipleVectorsOptions& options
) {
    if (left.is_sparse()) {
        if (left.prefer_rows()) {
            multiply_sparse_row_with_multiple_vectors<accumulators_>(left, right, output, options.sparse_row);
        } else {
            multiply_sparse_column_with_multiple_vectors(left, right, output, options.sparse_column);
        }
    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_multiple_vectors<accumulators_>(left, right, output, options.dense_row);
        } else {
            multiply_dense_column_with_multiple_vectors(left, right, output, options.dense_column);
        }
    }
}

/**
 * Overload that wraps `right` in a `tatami::DelayedTranspose` and calls `multiply_with_multiple_vectors()`.
 * 
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam Left_ Numeric type of the LHS vectors. 
 * @tparam Value_ Numeric type of the RHS matrix value.
 * @tparam Index_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param[in] left Vector of pointers, each of which points to an array of length `right.nrow()`.
 * Each entry contains a LHS vector with which to multiply `right`.
 * @param right RHS matrix to be multiplied.
 * @param[out] output Pointer to an array of length equal to the number of columns of `right`.
 * On output, the `i`-th entry stores the product `t(left[i]) * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Left_, typename Value_, typename Index_, typename Output_>
void multiply_with_multiple_vectors(
    const std::vector<Left_*>& left,
    const tatami::Matrix<Value_, Index_>& right,
    const std::vector<Output_*>& output,
    const MultiplyWithMultipleVectorsOptions& options
) {
    auto tright = tatami::make_DelayedTranspose(tatami::wrap_shared_ptr(&right));
    multiply_with_multiple_vectors<accumulators_>(*tright, left, output, options);
}

}

#endif
