#ifndef TATAMI_MULT_SINGLE_VECTOR_DISPATCH_HPP
#define TATAMI_MULT_SINGLE_VECTOR_DISPATCH_HPP

#include "dense_row.hpp"
#include "dense_column.hpp"
#include "sparse_row.hpp"
#include "sparse_column.hpp"

/**
 * @file dispatch.hpp
 * @brief Any matrix LHS, single vector RHS.
 */

namespace tatami_mult {

/**
 * @brief Options for `multiply_with_single_vector()`.
 */
struct MultiplyWithSingleVectorOptions {
    /**
     * Options to pass to `multiply_dense_row_with_single_vector()`, if `left` is a dense matrix that prefers row access.
     */
    MultiplyDenseRowWithSingleVectorOptions dense_row;

    /**
     * Options to pass to `multiply_dense_column_with_single_vector()`, if `left` is a dense matrix that prefers column access.
     */
    MultiplyDenseColumnWithSingleVectorOptions dense_column;

    /**
     * Options to pass to `multiply_sparse_row_with_single_vector()`, if `left` is a sparse matrix that prefers row access.
     */
    MultiplySparseRowWithSingleVectorOptions sparse_row;

    /**
     * Options to pass to `multiply_sparse_column_with_single_vector()`, if `left` is a sparse matrix that prefers column access.
     */
    MultiplySparseColumnWithSingleVectorOptions sparse_column;
};

/**
 * Set the number of threads to use in all multiplication functions involving single vector RHS.
 * Different numbers of threads may slightly change the results depending on the choice of delegated function.
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyWithSingleVectorOptions& options, int num_threads) {
    options.dense_row.num_threads = num_threads;
    options.dense_column.num_threads = num_threads;
    options.sparse_row.num_threads = num_threads;
    options.sparse_column.num_threads = num_threads;
}

/**
 * This function delegates to `multiply_sparse_row_with_single_vector()`,
 * `multiply_sparse_column_with_single_vector()`,
 * `multiply_dense_row_with_single_vector()`,
 * or `multiply_dense_column_with_single_vector()`,
 * depending on the properties of `left`.
 * 
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam Value_ Numeric type of the LHS matrix value.
 * @tparam Index_ Integer type of the LHS matrix index.
 * @tparam Right_ Numeric type of the RHS vector. 
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Value_, typename Index_, typename Right_, typename Output_>
void multiply_with_single_vector(
    const tatami::Matrix<Value_, Index_>& left,
    const Right_* const right,
    Output_* const output,
    const MultiplyWithSingleVectorOptions& options
) {
    if (left.is_sparse()) {
        if (left.prefer_rows()) {
            multiply_sparse_row_with_single_vector<accumulators_>(left, right, output, options.sparse_row);
        } else {
            multiply_sparse_column_with_single_vector(left, right, output, options.sparse_column);
        }
    } else {
        if (left.prefer_rows()) {
            multiply_dense_row_with_single_vector<accumulators_>(left, right, output, options.dense_row);
        } else {
            multiply_dense_column_with_single_vector(left, right, output, options.dense_column);
        }
    }
}

/**
 * Overload that wraps `right` in a `tatami::DelayedTranspose` and calls `multiply_with_single_vector()`.
 * 
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam Right_ Numeric type of the LHS vector. 
 * @tparam Value_ Numeric type of the RHS matrix value.
 * @tparam Index_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param[in] left Pointer to an array of length equal to the number of rows of `right`, containing the LHS vector.
 * @param right RHS matrix to be multiplied.
 * @param[out] output Pointer to an array of length equal to the number of columns of `right`.
 * On output, this stores the product `t(left) * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename Left_, typename Value_, typename Index_, typename Output_>
void multiply_with_single_vector(
    const Left_* const left,
    const tatami::Matrix<Value_, Index_>& right,
    Output_* const output,
    const MultiplyWithSingleVectorOptions& options
) {
    auto tright = tatami::make_DelayedTranspose(tatami::wrap_shared_ptr(&right));
    multiply_with_single_vector<accumulators_>(*tright, left, output, options);
}

}

#endif
