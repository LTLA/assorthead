#ifndef TATAMI_MULT_HPP
#define TATAMI_MULT_HPP

#include "single_vector/dispatch.hpp"
#include "multiple_vectors/dispatch.hpp"
#include "dense_matrix/dispatch.hpp"
#include "sparse_matrix/dispatch.hpp"

#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file tatami_mult.hpp
 * @brief Multiplication of **tatami** matrices.
 */

/**
 * @namespace tatami_mult
 * @brief Multiplication of **tatami** matrices.
 */
namespace tatami_mult {

/**
 * @brief Options for `multiply_with_matrix()`.
 */
struct MultiplyWithMatrixOptions {
    /**
     * Options to pass to `multiply_row_with_dense_matrix()`, if `right` is a dense matrix that prefers row access.
     */
    MultiplyWithDenseMatrixOptions dense_matrix;

    /**
     * Options to pass to `multiply_row_with_sparse_matrix()`, if `right` is a sparse matrix that prefers row access.
     */
    MultiplyWithSparseMatrixOptions sparse_matrix;

    /**
     * Whether to set the larger matrix as the LHS matrix in the delegated functions.
     * If `right` is larger, we transpose and swap `left` and `right` prior to calling the delegated functions.
     * This ensures that we only pass over the larger matrix once while potentially passing through the smaller matrix multiple times. 
     * The result does not change though the delegated function will now be chosen based on the transposed `left`.
     *
     * If this is false (or `right` is already smaller), the multiplication is performed exactly with the supplied left and right matrices.
     */
    bool larger_left = true;
};

/**
 * Set the number of threads to use in all multiplication functions involving two matrices.
 * Different numbers of threads may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param num_threads Number of threads, should be positive.
 */
inline void set_num_threads(MultiplyWithMatrixOptions& options, int num_threads) {
    set_num_threads(options.dense_matrix, num_threads);
    set_num_threads(options.sparse_matrix, num_threads);
}

/**
 * Set the primary block size to use in all multiplication functions involving two dense matrices.
 * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 *
 * @param options Options to be set.
 * @param primary_block_size Primary block size.
 */
inline void set_dense_primary_block_size(MultiplyWithMatrixOptions& options, int primary_block_size) {
    set_dense_primary_block_size(options.dense_matrix, primary_block_size);
}

/**
 * Set the secondary block size to use in all multiplication functions involving two dense matrices.
 * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
 * Different secondary block sizes may slightly change the results due to differences in floating-point round-off error, depending on the delegated function.
 *
 * @param options Options to be set.
 * @param secondary_block_size Secondary block size.
 */
inline void set_dense_secondary_block_size(MultiplyWithMatrixOptions& options, int secondary_block_size) {
    set_dense_secondary_block_size(options.dense_matrix, secondary_block_size);
}

/**
 * Set the block size to use in all multiplication functions involving a sparse matrix. 
 * See the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
 *
 * @param options Options to be set.
 * @param block_size Block size.
 */
inline void set_sparse_block_size(MultiplyWithMatrixOptions& options, int block_size) {
    set_sparse_block_size(options.dense_matrix, block_size);
    set_sparse_block_size(options.sparse_matrix, block_size);
}

/**
 * This function delegates to `multiply_with_dense_matrix()` or `multiply_with_sparse_matrix()`,
 * depending on the properties of `left`, `right` and the choice of `MultiplyWithMatrixOptions::larger_left`.
 *
 * This function will iterate over `left`, realizing rows/columns into memory as needed.
 * It may either simultaneously iterate over `right` or realize all of `right` into memory for fast repeated accesses.
 * If `MultiplyWithMatrixOptions::larger_left = true` and `right` is larger, this function will iterate over `right` instead, and may realize `left` into memory.
 *
 * depending on the choice of delegated function.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 *
 * @param left LHS matrix to be multiplied.
 * @param right RHS matrix to be multiplied.
 * `right.nrow()` and `left.ncol()` should be equal.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`. 
 * On output, this stores the product of `left` and `right` in either row- or column-major format depending on `output_row_major`.
 * @param output_row_major Whether to store the matrix product in row-major format in `output`.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_with_matrix(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const bool output_row_major,
    const MultiplyWithMatrixOptions& options
) {
    if (options.larger_left) {
        if (sanisizer::is_less_than(left.nrow(), right.ncol())) {
            auto tright = tatami::make_DelayedTranspose(tatami::wrap_shared_ptr(&right));
            auto tleft = tatami::make_DelayedTranspose(tatami::wrap_shared_ptr(&left));
            if (tleft->is_sparse()) {
                multiply_with_sparse_matrix(*tright, *tleft, output, !output_row_major, options.sparse_matrix);
            } else {
                multiply_with_dense_matrix(*tright, *tleft, output, !output_row_major, options.dense_matrix);
            }
            return;
        }
    }

    if (right.is_sparse()) {
        multiply_with_sparse_matrix(left, right, output, output_row_major, options.sparse_matrix);
    } else {
        multiply_with_dense_matrix(left, right, output, output_row_major, options.dense_matrix);
    }
}

/**
 * @cond
 */
// For back-compatibility only.
struct Options {
    int num_threads = 1;
    bool prefer_larger = true;
    bool column_major_output = true;
};

// For back-compatibility only.
template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply(const tatami::Matrix<Value_, Index_>& left, const Right_* right, Output_* output, const Options& opt) {
    multiply_with_single_vector(
        left,
        right,
        output,
        [&](){
            MultiplyWithSingleVectorOptions mopt;
            set_num_threads(mopt, opt.num_threads);
            return mopt;
        }()
    );
}

// For back-compatibility only.
template<typename Left_, typename Value_, typename Index_, typename Output_>
void multiply(const Left_* left, const tatami::Matrix<Value_, Index_>& right, Output_* output, const Options& opt) {
    multiply_with_single_vector(
        left,
        right,
        output,
        [&](){
            MultiplyWithSingleVectorOptions mopt;
            set_num_threads(mopt, opt.num_threads);
            return mopt;
        }()
    );
}

// For back-compatibility only.
template<typename Value_, typename Index_, typename Right_, typename Output_>
void multiply(const tatami::Matrix<Value_, Index_>& left, const std::vector<Right_*>& right, const std::vector<Output_*>& output, const Options& opt) {
    multiply_with_multiple_vectors(
        left,
        right,
        output,
        [&](){
            MultiplyWithMultipleVectorsOptions mopt;
            set_num_threads(mopt, opt.num_threads);
            return mopt;
        }()
    );
}

// For back-compatibility only.
template<typename Left_, typename Value_, typename Index_, typename Output_>
void multiply(const std::vector<Left_*>& left, const tatami::Matrix<Value_, Index_>& right, const std::vector<Output_*>& output, const Options& opt) {
    multiply_with_multiple_vectors(
        left,
        right,
        output,
        [&](){
            MultiplyWithMultipleVectorsOptions mopt;
            set_num_threads(mopt, opt.num_threads);
            return mopt;
        }()
    );
}

// For back-compatibility only.
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply(const tatami::Matrix<LeftValue_, LeftIndex_>& left, const tatami::Matrix<RightValue_, RightIndex_>& right, Output_* const output, const Options& opt) {
    multiply_with_matrix(
        left,
        right,
        output,
        !opt.column_major_output,
        [&](){
            MultiplyWithMatrixOptions mopt;
            set_num_threads(mopt, opt.num_threads);
            mopt.larger_left = opt.prefer_larger;
            return mopt;
        }()
    );
}
/**
 * @endcond
 */

}

#endif
