#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_COLUMN_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_COLUMN_TO_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file column_to_column.hpp
 * @brief Sparse row-major LHS, dense column-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_dense_column_matrix_to_column_output()`.
 */
struct MultiplySparseRowWithDenseColumnMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to be loaded at once.
     * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 16;
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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_column Function that accepts a `RightColumns_` in `[0, right_columns)` and returns a pointer to an array of length `left.ncol()`.
 * The array referenced by `get_right_column(i)` represents the `i`-th column of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this stores the matrix product in column-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightColumn_, typename Output_>
void multiply_sparse_row_with_dense_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightColumn_ get_right_column,
    Output_* const output,
    const MultiplySparseRowWithDenseColumnMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = sparse_dot_product<accumulators_>(
                        range.number, // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                        range.value,
                        range.index,
                        get_right_column(rc),
                        static_cast<Output_>(0)
                    );
                }
            }
        }, left_NR, options.num_threads);
        return;
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

        std::vector<std::vector<LeftValue_> > left_vbuffers;
        std::vector<std::vector<LeftIndex_> > left_ibuffers;
        std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
        {
            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            left_vbuffers.reserve(max_block_rows);
            left_ibuffers.reserve(max_block_rows);
            for (LeftIndex_ lr = 0; lr < max_block_rows; ++lr) {
                left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(common_dim));
            }
            sanisizer::resize(left_ranges, max_block_rows);
        }

        LeftIndex_ lr = 0;
        while (lr < length) {
            // No point skipping the LHS rows with no structural non-zeros.
            // We still need to set the corresponding entry of 'output' to zero, so we'd end up having to loop through the LHS rows anyway.
            // We might as well just let it be set to zero naturally in the existing loop below.
            const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
            for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                left_ranges[lr_counter] = ext->fetch(left_vbuffers[lr_counter].data(), left_ibuffers[lr_counter].data());
            }

            for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                const auto rcol = get_right_column(rc);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    const auto& currange = left_ranges[lr_counter];
                    output[sanisizer::nd_offset<std::size_t>(start + lr + lr_counter, left_NR, rc)] = sparse_dot_product<accumulators_>(
                        currange.number, // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                        currange.value,
                        currange.index,
                        rcol,
                        static_cast<Output_>(0)
                    );
                }
            }

            lr += lr_num;
        }
    }, left_NR, options.num_threads);
}

/**
 * Overload of `multiply_sparse_row_with_dense_column_matrix_to_column_output()` for a RHS `tatami::Matrix`.
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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_dense_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithDenseColumnMatrixToColumnOutputOptions& options
) {
    const auto right_NC = right.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    const auto common_dim = left.ncol();
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    multiply_sparse_row_with_dense_column_matrix_to_column_output<accumulators_>(
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
