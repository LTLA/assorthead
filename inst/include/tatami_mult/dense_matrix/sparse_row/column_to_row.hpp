#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_COLUMN_TO_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file column_to_row.hpp
 * @brief Sparse row-major LHS, dense column-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_dense_column_matrix_to_row_output()`.
 */
struct MultiplySparseRowWithDenseColumnMatrixToRowOutputOptions {
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
 * On output, this stores the matrix product in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightColumn_, typename Output_>
void multiply_sparse_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightColumn_ get_right_column,
    Output_* const output,
    const MultiplySparseRowWithDenseColumnMatrixToRowOutputOptions& options
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
                    output[sanisizer::nd_offset<std::size_t>(rc, right_columns, start + lr)] = sparse_dot_product<accumulators_>(
                        range.number, // implicit cast of range.number to size_t is safe, as per the tatami contract.
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
        std::vector<LeftIndex_> left_non_empty;
        {
            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            left_vbuffers.reserve(max_block_rows);
            left_ibuffers.reserve(max_block_rows);
            for (LeftIndex_ lr = 0; lr < max_block_rows; ++lr) {
                left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(common_dim));
            }
            sanisizer::resize(left_ranges, max_block_rows);
            left_non_empty.reserve(max_block_rows);
        }

        LeftIndex_ lr = 0;
        while (lr < length) {
            // We only consider the LHS rows with at least one structural non-zero.
            // Thus, our block consists of 'options.block_size' non-empty LHS rows, rather than fixed row-wise chunks of the LHS matrix.
            // This ensures that we don't waste iterations on LHS rows that will only have zeros in the output matrix (and are filled as such).
            const auto left_block_info = fetch_non_empty_sparse_block(
                *ext,
                left_vbuffers,
                left_ibuffers,
                left_ranges,
                left_non_empty,
                lr,
                length,
                options.block_size,
                /* zero = */ [&](const LeftIndex_ lr_copy) -> void {
                    std::fill_n(output + sanisizer::product_unsafe<std::size_t>(start + lr_copy, right_columns), right_columns, 0);
                }
            );
            const auto lr_num = left_block_info.num_non_empty;

            // If the LHS columns are all non-empty, we can speed up the loops by just using a simple counter to get the column indices.
            // Otherwise, we'll have to access the 'left_non_empty' vector to figure out the indices of each non-empty column.
            if (left_block_info.all_non_empty) {
                const auto lr_base = lr + start; 

                // Yes, we deliberately iterate over the RHS columns in the outer loop to keep the dense column in cache across multiple LHS vectors.
                // If we did it the other way around, this would defeat the purpose of blocking.
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    const auto rcol = get_right_column(rc);
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto& currange = left_ranges[lr_counter];
                        output[sanisizer::nd_offset<std::size_t>(rc, right_columns, lr_base + lr_counter)] = sparse_dot_product<accumulators_>(
                            currange.number, // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output_>(0)
                        );
                    }
                }

            } else {
                for (auto& lrne : left_non_empty) {
                    lrne += start;
                }
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) { // again, iterating over the RHS columns in the outer loop, see above.
                    const auto rcol = get_right_column(rc);
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto& currange = left_ranges[lr_counter];
                        output[sanisizer::nd_offset<std::size_t>(rc, right_columns, left_non_empty[lr_counter])] = sparse_dot_product<accumulators_>(
                            currange.number, // see above.
                            currange.value,
                            currange.index,
                            rcol,
                            static_cast<Output_>(0)
                        );
                    }
                }
            }

            lr = left_block_info.position;
        }
    }, left_NR, options.num_threads);
}

/**
 * Overload of `multiply_sparse_row_with_dense_column_matrix_to_row_output()` for a RHS `tatami::Matrix`.
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
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_dense_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithDenseColumnMatrixToRowOutputOptions& options
) {
    const auto right_NC = right.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    const auto common_dim = left.ncol();
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    multiply_sparse_row_with_dense_column_matrix_to_row_output<accumulators_>(
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
