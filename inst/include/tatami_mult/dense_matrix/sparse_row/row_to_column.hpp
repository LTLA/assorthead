#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_column.hpp
 * @brief Sparse row-major LHS, dense row-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_dense_row_matrix_to_column_output()`.
 */
struct MultiplySparseRowWithDenseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to use at once.
     * The matrix product is computed for the submatrix consisting of each block of rows,
     * and then transposed for storage in the column-major output array.
     *
     * The block size should be positive.
     * Larger values generally improve speed at the cost of increased memory usage.
     * If this is set to 1, no blocking is performed.
     */
    int block_size = 16;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightRow_ Functor that accepts a `LeftIndex_` and returns a pointer to an RHS row.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_row Function that accepts a `LeftIndex_` in `[0, left.ncol())` and returns a pointer to an array of length `right_columns`.
 * The array referenced by `get_right_row(i)` represents the `i`-th row of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this stores the matrix product in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightRow_, typename Output_>
void multiply_sparse_row_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightRow_ get_right_row,
    Output_* const output,
    const MultiplySparseRowWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

        if (options.block_size == 1) {
            auto tmp_output = tatami::create_container_of_Index_size<std::vector<Output_> >(right_columns);
            std::vector<LeftIndex_> left_empty;

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                if (range.number == 0) {
                    left_empty.push_back(lr);
                    continue;
                }

                for (LeftIndex_ x = 0; x < range.number; ++x) {
                    const auto rightrow = get_right_row(range.index[x]);
                    const Output_ mult = range.value[x];
                    for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                        tmp_output[rc] += mult * static_cast<Output_>(rightrow[rc]);
                    }
                }

                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = tmp_output[rc];
                }
                std::fill(tmp_output.begin(), tmp_output.end(), 0);
            }

            if (left_empty.size()) {
                // Zeroing the empty rows that we previously skipped. This is done with near-contiguous memory,
                // so if there are many empty LHS rows, their special-casing should improve efficiency.
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    for (const auto lr : left_empty) {
                        output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = 0;
                    }
                }
            }

        } else {
            const auto max_block_rows = sanisizer::min(length, options.block_size);
            std::vector<Output_> tmp_output(sanisizer::product<typename std::vector<Output_>::size_type>(max_block_rows, right_columns));

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lrnum = sanisizer::min(options.block_size, length - lr);
                bool any_non_empty = false;
                for (LeftIndex_ lrcopy = 0; lrcopy < lrnum; ++lrcopy) {
                    const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                    if (range.number == 0) {
                        continue;
                    }
                    any_non_empty = true;

                    for (LeftIndex_ x = 0; x < range.number; ++x) {
                        const auto rightrow = get_right_row(range.index[x]);
                        const Output_ mult = range.value[x];
                        for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                            tmp_output[sanisizer::nd_offset<std::size_t>(rc, right_columns, lrcopy)] += mult * static_cast<Output_>(rightrow[rc]);
                        }
                    }
                }

                // Now doing a blocked transposition.
                // This is, in fact, the only purpose of the blocking here.
                // We do this even if there were no non-empty LHS rows, because we have to zero the output array anyway.
                RightColumns_ rc = 0;
                while (rc < right_columns) {
                    const RightColumns_ rcend = rc + sanisizer::min(options.block_size, right_columns - rc);
                    for (LeftIndex_ lrcopy = 0; lrcopy < lrnum; ++lrcopy) {
                        for (auto rcopy = rc; rcopy < rcend; ++rcopy) {
                            const auto val = tmp_output[sanisizer::nd_offset<std::size_t>(rcopy, right_columns, lrcopy)];
                            output[sanisizer::nd_offset<std::size_t>(start + lr + lrcopy, left_NR, rcopy)] = val;
                        }
                    }
                    rc = rcend;
                }

                // Too much effort to track individual non-empty rows, but if they're all empty, we skip the zeroing.
                // If a few are non-empty, a single memset call is probably faster anyway than splitting it up.
                if (any_non_empty) {
                    std::fill_n(tmp_output.begin(), sanisizer::product_unsafe<std::size_t>(right_columns, lrnum), 0);
                }

                lr += lrnum;
            }
        }
    }, left_NR, options.num_threads);
}

/**
 * Overload of `multiply_sparse_row_with_dense_row_matrix_to_column_output()` for a RHS `tatami::Matrix`.
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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto common_dim = left.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(common_dim);
    const auto right_NC = right.ncol();
    populate_dense_buffers(true, common_dim, right_NC, right, right_buffers, right_ptrs, options.num_threads);

    multiply_sparse_row_with_dense_row_matrix_to_column_output(
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
