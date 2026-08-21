#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_ROW_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Sparse row-major LHS, dense row-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_dense_row_matrix_to_row_output()`.
 */
struct MultiplySparseRowWithDenseRowMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightRow_ Functor that accepts a `LeftIndex_` and returns a pointer to an RHS row.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_row Function that accepts a `LeftIndex_` in `[0, left.ncol())` and returns a pointer to an array of length `right_columns`.
 * The array referenced by `get_right_row(i)` represents the `i`-th row of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this stores the matrix product in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightRow_, typename Output_>
void multiply_sparse_row_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightRow_ get_right_row,
    Output_* const output,
    const MultiplySparseRowWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_columns), 0);
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

        std::optional<std::vector<Output_> > tmp_output;
        if (do_parallel) {
            tmp_output.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(right_columns));
        }

        for (LeftIndex_ lr = 0; lr < length; ++lr) {
            const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            const auto optr =  output + sanisizer::product_unsafe<std::size_t>(start + lr, right_columns);
            const auto tmp_optr = (do_parallel ? tmp_output->data() : optr);

            for (LeftIndex_ x = 0; x < range.number; ++x) {
                const auto rightrow = get_right_row(range.index[x]);
                const auto mult = range.value[x];
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    tmp_optr[rc] += mult * rightrow[rc];
                }
            }

            if (do_parallel) {
                if (range.number == 0) {
                    // If it's empty, we would have never modified the temporary buffer,
                    // so we can proceed to directly zeroing the output array.
                    std::fill_n(optr, right_columns, 0);
                } else {
                    std::copy_n(tmp_output->data(), right_columns, optr);
                    std::fill_n(tmp_output->data(), right_columns, 0);
                }
            }
        }
    }, left_NR, options.num_threads);
}

/**
 * Overload of `multiply_sparse_row_with_dense_row_matrix_to_row_output()` for a RHS `tatami::Matrix`.
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
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto common_dim = left.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(common_dim);
    const auto right_NC = right.ncol();
    populate_dense_buffers(true, common_dim, right_NC, right, right_buffers, right_ptrs, options.num_threads);

    multiply_sparse_row_with_dense_row_matrix_to_row_output(
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
