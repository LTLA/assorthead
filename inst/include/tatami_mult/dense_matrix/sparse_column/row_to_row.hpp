#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_ROW_TO_ROW_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Sparse column-major LHS, dense row-major matrix RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_column/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_column_with_dense_row_matrix_to_row_output()`.
 */
struct MultiplySparseColumnWithDenseRowMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
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
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_row Function that accepts a `LeftIndex_` in `[0, left.ncol())` and returns a pointer to an array of length `right_columns`.
 * The array referenced by `get_right_row(i)` represents the `i`-th row of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this contains the matrix product in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, typename GetRightRow_, typename Output_>
void multiply_sparse_column_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightRow_ get_right_row,
    Output_* const output,
    const MultiplySparseColumnWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    // Product must fit in a size_t in order for output to have been allocated correctly in the first place.
    // Technically, right_columns could be larger than a size_t if left_NR == 0, but the product after wraparound would still be zero, so it's fine.
    std::fill_n(output, sanisizer::product<std::size_t>(left_NR, right_columns), 0);

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_columns));
            outptr = tmp_output->data();
        }

        for (LeftIndex_ cd = 0; cd < length; ++cd) {
            const auto lrange = left_ext->fetch(vbuffer.data(), ibuffer.data());
            const auto rptr = get_right_row(start + cd);
            for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                const Output_ mult = lrange.value[x];
                const auto curout = outptr + sanisizer::product_unsafe<std::size_t>(lrange.index[x], right_columns);
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    curout[rc] += mult * static_cast<Output_>(rptr[rc]); 
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            const auto N = tmp.size();
            for (I<decltype(N)> x = 0; x < N; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

/**
 * Overload of `multiply_sparse_column_with_dense_row_matrix_to_row_output()` for a RHS `tatami::Matrix`.
 * This function will iterate over both `left` and `right` simultaneously, realizing columns and rows respectively into memory as needed.
 *
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in `right` should be equal to the number of columns in `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, this contains the product `left * right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_column_with_dense_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseColumnWithDenseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    std::fill_n(output, sanisizer::product<std::size_t>(left_NR, right_NC), 0);

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<false>(right, true, start, length);

        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
        auto rbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(right_NC);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        for (LeftIndex_ cd = 0; cd < length; ++cd) {
            const auto lrange = left_ext->fetch(vbuffer.data(), ibuffer.data());
            const auto rptr = right_ext->fetch(rbuffer.data());
            for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                const Output_ mult = lrange.value[x];
                const auto curout = outptr + sanisizer::product_unsafe<std::size_t>(lrange.index[x], right_NC);
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    curout[rc] += mult * static_cast<Output_>(rptr[rc]); 
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            const auto N = tmp.size();
            for (I<decltype(N)> x = 0; x < N; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

}

#endif
