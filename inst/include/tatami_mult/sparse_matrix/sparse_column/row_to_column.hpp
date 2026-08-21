#ifndef TATAMI_MULT_SPARSE_MATRIX_SPARSE_COLUMN_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_SPARSE_MATRIX_SPARSE_COLUMN_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_column.hpp
 * @brief Sparse row-major LHS, sparse row-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_column/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_column_with_sparse_row_matrix_to_column_output()`.
 */
struct MultiplySparseColumnWithSparseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;
};

/**
 * This function will iterate over both `left` and `right` simultaneously, realizing columns and rows respectively into memory as needed.
 *
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_column_with_sparse_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseColumnWithSparseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);

    const int num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<true>(right, true, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        auto left_vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
        auto left_ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
        auto right_vbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(right_NC);
        auto right_ibuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(right_NC);

        for (LeftIndex_ cd = 0; cd < length; ++cd) {
            const auto lrange = left_ext->fetch(left_vbuffer.data(), left_ibuffer.data());
            const auto rrange = right_ext->fetch(right_vbuffer.data(), right_ibuffer.data());

            // Skip should be after all fetch calls, otherwise extractors will go out of sync.
            if (lrange.number == 0 || rrange.number == 0) {
                continue;
            }

            for (RightIndex_ x = 0; x < rrange.number; ++x) {
                const auto idx = rrange.index[x]; 
                const Output_ mult = rrange.value[x];
                for (LeftIndex_ y = 0; y < lrange.number; ++y) {
                    outptr[sanisizer::nd_offset<std::size_t>(lrange.index[y], left_NR, idx)] += mult * static_cast<Output_>(lrange.value[y]);
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
