#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_COLUMN_TO_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file column_to_row.hpp
 * @brief Dense row-major LHS, sparse column-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_sparse_column_matrix_to_row_output()`.
 */
struct MultiplyDenseRowWithSparseColumnMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to be loaded at once.
     * See the \f$C\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 1;
};

/**
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
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_sparse_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithSparseColumnMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(right_NC);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(right_NC);
    populate_sparse_buffers(false, right_NC, common_dim, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = ext->fetch(dbuffer.data());

                // No point looping over the non-empty RHS columns, as we still need to zero the output columns corresponding to empty RHS columns.
                // So, we might as well handle the zeroing in the same loop and save ourselves the trouble.
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    const auto rrange = right_ranges[rc];
                    output[sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr)] = sparse_dot_product<accumulators_>(
                        rrange.number, // Implicit cast to size_t is safe, as per the tatami contract.
                        rrange.value,
                        rrange.index,
                        lptr,
                        static_cast<Output_>(0)
                    );
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            std::vector<std::vector<LeftValue_> > lbuffers;
            lbuffers.reserve(max_block_rows);
            for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
                lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
            }
            auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    lptrs[lr_counter] = ext->fetch(lbuffers[lr_counter].data());
                }

                // Deliberately iterating over the sparse RHS columns in the outer loop and the dense LHS rows in the inner loop.
                // This aims to keep the entirety of the dense LHS block in cache across multiple RHS columns, provided common_dim is small.
                // If we did it the other way around, it would just be the same as the block_size == 1 case, but with more looping overhead. 
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    const auto rrange = right_ranges[rc];
                    if (rrange.number == 0) {
                        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            output[sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr + lr_counter)] = 0;
                        }
                        continue;
                    }

                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        output[sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr + lr_counter)] = sparse_dot_product<accumulators_>(
                            rrange.number, // Implicit cast to size_t is safe, as per the tatami contract.
                            rrange.value,
                            rrange.index,
                            lptrs[lr_counter],
                            static_cast<Output_>(0)
                        );
                    }
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
