#ifndef TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_COLUMN_TO_COLUMN_HPP
#define TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_COLUMN_TO_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file column_to_column.hpp
 * @brief Sparse row-major LHS, sparse column-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_sparse_column_matrix_to_column_output()`.
 */
struct MultiplySparseRowWithSparseColumnMatrixToColumnOutputOptions {
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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_sparse_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithSparseColumnMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(right_NC);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(right_NC);
    populate_sparse_buffers(false, right_NC, common_dim, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    // If there are any empty RHS columns, we only iterate over the non-empty ones in the loop for each LHS row.
    auto right_non_empty = filter_non_empty_sparse(
        right_ranges,
        [&](const RightIndex_ rc) -> void {
            std::fill_n(output + sanisizer::product_unsafe<std::size_t>(left_NR, rc), left_NR, 0);
        }
    );

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

            // We expand each sparse vector into a dense buffer for easy look-up in sparse_dot_product().
            auto expanded = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lrange = ext->fetch(vbuffer.data(), ibuffer.data());
                for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                    expanded[lrange.index[x]] = lrange.value[x];
                }

                auto loop_body = [&](RightIndex_ rc) -> void {
                    const auto rrange = right_ranges[rc];
                    output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = sparse_dot_product<accumulators_>(
                        rrange.number, // Implicit cast to size_t is safe, as per the tatami contract.
                        rrange.value,
                        rrange.index,
                        expanded.data(),
                        static_cast<Output_>(0)
                    );
                };

                if (right_non_empty.has_value()) {
                    for (const auto rc : *right_non_empty) {
                        loop_body(rc);
                    }
                } else {
                    for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                        loop_body(rc);
                    }
                }

                for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                    expanded[lrange.index[x]] = 0;
                }
            }
        }, left_NR, options.num_threads);
        return;
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

        const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
        std::vector<std::vector<LeftValue_> > lvbuffers;
        std::vector<std::vector<LeftIndex_> > libuffers;
        std::vector<std::vector<LeftValue_> > expanded;
        lvbuffers.reserve(max_block_rows);
        libuffers.reserve(max_block_rows);
        expanded.reserve(max_block_rows);
        for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
            lvbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
            libuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(common_dim));
            expanded.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
        }
        auto lranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > >(max_block_rows);

        LeftIndex_ lr = 0;
        while (lr < length) {
            // No point skipping the LHS rows with no structural non-zeros.
            // We still need to set the corresponding entry of 'output' to zero, so we'd end up having to loop through the LHS rows anyway.
            // We might as well just let it be set to zero naturally in the existing loop below.
            const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
            for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                auto lrange = ext->fetch(lvbuffers[lr_counter].data(), libuffers[lr_counter].data());
                auto& curex = expanded[lr_counter];
                for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                    curex[lrange.index[x]] = lrange.value[x];
                }
                lranges[lr_counter] = std::move(lrange);
            }

            // Deliberately iterating over the (non-empty) sparse RHS columns in the outer loop and the dense (expanded) LHS rows in the inner loop.
            // This aims to keep the entirety of the dense LHS block in cache across multiple RHS columns, provided common_dim is small.
            // If we did it the other way around, it would just be the same as the block_size == 1 case, but with more looping overhead. 
            auto loop_body = [&](RightIndex_ rc) -> void {
                const auto rrange = right_ranges[rc];
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    const auto val = sparse_dot_product<accumulators_>(
                        rrange.number, // Implicit cast to size_t is safe, as per the tatami contract.
                        rrange.value,
                        rrange.index,
                        expanded[lr_counter].data(),
                        static_cast<Output_>(0)
                    );
                    output[sanisizer::nd_offset<std::size_t>(start + lr + lr_counter, left_NR, rc)] = val;
                }
            };

            if (right_non_empty.has_value()) {
                for (const auto rc : *right_non_empty) {
                    loop_body(rc);
                }
            } else {
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    loop_body(rc);
                }
            }

            for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                const auto& lrange = lranges[lr_counter]; 
                auto& curex = expanded[lr_counter];
                for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                    curex[lrange.index[x]] = 0;
                }
            }

            lr += lr_num;
        }
    }, left_NR, options.num_threads);
}

}

#endif
