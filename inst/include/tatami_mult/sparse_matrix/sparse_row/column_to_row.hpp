#ifndef TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_COLUMN_TO_ROW_HPP
#define TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_COLUMN_TO_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"
#include "../../sparse_dot_product.hpp"

/**
 * @file column_to_row.hpp
 * @brief Sparse row-major LHS, sparse column-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_sparse_column_matrix_to_row_output()`.
 */
struct MultiplySparseRowWithSparseColumnMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to be loaded at once.
     * See the \f$C\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     *
     * If this is set to 1, no blocking is performed.
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
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_sparse_column_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithSparseColumnMatrixToRowOutputOptions& options
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

                // No point looping over the non-empty RHS columns, as we still need to zero the output columns corresponding to empty RHS columns.
                // So, we might as well handle the zeroing in the same loop and save ourselves the trouble.
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    const auto rrange = right_ranges[rc];

                    // Some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                    output[sanisizer::nd_offset<std::size_t>(rc, right_NC, start + lr)] = sparse_dot_product<accumulators_>(
                        rrange.number, // Implicit cast to size_t is safe, as per the tatami contract.
                        rrange.value,
                        rrange.index,
                        expanded.data(),
                        static_cast<Output_>(0)
                    );
                }

                for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                    expanded[lrange.index[x]] = 0;
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            std::vector<std::vector<LeftValue_> > left_vbuffers;
            std::vector<std::vector<LeftIndex_> > left_ibuffers;
            std::vector<std::vector<LeftValue_> > expanded;
            left_vbuffers.reserve(max_block_rows);
            left_ibuffers.reserve(max_block_rows);
            expanded.reserve(max_block_rows);
            for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
                left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(common_dim));
                expanded.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
            }
            auto left_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > >(max_block_rows);
            std::vector<LeftIndex_> left_non_empty;
            left_non_empty.reserve(max_block_rows);

            LeftIndex_ lr = 0;
            while (lr < length) {
                // We only consider the LHS rows with at least one structural non-zero.
                // Thus, our block consists of 'options.block_size' non-empty LHS rows, rather than fixed row-wise chunks of the LHS matrix.
                // This ensures that we don't waste iterations on LHS rows that will only have zeros in the output matrix (and are filled as such).
                left_non_empty.clear();
                LeftIndex_ lr_num = 0;
                bool all_non_empty = true;
                auto new_lr = lr;

                do {
                    auto lrange = ext->fetch(left_vbuffers[lr_num].data(), left_ibuffers[lr_num].data());
                    if (lrange.number == 0) {
                        std::fill_n(output + sanisizer::product_unsafe<std::size_t>(start + new_lr, right_NC), right_NC, 0);
                        all_non_empty = false;
                        ++new_lr;
                        continue;
                    }

                    left_ranges[lr_num] = std::move(lrange);
                    auto& curex = expanded[lr_num];
                    for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                        curex[lrange.index[x]] = lrange.value[x];
                    }
                    left_non_empty.push_back(new_lr);
                    ++lr_num;
                    ++new_lr;

                    if (sanisizer::is_equal(lr_num, options.block_size)) {
                        break;
                    }
                } while (new_lr < length);

                auto task = [&](auto converter) -> void {
                    // Deliberately iterating over the sparse RHS columns in the outer loop and the dense (expanded) LHS rows in the inner loop.
                    // This aims to keep the entirety of the dense LHS block in cache across multiple RHS columns, provided common_dim is small.
                    // If we did it the other way around, it would just be the same as the block_size == 1 case, but with more looping overhead. 
                    for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                        const auto rrange = right_ranges[rc];
                        if (rrange.number == 0) {
                            for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                                output[sanisizer::nd_offset<std::size_t>(rc, right_NC, converter(lr_counter))] = 0;
                            }
                            continue;
                        }

                        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            // Some false sharing potential here, but we just touch each location once per outer loop, so it's fine.
                            output[sanisizer::nd_offset<std::size_t>(rc, right_NC, converter(lr_counter))] = sparse_dot_product<accumulators_>(
                                rrange.number, // Implicit cast to size_t is safe, as per the tatami contract.
                                rrange.value,
                                rrange.index,
                                expanded[lr_counter].data(),
                                static_cast<Output_>(0)
                            );
                        }
                    }
                };

                if (all_non_empty) {
                    task([&](const LeftIndex_ lr_counter) -> LeftIndex_ { return start + lr + lr_counter; });
                } else {
                    for (auto& lrne : left_non_empty) {
                        lrne += start;
                    }
                    task([&](const LeftIndex_ lr_counter) -> LeftIndex_ { return left_non_empty[lr_counter]; });
                }

                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    const auto& lrange = left_ranges[lr_counter]; 
                    auto& curex = expanded[lr_counter];
                    for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                        curex[lrange.index[x]] = 0;
                    }
                }

                lr = new_lr;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
