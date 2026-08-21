#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_ROW_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_column.hpp
 * @brief Dense row-major LHS, sparse row-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_sparse_row_matrix_to_column_output()`.
 */
struct MultiplyDenseRowWithSparseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., number of LHS rows to be loaded at once.
     * This is used to transpose the row-major LHS for more efficient operation with column-major output.
     *
     * The block size should be positive.
     * Larger values improve speed at the cost of increased memory usage.
     * If this is set to 1, no blocking is performed.
     */
    int block_size = 16;
};

/**
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
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_row_with_sparse_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseRowWithSparseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(common_dim);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(common_dim);
    populate_sparse_buffers(true, common_dim, right_NC, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    // We'll be skipping the empty RHS rows during iteration.
    auto right_non_empty = filter_non_empty_sparse(
        right_ranges,
        [&](RightIndex_) -> void {}
    );

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);

            // Use a temporary buffer to (i) improve data locality and (ii) avoid false sharing.
            auto tmp_row = tatami::create_container_of_Index_size<std::vector<Output_> >(right_NC);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = ext->fetch(dbuffer.data());

                auto loop_body = [&](LeftIndex_ cd) -> void {
                    const auto rrange = right_ranges[cd];
                    const Output_ mult = lptr[cd];
                    for (RightIndex_ x = 0; x < rrange.number; ++x) {
                        tmp_row[rrange.index[x]] += mult * static_cast<Output_>(rrange.value[x]);
                    }
                };

                if (right_non_empty.has_value()) {
                    for (const auto cd : *right_non_empty) {
                        loop_body(cd);
                    }
                } else {
                    for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                        loop_body(cd);
                    }
                }

                // Technically, we could limit the transposition to only those RHS columns that are not empty.
                // This avoids unnecessary accesses to non-contiguous memory that will always be zero.
                // To implement this, we'd need to do another pass through the RHS matrix to find the empty columns and store their IDs.
                // Such extra complexity is probably not worth it; we're not in the hot loop,
                // and accessing non-consecutive IDs may be a pessimization if it prevents compiler optimizations and/or strided prefetching.
                // Ensuring correct zeroing of the output columns also becomes a bit complicated if we skip empty RHS columns.
                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    output[sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc)] = tmp_row[rc];
                }
                std::fill(tmp_row.begin(), tmp_row.end(), 0);
            }
        }, left_NR, options.num_threads);

    } else {
        const bool do_parallel = options.num_threads > 1;
        if (!do_parallel) {
            std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);
        }

        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

            const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
            std::vector<std::vector<LeftValue_> > lbuffers;
            lbuffers.reserve(max_block_rows);
            for (LeftIndex_ b = 0; b < max_block_rows; ++b) {
                lbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
            }
            auto lptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);
            auto colbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(max_block_rows);

            // Create a temporary buffer to minimize false sharing during updates across all 'cd'.
            std::optional<std::vector<Output_> > tmp_cols;
            if (do_parallel) {
                tmp_cols.emplace(sanisizer::product<I<decltype(tmp_cols->size())> >(max_block_rows, right_NC));
            }

            LeftIndex_ lr = 0;
            while (lr < length) {
                const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    lptrs[lr_counter] = ext->fetch(lbuffers[lr_counter].data());
                }

                Output_* tmp_optr;
                LeftIndex_ out_row_offset;
                LeftIndex_ out_stride;
                if (do_parallel) {
                    tmp_optr = tmp_cols->data();
                    out_row_offset = 0;
                    out_stride = lr_num;
                } else {
                    tmp_optr = output;
                    out_row_offset = start + lr;
                    out_stride = left_NR;
                }

                auto loop_body = [&](LeftIndex_ cd) -> void {
                    const auto rrange = right_ranges[cd];

                    // Transfer this block of the 'cd'-th LHS column into a single buffer for a faster vector multiply-add in the inner loop.
                    // This rationale is unique to this function and is unlike any of the explanations in the sparse-blocking documentation.
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        colbuffer[lr_counter] = lptrs[lr_counter][cd];
                    }

                    for (RightIndex_ x = 0; x < rrange.number; ++x) {
                        const Output_ mult = rrange.value[x];
                        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                            tmp_optr[sanisizer::nd_offset<std::size_t>(out_row_offset + lr_counter, out_stride, rrange.index[x])] += mult * static_cast<Output_>(colbuffer[lr_counter]);
                        }
                    }
                };

                if (right_non_empty.has_value()) {
                    for (const auto cd : *right_non_empty) {
                        loop_body(cd);
                    }
                } else {
                    for (LeftIndex_ cd = 0; cd < common_dim; ++cd) {
                        loop_body(cd);
                    }
                }

                if (do_parallel) {
                    for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                        const auto src = tmp_cols->data() + sanisizer::product_unsafe<std::size_t>(rc, lr_num);
                        std::copy_n(src, lr_num, output + sanisizer::nd_offset<std::size_t>(start + lr, left_NR, rc));
                    }
                    std::fill_n(tmp_cols->data(), sanisizer::product_unsafe<std::size_t>(right_NC, lr_num), 0);
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

}

#endif
