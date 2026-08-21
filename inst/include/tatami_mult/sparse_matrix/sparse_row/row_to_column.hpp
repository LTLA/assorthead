#ifndef TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_column.hpp
 * @brief Sparse row-major LHS, sparse row-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_sparse_row_matrix_to_column_output()`.
 */
struct MultiplySparseRowWithSparseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;
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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_sparse_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithSparseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(common_dim);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(common_dim);
    populate_sparse_buffers(true, common_dim, right_NC, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

        // Use a temporary buffer to (i) improve data locality and (ii) avoid false sharing.
        auto tmp_row = tatami::create_container_of_Index_size<std::vector<Output_> >(right_NC);

        for (LeftIndex_ lr = 0; lr < length; ++lr) {
            const auto lrange = ext->fetch(vbuffer.data(), ibuffer.data());

            for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                const Output_ mult = lrange.value[x];
                const auto rrange = right_ranges[lrange.index[x]];
                for (RightIndex_ y = 0; y < rrange.number; ++y) {
                    tmp_row[rrange.index[y]] += mult * static_cast<Output_>(rrange.value[y]);
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
}

}

#endif
