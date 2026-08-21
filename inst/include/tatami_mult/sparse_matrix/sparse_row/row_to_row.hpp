#ifndef TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_ROW_TO_ROW_HPP
#define TATAMI_MULT_SPARSE_MATRIX_SPARSE_ROW_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Sparse row-major LHS, sparse row-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_sparse_row_matrix_to_row_output()`.
 */
struct MultiplySparseRowWithSparseRowMatrixToRowOutputOptions {
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
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_row_with_sparse_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseRowWithSparseRowMatrixToRowOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    auto right_vbuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(common_dim);
    auto right_ibuffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightIndex_> > >(common_dim);
    auto right_ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<RightValue_, RightIndex_> > >(common_dim);
    populate_sparse_buffers(true, common_dim, right_NC, right, right_vbuffers, right_ibuffers, right_ranges, options.num_threads);

    const bool do_parallel = options.num_threads > 1;
    if (!do_parallel) {
        std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);
    }

    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

        std::optional<std::vector<Output_> > tmp_row;
        if (do_parallel) {
            tmp_row.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(right_NC));
        }

        for (LeftIndex_ lr = 0; lr < length; ++lr) {
            const auto lrange = ext->fetch(vbuffer.data(), ibuffer.data());
            const auto optr = output + sanisizer::product_unsafe<std::size_t>(start + lr, right_NC);
            const auto tmp_optr = (do_parallel ? tmp_row->data() : optr);

            for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                const auto rrange = right_ranges[lrange.index[x]];
                const Output_ mult = lrange.value[x];
                for (RightIndex_ y = 0; y < rrange.number; ++y) {
                    tmp_optr[rrange.index[y]] += mult * static_cast<Output_>(rrange.value[y]);
                }
            };

            if (do_parallel) {
                std::copy_n(tmp_optr, right_NC, optr);

                // Technically, we only have to reset the positions at which there is at least one non-zero across all RHS rows.
                // However, the union of all non-zero positions across all RHS rows is probably quite dense.
                // It'll likely be faster to just zero the entire buffer rather than trying to zero specific positions;
                // for example, one 64-byte cache line contains 8 doubles, so you'd need a density below ~10% to even avoid loading every cache line.
                // And that's not even considering further optimizations in the memset call.
                std::fill_n(tmp_optr, right_NC, 0);
            }
        }
    }, left_NR, options.num_threads);
}

}

#endif
