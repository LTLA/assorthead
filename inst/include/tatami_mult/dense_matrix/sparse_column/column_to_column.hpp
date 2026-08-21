#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_COLUMN_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_COLUMN_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file column_to_column.hpp
 * @brief Sparse column-major LHS, dense column-major matrix RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_column/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_column_with_dense_column_matrix_to_column_output()`.
 */
struct MultiplySparseColumnWithDenseColumnMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;

    /**
     * Block size, i.e., number of LHS columns to be loaded at once.
     * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 16;
};




/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightColumn_ Functor that accepts a `RightColumns_` and returns a pointer to an RHS column.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_column Function that accepts a `RightColumns_` in `[0, right_columns)` and returns a pointer to an array of length `left.ncol()`.
 * The array referenced by `get_right_column(i)` represents the `i`-th column of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, this contains the product `left * right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightColumn_, typename Output_>
void multiply_sparse_column_with_dense_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightColumn_ get_right_column,
    Output_* const output,
    const MultiplySparseColumnWithDenseColumnMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    // Product must fit in a size_t in order for output to have been allocated correctly in the first place.
    // Technically, right_columns could be larger than a size_t if left_NR == 0, but the product after wraparound would still be zero, so it's fine.
    std::fill_n(output, sanisizer::product<std::size_t>(left_NR, right_columns), 0);

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, false, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_columns));
            outptr = tmp_output->data();
        }

        if (options.block_size == 1) {
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    const Output_ mult = get_right_column(rc)[start + cd];
                    for (LeftIndex_ x = 0; x < range.number; ++x) {
                        outptr[sanisizer::nd_offset<std::size_t>(range.index[x], left_NR, rc)] += mult * static_cast<Output_>(range.value[x]); 
                    }
                }
            }

        } else {
            std::vector<std::vector<LeftValue_> > left_vbuffers;
            std::vector<std::vector<LeftIndex_> > left_ibuffers;
            std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
            std::vector<LeftIndex_> left_non_empty;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_cols);
                left_ibuffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(left_NR));
                }
                sanisizer::resize(left_ranges, max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                // We only consider the LHS rows with at least one structural non-zero.
                // Thus, our block consists of 'options.block_size' non-empty LHS rows, rather than fixed row-wise chunks of the LHS matrix.
                // This ensures that we don't waste iterations on LHS rows that will only have zeros in the output matrix (and are filled as such).
                const auto left_block_info = fetch_non_empty_sparse_block(
                    *ext,
                    left_vbuffers,
                    left_ibuffers,
                    left_ranges,
                    left_non_empty,
                    cd,
                    length,
                    options.block_size,
                    /* zero = */ [&](const LeftIndex_) -> void {} // buffers should already be zeroed.
                );
                const auto cd_num = left_block_info.num_non_empty;

                // If the LHS rows are all non-empty, we can speed up the loops by just using a simple counter to get the column indices.
                // Otherwise, we'll have to access the 'left_non_empty' vector to figure out the indices of each non-empty column.
                if (left_block_info.all_non_empty) {
                    const LeftIndex_ cd_base = cd + start; 
                    for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                        const auto rightcol = get_right_column(rc);
                        for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto& currange = left_ranges[cd_counter];
                            const Output_ mult = rightcol[cd_base + cd_counter];
                            for (LeftIndex_ x = 0; x < currange.number; ++x) {
                                outptr[sanisizer::nd_offset<std::size_t>(currange.index[x], left_NR, rc)] += mult * static_cast<Output_>(currange.value[x]);
                            }
                        }
                    }

                } else {
                    for (auto& cdne : left_non_empty) {
                        cdne += start;
                    }
                    for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                        const auto rightcol = get_right_column(rc);
                        for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto& currange = left_ranges[cd_counter];
                            const Output_ mult = rightcol[left_non_empty[cd_counter]];
                            for (LeftIndex_ x = 0; x < currange.number; ++x) {
                                outptr[sanisizer::nd_offset<std::size_t>(currange.index[x], left_NR, rc)] += mult * static_cast<Output_>(currange.value[x]);
                            }
                        }
                    }
                }

                cd = left_block_info.position;
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
 * Overload of `multiply_sparse_column_with_dense_column_matrix_to_column_output()` for a RHS `tatami::Matrix`.
 * This function will iterate over `left`, realizing columns into memory as needed.
 * It will also realize all of `right` into memory for fast repeated accesses.
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
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * The number of rows in `right` should be equal to the number of columns in `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, this contains the product `left * right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_column_with_dense_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseColumnWithDenseColumnMatrixToColumnOutputOptions& options
) {
    const auto right_NC = right.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    const auto common_dim = left.ncol();
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    multiply_sparse_column_with_dense_column_matrix_to_column_output(
        left,
        right_NC,
        [&](const RightIndex_ rc) -> const RightValue_* {
            return right_ptrs[rc];
        },
        output,
        options
    );
}

}

#endif
