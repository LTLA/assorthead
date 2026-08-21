#ifndef TATAMI_MULT_DENSE_MATRIX_DENSE_COLUMN_COLUMN_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_DENSE_COLUMN_COLUMN_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file column_to_column.hpp
 * @brief Dense column-major LHS, dense column-major matrix RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_column/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_column_with_dense_column_matrix_to_column_output()`.
 */
struct MultiplyDenseColumnWithDenseColumnMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;

    /**
     * Primary block size, i.e., the number of LHS columns to be loaded at once.
     * This is also used to define the number of RHS columns in each block.
     * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size, i.e., the number of LHS rows to be processed in each block.
     * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     * Different secondary block sizes will not change the results.
     */
    int secondary_block_size = 64;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightColumn_ Functor that accepts a `RightColumns_` and returns a pointer to an RHS column.
 * @tparam Output_ Numeric type of the output array.
 *
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_column Function that accepts a `RightColumns_` in `[0, right_columns)` and returns a pointer to an array of length `left.ncol()`.
 * The array referenced by `get_right_column(i)` represents the `i`-th column of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this contains the matrix product in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, class GetRightColumn_, typename Output_>
void multiply_dense_column_with_dense_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightColumn_ get_right_column,
    Output_* const output,
    const MultiplyDenseColumnWithDenseColumnMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    // Product must fit in a size_t in order for output to have been allocated correctly in the first place.
    // Technically, right_columns could be larger than a size_t if left_NR == 0, but the product after wraparound would still be zero, so it's fine.
    std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_columns), 0);

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, false, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_columns));
            outptr = tmp_output->data();
        }

        if (options.primary_block_size == 1) {
            auto buffer = tatami::create_container_of_Index_size<std::vector<Output_> >(left_NR);
            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto ptr = ext->fetch(buffer.data());
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    const Output_ mult = get_right_column(rc)[start + cd];
                    for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                        outptr[sanisizer::nd_offset<std::size_t>(lr, left_NR, rc)] += mult * static_cast<Output_>(ptr[lr]);
                    }
                }
            }

        } else {
            std::vector<std::vector<LeftValue_> > left_buffers;
            std::vector<const LeftValue_*> left_ptrs;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.primary_block_size);
                left_buffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                }
                sanisizer::resize(left_ptrs, max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                const auto cd_num = sanisizer::min(options.primary_block_size, length - cd);
                for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                    left_ptrs[cd_counter] = ext->fetch(left_buffers[cd_counter].data());
                }

                RightColumns_ rc = 0;
                while (rc < right_columns) {
                    const RightColumns_ rc_end = rc + sanisizer::min(options.primary_block_size, right_columns - rc);
                    LeftIndex_ lr = 0;
                    while (lr < left_NR) {
                        const LeftIndex_ lr_end = lr + sanisizer::min(options.secondary_block_size, left_NR - lr);

                        for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto matcol = left_ptrs[cd_counter];
                            for (auto rc_copy = rc; rc_copy < rc_end; ++rc_copy) {
                                const Output_ mult = get_right_column(rc_copy)[start + cd + cd_counter];
                                for (auto lr_copy = lr; lr_copy < lr_end; ++lr_copy) {
                                    outptr[sanisizer::nd_offset<std::size_t>(lr_copy, left_NR, rc_copy)] += mult * static_cast<Output_>(matcol[lr_copy]);
                                }
                            }
                        }

                        lr = lr_end;
                    }
                    rc = rc_end;
                }
                cd += cd_num;
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
 * Overload of `multiply_dense_column_with_dense_column_matrix_to_column_output()` for a RHS `tatami::Matrix`.
 * This will iterate over `left`, realizing columns into memory as needed.
 * It will also realize all of `right` into memory for fast repeated accesses.
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
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * The number of rows in `right` should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this contains the product `left * right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_column_with_dense_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseColumnWithDenseColumnMatrixToColumnOutputOptions& options
) {
    const auto right_NC = right.ncol();
    auto right_buffers = tatami::create_container_of_Index_size<std::vector<std::vector<RightValue_> > >(right_NC);
    auto right_ptrs = tatami::create_container_of_Index_size<std::vector<const RightValue_*> >(right_NC);
    const auto common_dim = left.ncol();
    populate_dense_buffers(false, right_NC, common_dim, right, right_buffers, right_ptrs, options.num_threads);

    multiply_dense_column_with_dense_column_matrix_to_column_output(
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
