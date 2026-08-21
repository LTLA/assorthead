#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_COLUMN_ROW_TO_ROW_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_COLUMN_ROW_TO_ROW_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file row_to_row.hpp
 * @brief Dense row-major LHS, sparse row-major RHS, row-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_column/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_column_with_sparse_row_matrix_to_row_output()`.
 */
struct MultiplyDenseColumnWithSparseRowMatrixToRowOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS columns to be loaded at once.
     * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 16;
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
 * On output, this stores the product of `left` and `right` in row-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_column_with_sparse_row_matrix_to_row_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseColumnWithSparseRowMatrixToRowOutputOptions& options
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
        auto left_ext = tatami::consecutive_extractor<false>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<true>(right, true, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        if (options.block_size == 1) {
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(right_NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<RightIndex_> >(right_NC);

            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto lptr = left_ext->fetch(dbuffer.data());
                const auto rrange = right_ext->fetch(vbuffer.data(), ibuffer.data());

                // Make sure this skip is done after all fetch() calls, otherwise the extractors will not be in sync with the common dimension.
                if (rrange.number == 0) {
                    continue;
                }

                for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                    const Output_ mult = lptr[lr];
                    for (RightIndex_ x = 0; x < rrange.number; ++x) {
                        outptr[sanisizer::nd_offset<std::size_t>(rrange.index[x], right_NC, lr)] += mult * static_cast<Output_>(rrange.value[x]);
                    }
                }
            }

        } else {
            std::vector<std::vector<LeftValue_> > left_dbuffers;
            std::vector<const LeftValue_*> left_ptrs;
            std::vector<std::vector<RightValue_> > right_vbuffers;
            std::vector<std::vector<RightIndex_> > right_ibuffers;
            std::vector<tatami::SparseRange<RightValue_, RightIndex_> > right_ranges;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.block_size);
                left_dbuffers.reserve(max_block_cols);
                right_vbuffers.reserve(max_block_cols);
                right_ibuffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_dbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                    right_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<RightValue_> >(right_NC));
                    right_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<RightIndex_> >(right_NC));
                }
                sanisizer::resize(left_ptrs, max_block_cols);
                sanisizer::resize(right_ranges, max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                // Only processing LHS columns if the corresponding RHS row has some structural non-zeros.
                // If not, we just skip it altogether; no need to zero or do anything else, as we're skipping the corresponding RHS row too.
                LeftIndex_ cd_num = 0;
                do {
                    auto lptr = left_ext->fetch(left_dbuffers[cd_num].data());
                    auto rrange = right_ext->fetch(right_vbuffers[cd_num].data(), right_ibuffers[cd_num].data());

                    // Again, this skip must be done after the LHS row is fetched, otherwise the extractors will be out of sync.
                    if (rrange.number == 0) {
                        ++cd;
                        continue;
                    }

                    left_ptrs[cd_num] = lptr;
                    right_ranges[cd_num] = std::move(rrange);
                    ++cd_num;
                    ++cd;

                    if (sanisizer::is_equal(cd_num, options.block_size)) {
                        break;
                    }
                } while (cd < length);

                for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                    for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                        const auto mult = left_ptrs[cd_counter][lr];
                        const auto rrange = right_ranges[cd_counter];
                        for (RightIndex_ x = 0; x < rrange.number; ++x) {
                            outptr[sanisizer::nd_offset<std::size_t>(rrange.index[x], right_NC, lr)] += mult * static_cast<Output_>(rrange.value[x]);
                        }
                    }
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
