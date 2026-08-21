#ifndef TATAMI_MULT_SPARSE_MATRIX_DENSE_COLUMN_COLUMN_TO_COLUMN_HPP
#define TATAMI_MULT_SPARSE_MATRIX_DENSE_COLUMN_COLUMN_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"
#include "../../utils.hpp"

/**
 * @file column_to_column.hpp
 * @brief Dense row-major LHS, sparse column-major RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_column/sparse_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_column_with_sparse_column_matrix_to_column_output()`.
 */
struct MultiplyDenseColumnWithSparseColumnMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;
};

/**
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
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * The number of rows in this matrix should be equal to the number of columns in `left`.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right.ncol()`.
 * On output, this stores the product of `left` and `right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_dense_column_with_sparse_column_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplyDenseColumnWithSparseColumnMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();

    tatami::RetrieveFragmentedSparseContentsOptions conv_opt;
    conv_opt.two_pass = false;
    conv_opt.num_threads = options.num_threads;
    auto rhs_data = tatami::retrieve_fragmented_sparse_contents<RightValue_, RightIndex_>(right, true, conv_opt);

    // If there are any empty RHS rows, we only iterate over the non-empty ones in the outer loop.
    auto right_non_empty = filter_non_empty_sparse(
        rhs_data.index,
        [&](const RightIndex_) -> void {}
    );

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    std::fill_n(output, sanisizer::product_unsafe<std::size_t>(left_NR, right_NC), 0);

    // If we have empty RHS rows, we completely skip the corresponding LHS columns. 
    // Otherwise doing the easier approach of just looping with a counter.
    const LeftIndex_ cd_total = (right_non_empty.has_value() ? static_cast<LeftIndex_>(right_non_empty->size()) : common_dim);
    const int num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        auto dbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);

        auto task = [&](std::unique_ptr<tatami::OracularDenseExtractor<LeftValue_, LeftIndex_> >& ext, auto converter) -> void {
            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto lptr = ext->fetch(dbuffer.data());
                const auto actual_cd = converter(cd);
                const auto& right_values = rhs_data.value[actual_cd];
                const auto& right_indices = rhs_data.index[actual_cd];
                const RightIndex_ right_nnz = right_values.size();
                for (RightIndex_ x = 0; x < right_nnz; ++x) {
                    const Output_ mult = right_values[x];
                    const auto idx = right_indices[x];
                    for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                        outptr[sanisizer::nd_offset<std::size_t>(lr, left_NR, idx)] += mult * static_cast<Output_>(lptr[lr]);
                    }
                }
            }
        };

        if (right_non_empty.has_value()) {
            auto ext = tatami::new_extractor<false, true>(left, false, std::make_shared<tatami::FixedViewOracle<LeftIndex_> >(right_non_empty->data() + start, length));
            task(
                ext,
                [&](const LeftIndex_ cd) -> LeftIndex_ { 
                    return (*right_non_empty)[start + cd];
                }
            );
        } else {
            auto ext = tatami::consecutive_extractor<false>(left, false, start, length);
            task(
                ext,
                [&](const LeftIndex_ cd) -> LeftIndex_ {
                    return cd + start;
                }
            );
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, cd_total, options.num_threads);

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
