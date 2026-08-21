#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"

/**
 * @file dense_column.hpp
 * @brief Dense column-major LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_column/multiple_vectors
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_column_with_multiple_vectors()`.
 */
struct MultiplyDenseColumnWithMultipleVectorsOptions {
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
 * @cond
 */
template<typename LeftValue_, typename LeftIndex_, typename RightVectors_, typename GetRightVector_, typename GetOutputVector_>
void multiply_dense_column_with_multiple_vectors_internal(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const LeftIndex_ start,
    const LeftIndex_ length,
    const LeftIndex_ left_NR,
    const RightVectors_ right_vectors,
    GetRightVector_ get_right_vector,
    GetOutputVector_ get_output_vector,
    const MultiplyDenseColumnWithMultipleVectorsOptions& options
) {
    auto ext = tatami::consecutive_extractor<false>(left, false, start, length);
    typedef I<decltype(get_output_vector(0)[0])> Output;

    if (options.primary_block_size == 1) {
        auto buffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
        for (LeftIndex_ cd = 0; cd < length; ++cd) {
            const auto ptr = ext->fetch(buffer.data());
            for (RightVectors_ rv = 0; rv < right_vectors; ++rv) {
                const auto optr = get_output_vector(rv);
                const Output mult = get_right_vector(rv)[start + cd];
                for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                    optr[lr] += mult * static_cast<Output>(ptr[lr]);
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
            const LeftIndex_ cd_num = sanisizer::min(options.primary_block_size, length - cd);
            for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                left_ptrs[cd_counter] = ext->fetch(left_buffers[cd_counter].data());
            }

            RightVectors_ rv = 0;
            while (rv < right_vectors) {
                const RightVectors_ rv_end = rv + sanisizer::min(options.primary_block_size, right_vectors - rv);
                LeftIndex_ lr = 0;
                while (lr < left_NR) {
                    const LeftIndex_ lr_end = lr + sanisizer::min(options.secondary_block_size, left_NR - lr);

                    for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                        const auto matcol = left_ptrs[cd_counter];
                        for (auto rv_copy = rv; rv_copy < rv_end; ++rv_copy) {
                            const Output mult = get_right_vector(rv_copy)[start + cd + cd_counter];
                            const auto outvec = get_output_vector(rv_copy);
                            for (auto lr_copy = lr; lr_copy < lr_end; ++lr_copy) {
                                outvec[lr_copy] += mult * static_cast<Output>(matcol[lr_copy]);
                            }
                        }
                    }

                    lr = lr_end;
                }
                rv = rv_end;
            }
            cd += cd_num;
        }
    }
}
/**
 * @endcond
 */

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightVectors_ Integer type of the number of RHS vectors.
 * @tparam GetRightVector_ Functor that accepts a `RightVectors_` and returns a pointer to a numeric (typically floating-point) array.
 * @tparam GetOutput_ Functor that accepts a `RightVectors_` and returns a pointer to a numeric (typically floating-point) array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param right_vectors Number of RHS vectors.
 * @param get_right_vector Function that accepts a `RightVectors_` in `[0, num_right)` and returns a pointer to an array of length `left.ncol()`.
 * The array referenced by `get_right_vector(i)` represents the `i`-th RHS vector with which to multiply `left`.
 * This function should be thread-safe.
 * @param get_output_vector Function that accepts a `RightVectors_` in `[0, num_right)` and returns a pointer to an array of length `left.nrow()`.
 * On output, the array referenced by `get_output_vector(i)` stores the product of `left` with the `i`-th RHS vector.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightVectors_, typename GetRightVector_, typename GetOutput_>
void multiply_dense_column_with_multiple_vectors(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightVectors_ right_vectors,
    GetRightVector_ get_right_vector,
    GetOutput_ get_output_vector,
    const MultiplyDenseColumnWithMultipleVectorsOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    for (RightVectors_ rv = 0; rv < right_vectors; ++rv) {
        std::fill_n(get_output_vector(rv), left_NR, 0);
    }

    const bool do_parallel = options.num_threads > 1;
    typedef I<decltype(get_output_vector(0)[0])> Output;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output> > > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        if (!do_parallel || t == 0) {
            multiply_dense_column_with_multiple_vectors_internal(
                left,
                start,
                length,
                left_NR,
                right_vectors,
                get_right_vector,
                get_output_vector,
                options
            );

        } else {
            std::vector<std::vector<Output> > tmp_output;
            tmp_output.reserve(right_vectors);
            for (RightVectors_ rv = 0; rv < right_vectors; ++rv) {
                tmp_output.emplace_back(tatami::cast_Index_to_container_size<std::vector<Output> >(left_NR));
            }
            multiply_dense_column_with_multiple_vectors_internal(
                left,
                start,
                length,
                left_NR,
                right_vectors,
                get_right_vector,
                [&](const RightVectors_ rv) -> Output* {
                    return tmp_output[rv].data();
                },
                options
            );
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (RightVectors_ rv = 0; rv < right_vectors; ++rv) {
                const auto& tmpvec = tmp[rv];
                const auto outptr = get_output_vector(rv);
                for (LeftIndex_ lr = 0; lr < left_NR; ++lr) {
                    outptr[lr] += tmpvec[lr];
                }
            }
        }
    }
}

/**
 * Overload of `multiply_dense_column_with_multiple_vectors()` that uses a vector of pointers to represent the RHS and output vectors.
 *
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS vectors.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer column access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains an RHS vector with which to multiply `left`.
 * @param[out] output Vector of length equal to `right.size()`.
 * Each entry is a pointer to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename Output_>
void multiply_dense_column_with_multiple_vectors(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const std::vector<RightValue_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyDenseColumnWithMultipleVectorsOptions& options
) {
    const auto right_vectors = right.size();
    typedef I<decltype(right_vectors)> RightVectors;
    multiply_dense_column_with_multiple_vectors(
        left,
        right_vectors,
        [&](const RightVectors rv) -> const RightValue_* {
            return right[rv];
        },
        [&](const RightVectors rv) -> Output_* {
            return output[rv];
        },
        options
    );
}

}

#endif
