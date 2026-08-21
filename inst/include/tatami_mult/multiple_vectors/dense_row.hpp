#ifndef TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_DENSE_ROW_HPP

#include <cstddef>
#include <vector>
#include <type_traits>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../dense_dot_product.hpp"

/**
 * @file dense_row.hpp
 * @brief Dense row-major LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/multiple_vectors
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_multiple_vectors()`.
 */
struct MultiplyDenseRowWithMultipleVectorsOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Primary block size, i.e., the number of LHS rows to be loaded at once.
     * This is also used to define the number of RHS columns in each block.
     * See the \f$B\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     */
    int primary_block_size = 16;

    /**
     * Secondary block size, i.e., the number of LHS columns to be processed in each block.
     * See the \f$C\f$ parameter in the @ref dense-blocking "Blocking for dense matrices" section for more details.
     * Different secondary block sizes may slightly change the results due to differences in floating-point round-off error.
     */
    int secondary_block_size = 64;
};

/**
 * @cond
 */
template<std::size_t accumulators_, bool use_local_buffer_, typename LeftValue_, typename LeftIndex_, typename RightVectors_, typename GetRightVector_, typename GetOutputVector_>
void multiply_dense_row_with_multiple_vectors_blocked_internal(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const LeftIndex_ start,
    const LeftIndex_ length,
    const LeftIndex_ common_dim,
    const RightVectors_ right_vectors,
    GetRightVector_ get_right_vector,
    GetOutputVector_ get_output_vector,
    const MultiplyDenseRowWithMultipleVectorsOptions& options
) {
    auto ext = tatami::consecutive_extractor<false>(left, true, start, length);

    const LeftIndex_ max_block_rows = sanisizer::min(length, options.primary_block_size);
    std::vector<std::vector<LeftValue_> > left_buffers;
    left_buffers.reserve(max_block_rows);
    for (LeftIndex_ lr = 0; lr < max_block_rows; ++lr) {
        left_buffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
    }
    auto left_ptrs = tatami::create_container_of_Index_size<std::vector<const LeftValue_*> >(max_block_rows);

    typedef I<decltype(get_output_vector(0)[0])> Output;
    typename std::conditional<use_local_buffer_, std::vector<std::vector<Output> >, bool>::type tmp_output;
    if constexpr(!use_local_buffer_) {
        // Zeroing all of the buffers if we're operating on a single thread,
        // as we're computing partial dot products and we need to start from zero.
        for (RightVectors_ rc = 0; rc < right_vectors; ++rc) {
            std::fill_n(get_output_vector(rc), length, 0);
        }
    } else {
        // For the multi-threaded case, we create some temporary buffers to hold the partial dot products for the current set of submatrices.
        // This aims to mitigate false sharing as we update each block's partial dot products in the loop over the common dimension.
        // There is still some potential for false sharing when we transfer the results to the output buffers,
        // but this is the same as the unblocked case so we won't worry about it.
        const RightVectors_ max_block_cols = sanisizer::min(right_vectors, options.primary_block_size);
        tmp_output.reserve(max_block_cols);
        for (RightVectors_ rc = 0; rc < max_block_cols; ++rc) {
            tmp_output.emplace_back(tatami::cast_Index_to_container_size<std::vector<Output> >(max_block_rows));
        }
    }

    LeftIndex_ lr = 0;
    while (lr < length) {
        const LeftIndex_ lr_num = sanisizer::min(options.primary_block_size, length - lr);
        for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
            left_ptrs[lr_counter] = ext->fetch(left_buffers[lr_counter].data());
        }

        RightVectors_ rc = 0;
        while (rc < right_vectors) {
            const RightVectors_ rc_num = sanisizer::min(options.primary_block_size, right_vectors - rc);

            LeftIndex_ cd = 0;
            while (cd < common_dim) {
                const LeftIndex_ cd_num = sanisizer::min(options.secondary_block_size, common_dim - cd);
                for (RightVectors_ rc_counter = 0; rc_counter < rc_num; ++rc_counter) {
                    const auto outvec = [&](){
                        if constexpr(!use_local_buffer_) {
                            return get_output_vector(rc + rc_counter) + start + lr;
                        } else {
                            return tmp_output[rc_counter].data();
                        }
                    }();
                    const auto rightvec = get_right_vector(rc + rc_counter);

                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        auto& dest = outvec[lr_counter]; 
                        dest = dense_dot_product<accumulators_>(
                            cd_num, // Implicit cast to std::size_t is safe, as per the tatami contract.
                            rightvec + cd,
                            left_ptrs[lr_counter] + cd,
                            dest
                        );
                    }
                }
                cd += cd_num;
            }

            if constexpr(use_local_buffer_) {
                for (RightVectors_ rc_counter = 0; rc_counter < rc_num; ++rc_counter) {
                    auto& src = tmp_output[rc_counter];
                    std::copy_n(src.begin(), lr_num, get_output_vector(rc + rc_counter) + start + lr);
                    std::fill_n(src.begin(), lr_num, 0);
                }
            }

            rc += rc_num;
        }
        lr += lr_num;
    }
}
/**
 * @endcond
 */

/**
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightVectors_ Integer type of the number of RHS vectors.
 * @tparam GetRightVector_ Functor that accepts a `RightVectors_` and returns a pointer to a numeric (typically floating-point) array.
 * @tparam GetOutputVector_ Functor that accepts a `RightVectors_` and returns a pointer to a numeric (typically floating-point) array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param right_vectors Number of RHS vectors.
 * @param get_right_vector Function that accepts a `RightVectors_` in `[0, right_vectors)` and returns a pointer to an array of length `left.ncol()`.
 * The array referenced by `get_right_vector(i)` represents the `i`-th RHS vector with which to multiply `left`.
 * This function should be thread-safe.
 * @param get_output_vector Function that accepts a `RightVectors_` in `[0, right_vectors)` and returns a pointer to an array of length `left.nrow()`.
 * On output, the array referenced by by `get_output_vector(i)` stores the product `left * right[i]`.
 * This function should be thread-safe.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightVectors_, typename GetRightVector_, typename GetOutputVector_>
void multiply_dense_row_with_multiple_vectors(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightVectors_ right_vectors,
    GetRightVector_ get_right_vector,
    GetOutputVector_ get_output_vector,
    const MultiplyDenseRowWithMultipleVectorsOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    typedef I<decltype(get_output_vector(0)[0])> Output;

    if (options.primary_block_size == 1) {
        tatami::parallelize([&](int, const LeftIndex_ start, const LeftIndex_ length) -> void {
            auto lext = tatami::consecutive_extractor<false>(left, true, start, length);
            auto lbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto lptr = lext->fetch(lbuffer.data());
                for (RightVectors_ rv = 0; rv < right_vectors; ++rv) {
                    get_output_vector(rv)[start + lr] = dense_dot_product<accumulators_>(
                        common_dim, // Implicit cast to std::size_t is safe, as per the tatami contract.
                        lptr,
                        get_right_vector(rv),
                        static_cast<Output>(0)
                    );
                }
            }
        }, left_NR, options.num_threads);
        return;
    } 

    const bool do_parallel = options.num_threads > 1;
    tatami::parallelize([&](int, const LeftIndex_ start, const LeftIndex_ length) -> void {
        if (!do_parallel) {
            multiply_dense_row_with_multiple_vectors_blocked_internal<accumulators_, false>(left, start, length, common_dim, right_vectors, get_right_vector, get_output_vector, options);
        } else {
            multiply_dense_row_with_multiple_vectors_blocked_internal<accumulators_, true>(left, start, length, common_dim, right_vectors, get_right_vector, get_output_vector, options);
        }
    }, left_NR, options.num_threads);
}

/**
 * Overload of `multiply_dense_row_with_multiple_vectors()` that uses a vector of pointers to represent the RHS and output vectors.
 *
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS vectors. 
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains an RHS vector with which to multiply `left`.
 * @param[out] output Vector of length equal to `right.size()`.
 * Each entry is a pointer to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename Output_>
void multiply_dense_row_with_multiple_vectors(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const std::vector<RightValue_*>& right,
    const std::vector<Output_*>& output,
    const MultiplyDenseRowWithMultipleVectorsOptions& options
) {
    const auto right_vectors = right.size();
    typedef I<decltype(right_vectors)> RightVectors;
    multiply_dense_row_with_multiple_vectors(
        left,
        right_vectors,
        [&](const RightVectors rc) -> const RightValue_* {
            return right[rc];
        },
        [&](const RightVectors rc) -> Output_* {
            return output[rc];
        },
        options
    );
}

}

#endif
