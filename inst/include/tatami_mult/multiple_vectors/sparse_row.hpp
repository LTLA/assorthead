#ifndef TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_HPP
#define TATAMI_MULT_MULTIPLE_VECTORS_SPARSE_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../utils.hpp"
#include "../sparse_dot_product.hpp"

/**
 * @file sparse_row.hpp
 * @brief Sparse row-major LHS, multiple vectors RHS.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/multiple_vectors
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_multiple_vectors()`.
 */
struct MultiplySparseRowWithMultipleVectorsOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;

    /**
     * Block size, i.e., the number of LHS rows to be loaded at once.
     * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 16;
};

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
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
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
void multiply_sparse_row_with_multiple_vectors(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightVectors_ right_vectors,
    GetRightVector_ get_right_vector,
    GetOutputVector_ get_output_vector,
    const MultiplySparseRowWithMultipleVectorsOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right_vectors; // using an alias just for consistent terminology.
    typedef I<decltype(get_output_vector(0)[0])> Output;;

    if (options.block_size == 1) {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(common_dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(common_dim);

            for (LeftIndex_ lr = 0; lr < length; ++lr) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                if (range.number == 0) {
                    for (RightVectors_ rv = 0; rv < right_NC; ++rv) {
                        get_output_vector(rv)[start + lr] = 0;
                    }
                    continue;
                }

                for (RightVectors_ rv = 0; rv < right_NC; ++rv) {
                    get_output_vector(rv)[start + lr] = sparse_dot_product<accumulators_>(
                        range.number, // Implicit cast to size_t is safe, as per the tatami contract.
                        range.value,
                        range.index,
                        get_right_vector(rv),
                        static_cast<Output>(0)
                    );
                }
            }
        }, left_NR, options.num_threads);

    } else {
        tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
            auto ext = tatami::consecutive_extractor<true>(left, true, start, length);

            std::vector<std::vector<LeftValue_> > left_vbuffers;
            std::vector<std::vector<LeftIndex_> > left_ibuffers;
            std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
            {
                const LeftIndex_ max_block_rows = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_rows);
                left_ibuffers.reserve(max_block_rows);
                for (LeftIndex_ lr = 0; lr < max_block_rows; ++lr) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(common_dim));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(common_dim));
                }
                sanisizer::resize(left_ranges, max_block_rows);
            }

            LeftIndex_ lr = 0;
            while (lr < length) {
                // No point skipping the LHS rows with no structural non-zeros.
                // We still need to set the corresponding entry of 'outvec' to zero, so we'd end up having to loop through the LHS rows anyway.
                // We might as well just let it be set to zero naturally in the existing loop below.
                const LeftIndex_ lr_num = sanisizer::min(options.block_size, length - lr);
                for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                    left_ranges[lr_counter] = ext->fetch(left_vbuffers[lr_counter].data(), left_ibuffers[lr_counter].data());
                }

                for (RightVectors_ rv = 0; rv < right_NC; ++rv) {
                    const auto rightvec = get_right_vector(rv);
                    const auto outvec = get_output_vector(rv);
                    for (LeftIndex_ lr_counter = 0; lr_counter < lr_num; ++lr_counter) {
                        const auto& currange = left_ranges[lr_counter];
                        outvec[start + lr + lr_counter] = sparse_dot_product<accumulators_>(
                            currange.number, // Implicit cast of range.number to size_t is safe, as per the tatami contract.
                            currange.value,
                            currange.index,
                            rightvec,
                            static_cast<Output>(0)
                        );
                    }
                }

                lr += lr_num;
            }
        }, left_NR, options.num_threads);
    }
}

/**
 * Overload of `multiply_sparse_row_with_multiple_vectors()` that uses a vector of pointers to represent the RHS and output vectors.
 *
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS vectors. 
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param[in] right Vector of pointers, each of which points to an array of length `left.ncol()`.
 * Each entry contains an RHS vector with which to multiply `left`.
 * @param[out] output Vector of length equal to `right.size()`.
 * Each entry is a pointer to an array of length `left.nrow()`.
 * On output, the `i`-th entry stores the product `left * right[i]`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename Output_>
void multiply_sparse_row_with_multiple_vectors(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const std::vector<RightValue_*>& right,
    const std::vector<Output_*>& output,
    const MultiplySparseRowWithMultipleVectorsOptions& options
) {
    const auto right_vectors = right.size();
    typedef I<decltype(right_vectors)> RightVectors;
    multiply_sparse_row_with_multiple_vectors(
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
