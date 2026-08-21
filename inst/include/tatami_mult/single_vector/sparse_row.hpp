#ifndef TATAMI_MULT_SINGLE_VECTOR_SPARSE_ROW_HPP
#define TATAMI_MULT_SINGLE_VECTOR_SPARSE_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../sparse_dot_product.hpp"

/**
 * @file sparse_row.hpp
 * @brief Sparse row-major LHS, single vector RHS.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_row/single_vector
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_row_with_single_vector()`.
 */
struct MultiplySparseRowWithSingleVectorOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads will not change the results. 
     */
    int num_threads = 1;
};

/**
 * @tparam accumulators_ Number of accumulators for computing the dot product,
 * see the @ref multiple-accumulators "Multiple accumulators" section for more details.
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS vector. 
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer row access, but will work with all matrices.
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename Output_>
void multiply_sparse_row_with_single_vector(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightValue_* const right,
    Output_* const output,
    const MultiplySparseRowWithSingleVectorOptions& options
) {
    const auto NR = left.nrow();
    const auto NC = left.ncol();
    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, true, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(NC);
        for (LeftIndex_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            output[r] = sparse_dot_product<accumulators_>(
                range.number, // tatami guarantees that range.number will fit in a std::size_t, so no need to protect the function call.
                range.value,
                range.index,
                right,
                static_cast<Output_>(0)
            );
        }
    }, NR, options.num_threads);
}

}

#endif
