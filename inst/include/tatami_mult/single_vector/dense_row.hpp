#ifndef TATAMI_MULT_SINGLE_VECTOR_DENSE_ROW_HPP
#define TATAMI_MULT_SINGLE_VECTOR_DENSE_ROW_HPP

#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"

#include "../dense_dot_product.hpp"

/**
 * @file dense_row.hpp
 * @brief Dense row-major LHS, single vector RHS.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/dense_row/single_vector
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_dense_row_with_single_vector()`.
 */
struct MultiplyDenseRowWithSingleVectorOptions {
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
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<std::size_t accumulators_ = 4, typename LeftValue_, typename LeftIndex_, typename RightValue_, typename Output_>
void multiply_dense_row_with_single_vector(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightValue_* const right,
    Output_* const output,
    const MultiplyDenseRowWithSingleVectorOptions& options
) {
    const auto NR = left.nrow();
    const auto NC = left.ncol();
    tatami::parallelize([&](int, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<false>(left, true, start, length);
        auto buffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(NC);
        for (LeftIndex_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());
            output[r] = dense_dot_product<accumulators_>(
                NC, // tatami's contract guarantees that NC will fit in a std::size_t, so no need to protect the function call.
                ptr,
                right,
                static_cast<Output_>(0)
            );
        }
    }, NR, options.num_threads);
}

}

#endif
