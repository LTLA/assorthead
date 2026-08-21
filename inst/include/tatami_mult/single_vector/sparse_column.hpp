#ifndef TATAMI_MULT_SPARSE_COLUMN_HPP
#define TATAMI_MULT_SPARSE_COLUMN_HPP

#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"

/**
 * @file sparse_column.hpp
 * @brief Sparse column-major LHS, single vector RHS.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_column/single_vector
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_column_with_single_vector()`.
 */
struct MultiplySparseColumnWithSingleVectorOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS vector. 
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param[in] right Pointer to an array of length equal to the number of columns of `left`,
 * containing the RHS vector.
 * @param[out] output Pointer to an array of length equal to the number of rows of `left`.
 * On output, this stores the product `left * right`.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename Output_>
void multiply_sparse_column_with_single_vector(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightValue_* const right,
    Output_* const output,
    const MultiplySparseColumnWithSingleVectorOptions& options
) {
    const auto NR = left.nrow();
    const auto NC = left.ncol();

    const bool do_parallel = options.num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }
    std::fill_n(output, NR, 0);

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(NR);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(NR);

        Output_* optr;
        std::optional<std::vector<Output_> > cur_output;
        if (!do_parallel || t == 0) {
            optr = output;
        } else {
            cur_output.emplace(tatami::cast_Index_to_container_size<I<decltype(*cur_output)> >(NR));
            optr = cur_output->data();
        }

        for (LeftIndex_ c = 0; c < length; ++c) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());
            const Output_ mult = right[start + c];
            for (LeftIndex_ r = 0; r < range.number; ++r) {
                optr[range.index[r]] += mult * range.value[r];
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(cur_output);            
        }
    }, NC, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            for (LeftIndex_ r = 0; r < NR; ++r) {
                output[r] += tmp[r];
            }
        }
    }
}

}

#endif
