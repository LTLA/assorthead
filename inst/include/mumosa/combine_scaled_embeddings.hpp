#ifndef MUMOSA_COMBINE_SCALED_EMBEDDINGS_HPP
#define MUMOSA_COMBINE_SCALED_EMBEDDINGS_HPP

#include <vector>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file combine_scaled_embeddings.hpp
 * @brief Scale and combine embeddings.
 */

namespace mumosa {

/**
 * Scale the embedding for each modality and combine all embeddings from different modalities into a single matrix for further analyses.
 * Each cell in the combined matrix will contain a concatenation of the scaled coordinates from all of the individual embeddings.
 * 
 * @tparam Index_ Integer type of the number of cells.
 * @tparam Input_ Floating-point type of the input data.
 * @tparam Scale_ Floating-point type of the scaling factor.
 * @tparam Output_ Floating-point type of the output data.
 * 
 * @param num_dims Vector containing the number of dimensions in each embedding.
 * @param num_cells Number of cells in each embedding.
 * @param embeddings Vector of pointers of length equal to that of `num_dims`.
 * Each pointer refers to an array containing an embedding matrix for a single modality, which should be in column-major format with dimensions in rows and cells in columns.
 * The number of rows of the `i`-th matrix should be equal to `num_dims[i]` and the number of columns should be equal to `num_cells`.
 * @param scaling Scaling to apply to each embedding, usually from `compute_scale()`.
 * This should be of length equal to that of `num_dims`.
 * @param[out] output Pointer to the output array.
 * This should be of length equal to the product of `num_cells` and the sum of `num_dims`.
 * On completion, `output` is filled with the combined embeddings in column-major format.
 * Each row corresponds to a dimension while each column corresponds to a cell.
 */
template<typename Index_, typename Input_, typename Scale_, typename Output_>
void combine_scaled_embeddings(
    const std::vector<std::size_t>& num_dims,
    const Index_ num_cells,
    const std::vector<Input_*>& embeddings,
    const std::vector<Scale_>& scaling,
    Output_* const output
) {
    const auto nembed = num_dims.size();
    if (embeddings.size() != nembed || scaling.size() != nembed) {
        throw std::runtime_error("'num_dims', 'embeddings' and 'scale' should have the same length");
    }

    const std::size_t ntotal = std::accumulate(num_dims.begin(), num_dims.end(), static_cast<std::size_t>(0));
    std::size_t starting_dim = 0;

    for (I<decltype(nembed)> e = 0; e < nembed; ++e) {
        const auto curdim = num_dims[e];
        const auto inptr = embeddings[e];
        const auto s = scaling[e];

        if (std::isinf(s)) {
            // If the scaling factor is infinite, it implies that the current
            // embedding is all-zero, so we just fill with zeros, and move on.
            for (Index_ c = 0; c < num_cells; ++c) {
                const auto out_offset = sanisizer::nd_offset<std::size_t>(starting_dim, ntotal, c);
                std::fill_n(output + out_offset, curdim, 0);
            }
        } else {
            for (Index_ c = 0; c < num_cells; ++c) {
                for (I<decltype(curdim)> d = 0; d < curdim; ++d) {
                    const auto out_offset = sanisizer::nd_offset<std::size_t>(starting_dim + d, ntotal, c);
                    const auto in_offset = sanisizer::nd_offset<std::size_t>(d, curdim, c);
                    output[out_offset] = inptr[in_offset] * s;
                }
            }
        }

        starting_dim += curdim;
    }
}

}

#endif
