#ifndef SCRAN_BLOCKS_BLOCK_WEIGHTS_HPP
#define SCRAN_BLOCKS_BLOCK_WEIGHTS_HPP

#include <vector>
#include <cstddef>
#include <algorithm>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file block_weights.hpp
 * @brief Calculation of per-block weights.
 */

namespace scran_blocks {

/**
 * Policy for weighting blocks based on their size, i.e., the number of cells in each block.
 * This determines the nature of the weight calculations in `compute_weights()`.
 *
 * - `SIZE`: blocks are weighted in proportion to their size.
 *   Larger blocks will contribute more to the weighted average. 
 * - `EQUAL`: each non-empty block is assigned equal weight, regardless of its size.
 *   Equivalent to averaging across non-empty blocks without weights.
 * - `VARIABLE`: each batch is weighted using the logic in `compute_variable_weight()`.
 *   This penalizes small blocks with unreliable statistics while equally weighting all large blocks.
 * - `NONE`: a deprecated alias for `SIZE`.
 */
enum class WeightPolicy : char { NONE, SIZE, VARIABLE, EQUAL };

/**
 * @brief Parameters for `compute_variable_weight()`.
 */
struct VariableWeightParameters {
    /**
     * Lower bound for the block weight calculation.
     * This should be non-negative.
     */
    double lower_bound = 0;

    /**
     * Upper bound for the block weight calculation.
     * This should be no less than `lower_bound`.
     */
    double upper_bound = 1000;
};

/**
 * Assign a variable weight to each block of cells, for use in computing a weighted average across blocks.
 * The weight for each block is calculated from the size of that block.
 *
 * - If the block size is less than `VariableWeightParameters::lower_bound`, it has zero weight.
 * - If the block size is greater than `VariableWeightParameters::upper_bound`, it has weight of 1.
 * - Otherwise, the block has weight proportional to its size, increasing linearly from 0 to 1 between the two bounds.
 *
 * Blocks that are "large enough" (i.e., above the upper bound) are considered to be equally trustworthy and receive the same weight,
 * ensuring that each block contributes equally to the weighted average.
 * By comparison, very small blocks receive lower weight as their statistics are generally less stable.
 *
 * @param s Size of the block, in terms of the number of cells in that block.
 * @param params Parameters for the weight calculation, consisting of the lower and upper bounds.
 *
 * @return Weight of the block, to use for computing a weighted average across blocks. 
 */
inline double compute_variable_weight(const double s, const VariableWeightParameters& params) {
    if (s < params.lower_bound || s == 0) {
        return 0;
    }

    if (s > params.upper_bound) {
        return 1;
    }

    return (s - params.lower_bound) / (params.upper_bound - params.lower_bound);
}

/**
 * Compute weights for multiple blocks based on their size and the weighting policy.
 * For variable weights, this function will call `compute_variable_weight()` for each block.
 *
 * Weights should be interpreted as relative values within a single `compute_weights()` call, i.e., weights from different calls may not be comparable.
 * They are typically used in functions like `parallel_weighted_means()` to compute a weighted average of statistics across blocks.
 *
 * @tparam Size_ Numeric type of the block size.
 * @tparam Weight_ Floating-point type of the output weights.
 *
 * @param num_blocks Number of blocks.
 * @param[in] sizes Pointer to an array of length `num_blocks`, containing the size of each block.
 * @param policy Policy for weighting blocks of different sizes.
 * @param variable Parameters for the variable block weights.
 * @param[out] weights Pointer to an array of length `num_blocks`.
 * On output, this is filled with the weight of each block.
 */
template<typename Size_, typename Weight_>
void compute_weights(const std::size_t num_blocks, const Size_* const sizes, const WeightPolicy policy, const VariableWeightParameters& variable, Weight_* const weights) {
    if (policy == WeightPolicy::NONE || policy == WeightPolicy::SIZE) {
        std::copy_n(sizes, num_blocks, weights);
    } else if (policy == WeightPolicy::EQUAL) {
        for (I<decltype(num_blocks)> s = 0; s < num_blocks; ++s) {
            weights[s] = sizes[s] > 0;
        }
    } else {
        for (I<decltype(num_blocks)> s = 0; s < num_blocks; ++s) {
            weights[s] = compute_variable_weight(sizes[s], variable);
        }
    }
}

/**
 * A convenience overload for `compute_weights()` that accepts and returns vectors. 
 *
 * @tparam Size_ Numeric type of the block size.
 * @tparam Weight_ Floating-point type of the output weights.
 *
 * @param sizes Vector containing the size of each block.
 * @param policy Policy for weighting blocks of different sizes.
 * @param variable Parameters for the variable block weights.
 *
 * @return Vector of block weights.
 */
template<typename Weight_ = double, typename Size_>
std::vector<Weight_> compute_weights(const std::vector<Size_>& sizes, const WeightPolicy policy, const VariableWeightParameters& variable) {
    auto output = sanisizer::create<std::vector<Weight_> >(sizes.size());
    compute_weights(sizes.size(), sizes.data(), policy, variable, output.data());
    return output;
}

}

#endif
