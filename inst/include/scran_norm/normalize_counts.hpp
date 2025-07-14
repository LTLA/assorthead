#ifndef SCRAN_NORM_NORMALIZE_COUNTS_HPP
#define SCRAN_NORM_NORMALIZE_COUNTS_HPP

#include <type_traits>
#include <vector>
#include <memory>

#include "tatami/tatami.hpp"

/**
 * @file normalize_counts.hpp
 * @brief Normalize and log-transform counts.
 */

namespace scran_norm {

/**
 * @brief Options for `normalize_counts()`.
 */
struct NormalizeCountsOptions {
    /**
     * Pseudo-count to add to each value prior to log-transformation.
     * All values should be positive to ensure that log-transformed values are finite.
     * The default value of 1 preserves sparsity in the log-count matrix.
     * Larger values shrink the differences between cells towards zero, reducing variance at the cost of increasing bias.
     * Ignored if `NormalizeCountsOptions::log = false`.
     */
    double pseudo_count = 1;

    /**
     * Whether to preserve sparsity for non-unity pseudo-counts.
     * If true, we multiply the size factors by the `pseudo_count` and add 1 before log-transformation.
     * This does not change the differences between entries of the resulting matrix,
     * and adding `log(pseudo_count)` will recover the expected log-count values.
     * Ignored if `NormalizeCountsOptions::log = false`.
     */
    bool preserve_sparsity = false;

    /**
     * Whether to log-transform the normalized counts in the output matrix.
     */
    bool log = true;

    /**
     * Base for the log-transformation.
     * Only used if `NormalizeCountsOptions::log = true`.
     */
    double log_base = 2;
};

/**
 * Given a count matrix and a set of size factors, compute log-transformed normalized expression values.
 * All operations are done in a delayed manner using the `tatami::DelayedUnaryIsometricOperation` class.
 *
 * For normalization, each cell's counts are divided by the cell's size factor to remove uninteresting scaling differences.
 * The simplest and most common method for defining size factors is to use the centered library sizes, see `center_size_factors()` for details.
 * This removes scaling biases caused by sequencing depth, capture efficiency etc. between cells,
 * while the centering preserves the scale of the counts in the normalized expression values.
 * That said, users can define size factors from any method of their choice (e.g., median-based normalization, TMM) as long as they are positive for all cells.
 *
 * Normalized values are then log-transformed so that differences in log-values represent log-fold changes in expression.
 * This ensures that downstream analyses like t-tests and distance calculations focus on relative fold-changes rather than absolute differences.
 * The log-transformation also provides some measure of variance stabilization so that the downstream analyses are not dominated by sampling noise at large counts.
 *
 * @tparam OutputValue_ Floating-point type for the output matrix.
 * @tparam InputValue_ Data type for the input matrix.
 * @tparam InputIndex_ Integer type for the input matrix.
 * @tparam SizeFactors_ Container of floats for the size factors.
 * This should have the `size()`, `begin()`, `end()` and `operator[]` methods.
 *
 * @param counts Pointer to a `tatami::Matrix` containing counts.
 * Rows should correspond to genes while columns should correspond to cells.
 * @param size_factors Vector of length equal to the number of columns in `counts`, containing the size factor for each cell.
 * All values should be positive. 
 * @param options Further options.
 *
 * @return Matrix of normalized expression values.
 * These are log-transformed if `NormalizeCountsOptions::log = true`.
 */
template<typename OutputValue_ = double, typename InputValue_, typename Index_, class SizeFactors_>
std::shared_ptr<tatami::Matrix<OutputValue_, Index_> > normalize_counts(
    std::shared_ptr<const tatami::Matrix<InputValue_, Index_> > counts, 
    SizeFactors_ size_factors, 
    const NormalizeCountsOptions& options) 
{
    auto current_pseudo = options.pseudo_count;
    if (options.preserve_sparsity && current_pseudo != 1 && options.log) {
        for (auto& x : size_factors) { 
            x *= current_pseudo;
        }
        current_pseudo = 1;
    }

    static_assert(std::is_floating_point<OutputValue_>::value);
    if (static_cast<size_t>(size_factors.size()) != static_cast<size_t>(counts->ncol())) {
        throw std::runtime_error("length of 'size_factors' should be equal to the number of columns of 'counts'");
    }

    auto div = std::make_shared<tatami::DelayedUnaryIsometricOperation<OutputValue_, InputValue_, Index_> >(
        std::move(counts), 
        std::make_shared<tatami::DelayedUnaryIsometricDivideVectorHelper<true, OutputValue_, InputValue_, Index_, SizeFactors_> >(std::move(size_factors), false)
    );

    if (!options.log) {
        return div;
    }

    if (current_pseudo == 1) {
        return std::make_shared<tatami::DelayedUnaryIsometricOperation<OutputValue_, OutputValue_, Index_> >(
            std::move(div), 
            std::make_shared<tatami::DelayedUnaryIsometricLog1pHelper<OutputValue_, OutputValue_, Index_, OutputValue_> >(options.log_base)
        );
    } else {
        auto add = std::make_shared<tatami::DelayedUnaryIsometricOperation<OutputValue_, OutputValue_, Index_> >(
            std::move(div), 
            std::make_shared<tatami::DelayedUnaryIsometricAddScalarHelper<OutputValue_, OutputValue_, Index_, OutputValue_> >(current_pseudo)
        );
        return std::make_shared<tatami::DelayedUnaryIsometricOperation<OutputValue_, OutputValue_, Index_> >(
            std::move(add), 
            std::make_shared<tatami::DelayedUnaryIsometricLogHelper<OutputValue_, OutputValue_, Index_, OutputValue_> >(options.log_base)
        );
    }
};

/**
 * @cond
 */
// Overload for template deduction.
template<typename OutputValue_ = double, typename InputValue_, typename Index_, class SizeFactors_>
std::shared_ptr<tatami::Matrix<OutputValue_, Index_> > normalize_counts(
    std::shared_ptr<tatami::Matrix<InputValue_, Index_> > counts,
    SizeFactors_ size_factors,
    const NormalizeCountsOptions& options)
{
    return normalize_counts(std::shared_ptr<const tatami::Matrix<InputValue_, Index_> >(std::move(counts)), std::move(size_factors), options);
}
/**
 * @endcond
 */

}

#endif
