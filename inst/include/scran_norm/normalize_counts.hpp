#ifndef SCRAN_NORM_NORMALIZE_COUNTS_HPP
#define SCRAN_NORM_NORMALIZE_COUNTS_HPP

#include <type_traits>
#include <vector>
#include <memory>
#include <cassert>

#include "tatami/tatami.hpp"

/**
 * @file normalize_counts.hpp
 * @brief Normalize and log-transform counts.
 */

namespace scran_norm {

/**
 * @brief Helper for delayed log-normalization.
 *
 * This is a helper subclass that performs scaling normalization and log-transformation for each cell in `tatami::DelayedUnaryIsometricOperation`.
 * It should be constructed with `normalize_counts()` and is equivalent to chaining `tatami::DelayedUnaryIsometricDivideVectorHelper` (for the scaling by size factors),
 * with `tatami::DelayedUnaryIsometricLog1pHelper` and `tatami::DelayedUnaryIsometricDivideScalarHelper` (for log-transformation with a pseudo-count of 1 and an arbitrary base)
 * or `tatami::DelayedUnaryIsometricAddScalarHelper` and `tatami::DelayedUnaryIsometricCustomLogHelper` (for log-transformation with a non-unity pseudo-count).
 *
 * @tparam OutputValue_ Type of the result of the operation.
 * @tparam InputValue_ Type of the matrix value used in the operation.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam SizeFactors_ Container for the size factors.
 * This should support `[]`, `size()`, `begin()` and `end()`.
 */
template<typename OutputValue_, typename InputValue_, typename Index_, typename SizeFactors_>
class DelayedLogNormalizeHelper final : public tatami::DelayedUnaryIsometricOperationHelper<OutputValue_, InputValue_, Index_> {
public:
    /**
     * @param vector Vector of size factors, of length equal to the number of columns.
     * @param log_base Base of the log, usually 2.
     * @param pseudo_count Pseudo-count to add before log-transformation, usually 1.
     */
    DelayedLogNormalizeHelper(SizeFactors_ size_factors, OutputValue_ log_base, OutputValue_ pseudo_count) : 
        my_sf(std::move(size_factors)),
        my_base(std::log(log_base)),
        my_pseudo(pseudo_count)
    {
        sanisizer::cast<Index_>(my_sf.size());
        if (my_pseudo != 1) {
            my_sparse = false;
        }
        for (const auto x : my_sf) {
            if (x == 0) {
                my_sparse = false;
                my_zero_sf = true;
                break;
            }
        }
    }

private:
    SizeFactors_ my_sf;
    OutputValue_ my_base, my_pseudo;
    bool my_zero_sf = false, my_sparse = true;

public:
    std::optional<Index_> nrow() const {
        return std::nullopt;
    }

    std::optional<Index_> ncol() const {
        return my_sf.size();
    }

public:
    bool zero_depends_on_row() const {
        return false;
    }

    bool zero_depends_on_column() const {
        return my_zero_sf;
    }

    bool non_zero_depends_on_row() const {
        return false;
    }

    bool non_zero_depends_on_column() const {
        return true;
    }

private:
    void log_normalize(const Index_ idx, const Index_ length, const InputValue_* input, OutputValue_* const output) const {
        const auto sf = my_sf[idx];
        if (my_pseudo == 1) {
            for (Index_ i = 0; i < length; ++i) {
                output[i] = std::log1p(input[i] / sf) / my_base;
            }
        } else {
            for (Index_ i = 0; i < length; ++i) {
                output[i] = std::log(input[i] / sf + my_pseudo) / my_base;
            }
        }
    }

public:
    void dense(const bool row, const Index_ idx, const Index_ start, const Index_ length, const InputValue_* input, OutputValue_* const output) const {
        if constexpr(std::is_same<InputValue_, OutputValue_>::value) {
            input = output; // basically an assertion to the compiler to skip aliasing protection.
        }

        if (row) {
            if (my_pseudo == 1) {
                for (Index_ i = 0; i < length; ++i) {
                    output[i] = std::log1p(input[i] / my_sf[i + start]) / my_base;
                }
            } else {
                for (Index_ i = 0; i < length; ++i) {
                    output[i] = std::log(input[i] / my_sf[i + start] + my_pseudo) / my_base;
                }
            }
        } else {
            log_normalize(idx, length, input, output);
        }
    }

    void dense(const bool row, const Index_ idx, const std::vector<Index_>& indices, const InputValue_* input, OutputValue_* const output) const {
        if constexpr(std::is_same<InputValue_, OutputValue_>::value) {
            input = output; // basically an assertion to the compiler to skip aliasing protection.
        }
        const Index_ length = indices.size();

        if (row) {
            if (my_pseudo == 1) {
                for (Index_ i = 0; i < length; ++i) {
                    output[i] = std::log1p(input[i] / my_sf[indices[i]]) / my_base;
               }
            } else {
                for (Index_ i = 0; i < length; ++i) {
                    output[i] = std::log(input[i] / my_sf[indices[i]] + my_pseudo) / my_base;
                }
            }
        } else {
            log_normalize(idx, length, input, output);
        }
    }

public:
    bool is_sparse() const {
        return my_sparse;
    }

    void sparse(
        const bool row,
        const Index_ idx,
        const Index_ number,
        const InputValue_* input_value,
        const Index_* const index,
        OutputValue_* const output_value
    ) const {
        if constexpr(std::is_same<InputValue_, OutputValue_>::value) {
            input_value = output_value; // basically an assertion to the compiler to skip aliasing protection.
        }

        if (row) {
            if (my_pseudo == 1) {
                for (Index_ i = 0; i < number; ++i) {
                    output_value[i] = std::log1p(input_value[i] / my_sf[index[i]]) / my_base;
                }
            } else {
                for (Index_ i = 0; i < number; ++i) {
                    output_value[i] = std::log(input_value[i] / my_sf[index[i]] + my_pseudo) / my_base;
                }
            }
        } else {
            log_normalize(idx, number, input_value, output_value);
        }
    }

    OutputValue_ fill(const bool row, const Index_ idx) const {
        if (row) {
            // This should never be called in the presence of size factors of zero, as these will lead to NaNs.
            assert(!my_zero_sf);
            if (my_pseudo == 1) {
                return 0;
            } else {
                return std::log(my_pseudo) / my_base;
            }
        } else {
            if (my_pseudo == 1) {
                return std::log1p(static_cast<InputValue_>(0) / my_sf[idx]) / my_base;
            } else {
                return std::log(static_cast<InputValue_>(0) / my_sf[idx] + my_pseudo) / my_base;
            }
        }
    }
};

/**
 * @brief Options for `normalize_counts()`.
 */
struct NormalizeCountsOptions {
    /**
     * Pseudo-count to add to each value prior to log-transformation.
     * All values should be positive to ensure that log-transformed values are finite.
     * The default value of 1 preserves sparsity in the log-count matrix.
     * Larger values shrink the differences between cells towards zero, reducing spurious differences (but also signal) at low counts - see `choose_pseudo_count_raw()` for comments.
     * Ignored if `NormalizeCountsOptions::log = false`.
     */
    double pseudo_count = 1;

    /**
     * Whether to preserve sparsity for non-unity pseudo-counts.
     * If true, we multiply the size factors by the `NormalizeCountsOptions::pseudo_count` and add 1 before log-transformation.
     * This does not change the differences between entries of the resulting matrix,
     * and adding `log(pseudo_count/log(log_base)` will recover the expected log-count values.
     * Ignored if `NormalizeCountsOptions::log = false` or `NormalizeCountsOptions::pseudo_count = 1`.
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
 * The simplest and most common method for defining size factors is to use the centered library sizes (see `center_size_factors()` for details).
 * This removes scaling biases caused by differences in sequencing depth, capture efficiency etc. between cells.
 * The centering preserves the scale of the counts in the normalized expression values.
 * That said, users can define size factors from any method of their choice (e.g., median-based normalization, TMM) as long as they are positive for all cells.
 *
 * Normalized values are then typically log-transformed so that differences in log-values represent log-fold changes in expression.
 * This ensures that downstream analyses like t-tests and distance calculations focus on relative fold-changes rather than absolute differences.
 * The log-transformation also provides some measure of variance stabilization so that the downstream analyses are not dominated by sampling noise at large counts.
 * See also `DelayedLogNormalizeHelper`, which handles the delayed calculation of the log-transformed values.
 *
 * @tparam OutputValue_ Floating-point type of the output matrix.
 * @tparam InputValue_ Data type of the input matrix.
 * @tparam InputIndex_ Integer type of the input matrix.
 * @tparam SizeFactors_ Container of floats of the size factors.
 * This should have the `size()`, `begin()`, `end()` and `operator[]` methods.
 *
 * @param counts Pointer to a matrix of non-negative counts.
 * Rows should correspond to genes while columns should correspond to cells.
 * @param size_factors Vector of length equal to the number of columns in `counts`, containing the size factor for each cell.
 * All values should be positive, and any invalid values should be replaced with `sanitize_size_factors()`. 
 * In most applications, the size factors should also be centered via, e.g., `center_size_factors()`. 
 * @param options Further options.
 *
 * @return Pointer to a matrix of normalized expression values.
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


    if (!options.log) {
        return std::make_shared<tatami::DelayedUnaryIsometricOperation<OutputValue_, InputValue_, Index_> >(
            std::move(counts), 
            std::make_shared<tatami::DelayedUnaryIsometricDivideVectorHelper<true, OutputValue_, InputValue_, Index_, SizeFactors_> >(std::move(size_factors), false)
        );
    } else {
        return std::make_shared<tatami::DelayedUnaryIsometricOperation<OutputValue_, InputValue_, Index_> >(
            std::move(counts), 
            std::make_shared<DelayedLogNormalizeHelper<OutputValue_, InputValue_, Index_, SizeFactors_> >(std::move(size_factors), options.log_base, current_pseudo)
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
