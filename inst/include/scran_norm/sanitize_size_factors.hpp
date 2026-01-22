#ifndef SCRAN_SANITIZE_SIZE_FACTORS_HPP
#define SCRAN_SANITIZE_SIZE_FACTORS_HPP

#include <cmath>
#include <stdexcept>
#include <cstddef>

#include "utils.hpp"

/**
 * @file sanitize_size_factors.hpp
 * @brief Sanitize invalid size factors.
 */

namespace scran_norm {

/**
 * @brief Diagnostics for the size factors.
 */
struct SizeFactorDiagnostics {
    /**
     * Whether negative factors were detected.
     */
    bool has_negative = false;

    /**
     * Whether size factors of zero were detected.
     */
    bool has_zero = false;

    /**
     * Whether NaN size factors were detected.
     */
    bool has_nan = false;

    /**
     * Whether size factors of positive infinity were detected.
     */
    bool has_infinite = false;
};

/**
 * @cond
 */
namespace internal {

template<typename SizeFactor_>
bool is_invalid(SizeFactor_ sf, SizeFactorDiagnostics& output) {
    if (sf < 0) {
        output.has_negative = true;
        return true;
    }

    if (sf == 0) {
        output.has_zero = true;
        return true;
    }

    if (std::isnan(sf)) {
        output.has_nan = true;
        return true;
    }

    if (std::isinf(sf)) {
        output.has_infinite = true;
        return true;
    }

    return false;
}

template<typename SizeFactor_>
SizeFactor_ find_smallest_valid_factor(const std::size_t num, const SizeFactor_* const size_factors) {
    SizeFactor_ smallest = 1;
    bool found = false;

    for (I<decltype(num)> i = 0; i < num; ++i) {
        const auto s = size_factors[i];
        if (std::isfinite(s) && s > 0) {
            if (!found || smallest > s) {
                smallest = s;
                found = true;
            }
        }
    }

    return smallest;
}

template<typename SizeFactor_>
SizeFactor_ find_largest_valid_factor(const std::size_t num, const SizeFactor_* const size_factors) {
    SizeFactor_ largest = 1;
    bool found = false;

    for (I<decltype(num)> i = 0; i < num; ++i) {
        const auto s = size_factors[i];
        if (std::isfinite(s) && s > 0) {
            if (!found || largest < s) {
                largest = s;
                found = true;
            }
        }
    }

    return largest;
}

}
/**
 * @endcond
 */

/**
 * Check whether there are any invalid size factors.
 * Size factors are only valid if they are finite and positive.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of size factors.
 * @param[in] size_factors Pointer to an array of size factors of length `num`.
 *
 * @return Validation results, indicating whether any zero or non-finite size factors exist.
 */
template<typename SizeFactor_>
SizeFactorDiagnostics check_size_factor_sanity(const std::size_t num, const SizeFactor_* const size_factors) {
    SizeFactorDiagnostics output;
    for (I<decltype(num)> i = 0; i < num; ++i) {
        internal::is_invalid(size_factors[i], output);
    }
    return output;
}

/**
 * How invalid size factors should be handled:
 *
 * - `IGNORE`: ignore invalid size factors with no error or change.
 * - `ERROR`: throw an error.
 * - `SANITIZE`: fix each invalid size factor.
 */
enum class SanitizeAction : char { IGNORE, ERROR, SANITIZE };

/**
 * @brief Options for `sanitize_size_factors()`.
 */
struct SanitizeSizeFactorsOptions {
    /**
     * How should we handle size factors of zero?
     *
     * If set to `SANITIZE`, they will be automatically set to the smallest valid size factor (or 1, if all size factors are invalid).
     * This ensures that the normalized values will be large to reflect the extremity of the scaling, but still finite for sensible downstream processing. 
     *
     * In most cases, size factors of zero are only obtained for cells with all-zero counts.
     * By replacing the size factor with a finite value, we ensure that any all-zero cells are represented by all-zero columns in the normalized matrix,
     * which is a reasonable outcome if those cells cannot be filtered out during upstream quality control.
     */
    SanitizeAction handle_zero = SanitizeAction::ERROR;

    /**
     * How should we handle negative size factors?
     *
     * If set to `SANITIZE`, they will be automatically set to the smallest valid size factor (or 1, if all size factors are invalid),
     * following the same logic as `SanitizeSizeFactorsOptions::handle_zero`.
     */
    SanitizeAction handle_negative = SanitizeAction::ERROR;

    /**
     * How should we handle NaN size factors?
     *
     * If set to `SANITIZE, NaN size factors will be automatically set to 1, meaning that scaling is a no-op.
     */
    SanitizeAction handle_nan = SanitizeAction::ERROR;

    /**
     * How should we handle infinite size factors?
     *
     * If set to `SANITIZE`, infinite size factors will be automatically set to the largest valid size factor (or 1, if all size factors are invalid).
     * This ensures that any normalized values will be, at least, finite; the choice of a relatively large replacement value reflects the extremity of the scaling.
     */
    SanitizeAction handle_infinite = SanitizeAction::ERROR;
};

/**
 * Replace zero, missing or infinite values in the size factor array so that they can be used to compute well-defined normalized values in `normalize_counts()`.
 * Such size factors can occasionally arise if, e.g., insufficient quality control was performed upstream.
 * Check out the documentation in `SanitizeSizeFactorsOptions` to see what placeholder value is used for each type of invalid size factor.
 *
 * In general, sanitization should occur after calls to `center_size_factors()`, `choose_pseudo_count()`, or any function that computes a statistic based on the distribution of size factors.
 * This ensures that the results of those functions are not affected by the placeholder values used to replace the invalid size factors.
 * As a rule of thumb, `sanitize_size_factors()` should be called just before passing those size factors to `normalize_counts()`.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of size factors.
 * @param[in,out] size_factors Pointer to an array of positive size factors of length `n`.
 * On output, invalid size factors may be replaced depending on the settings in `options`.
 * @param status A pre-computed object indicating whether invalid size factors are present in `size_factors`.
 * This can be useful if this information is already provided by, e.g., `check_size_factor_sanity()` or `center_size_factors()`.
 * @param options Further options.
 */
template<typename SizeFactor_>
void sanitize_size_factors(const std::size_t num, SizeFactor_* const size_factors, const SizeFactorDiagnostics& status, const SanitizeSizeFactorsOptions& options) {
    SizeFactor_ smallest = -1;

    if (status.has_negative) {
        if (options.handle_negative == SanitizeAction::ERROR) {
            throw std::runtime_error("detected negative size factor");
        } else if (options.handle_negative == SanitizeAction::SANITIZE) {
            smallest = internal::find_smallest_valid_factor(num, size_factors);
            for (I<decltype(num)> i = 0; i < num; ++i) {
                auto& s = size_factors[i];
                if (s < 0) {
                    s = smallest;
                }
            }
        }
    }

    if (status.has_zero) {
        if (options.handle_zero == SanitizeAction::ERROR) {
            throw std::runtime_error("detected size factor of zero");
        } else if (options.handle_zero == SanitizeAction::SANITIZE) {
            if (smallest < 0) {
                smallest = internal::find_smallest_valid_factor(num, size_factors);
            }
            for (I<decltype(num)> i = 0; i < num; ++i) {
                auto& s = size_factors[i];
                if (s == 0) {
                    s = smallest;
                }
            }
        }
    }

    if (status.has_nan) {
        if (options.handle_nan == SanitizeAction::ERROR) {
            throw std::runtime_error("detected NaN size factor");
        } else if (options.handle_nan == SanitizeAction::SANITIZE) {
            for (I<decltype(num)> i = 0; i < num; ++i) {
                auto& s = size_factors[i];
                if (std::isnan(s)) {
                    s = 1;
                }
            }
        }
    }

    if (status.has_infinite) {
        if (options.handle_infinite == SanitizeAction::ERROR) {
            throw std::runtime_error("detected infinite size factor");
        } else if (options.handle_infinite == SanitizeAction::SANITIZE) {
            auto largest = internal::find_largest_valid_factor(num, size_factors);
            for (I<decltype(num)> i = 0; i < num; ++i) {
                auto& s = size_factors[i];
                if (std::isinf(s)) {
                    s = largest;
                }
            }
        }
    }
}

/**
 * Overload of `sanitize_size_factors()` that calls `check_size_factor_sanity()` internally.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of size factors.
 * @param[in,out] size_factors Pointer to an array of positive size factors of length `n`.
 * On output, invalid size factors are replaced.
 * @param options Further options.
 *
 * @return An object indicating whether each type of invalid size factors is present in `size_factors`.
 */
template<typename SizeFactor_>
SizeFactorDiagnostics sanitize_size_factors(const std::size_t num, SizeFactor_* const size_factors, const SanitizeSizeFactorsOptions& options) {
    const auto output = check_size_factor_sanity(num, size_factors);
    sanitize_size_factors(num, size_factors, output, options);
    return output;
}

}

#endif
