#ifndef SCRAN_NORM_CHOOSE_PSEUDO_COUNT_HPP
#define SCRAN_NORM_CHOOSE_PSEUDO_COUNT_HPP

#include <algorithm>
#include <vector>
#include <cstddef>

#include "utils.hpp"

/**
 * @file choose_pseudo_count.hpp
 * @brief Choose a pseudo-count for log-transformation.
 */

namespace scran_norm {

/**
 * @brief Options for `choose_pseudo_count()`.
 */
struct ChoosePseudoCountOptions {
    /**
     * Quantile to use for finding the smallest/largest size factors.
     * Setting this to zero will use the observed minimum and maximum, though in practice, this is usually too sensitive to outliers.
     * The default is to take the 5th and 95th percentile to obtain a range that captures most of the distribution.
     */
    double quantile = 0.05;

    /**
     * Acceptable upper bound on the log2-transformation bias.
     * This is the magnitude of any spurious log2-fold change between the cells with the smallest and largest size factors.
     */
    double max_bias = 0.1;

    /**
     * Minimum value for the pseudo-count returned by `choose_pseudo_count()`.
     * Defaults to 1 to stabilize near-zero normalized expression values, otherwise these manifest as avoid large negative values.
     */
    double min_value = 1;
};

/**
 * @cond
 */
namespace internal {

template<typename Float_>
Float_ find_quantile(Float_ quantile, std::size_t n, Float_* ptr) {
    double raw = static_cast<double>(n - 1) * quantile;
    std::size_t index = std::ceil(raw);
    std::nth_element(ptr, ptr + index, ptr + n);
    double upper = *(ptr + index);
    std::nth_element(ptr, ptr + index - 1, ptr + index);
    double lower = *(ptr + index - 1);
    return lower * (index - raw) + upper * (raw - (index - 1));
}

}
/**
 * @endcond
 */

/**
 * Choose a pseudo-count for log-transformation (see `NormalizeCountsOptions::pseudo_count`) that aims to control the transformation-induced bias.
 *
 * Log-transformation is commonly applied to sequencing count data prior to further analyses (see `NormalizeCountsOptions::log`).
 * However, this can introduce spurious differences in the expected log-normalized expression between cells with very different size factors (Lun, 2018).
 * This bias is typically modest in datasets where there are stronger sources of variation,
 * but when observed, it manifests as a library size-dependent trend in the log-normalized expression values.
 * It is difficult to regress out without also removing biology that is associated with, e.g., total RNA content.
 *
 * A simpler solution is to increase the pseudo-count to suppress the bias.
 * This shrinks all log-expression values towards the zero-expression baseline, thus also shrinking log-differences between cells towards zero. 
 * The increased shrinkage is strongest at low counts where the data is least informative and the transformation bias is most pronounced.
 * At large counts, the shrinkage has less effect as the log-differences are driven by the data.
 * Our aim is to pick a pseudo-count that is large enough to mitigate the bias while being small enough to avoid shrinking the biological differences.
 * 
 * No centering is performed by this function, so the size factors should be passed through `center_size_factors()` before calling this function.
 * Invalid size factors (e.g., zero, negative, non-finite) are automatically ignored, so prior sanitization should not be performed -
 * this ensures that we do not include the replacement values in the various quantile calculations.
 *
 * @see
 * Lun ATL (2018).
 * Overcoming systematic errors caused by log-transformation of normalized single-cell RNA sequencing data.
 * _biorXiv_ doi:10.1101/404962
 *
 * @tparam Float_ Floating-point type of the size factors.
 *
 * @param num Number of size factors.
 * @param[in] size_factors Pointer to an array of size factors of length `num`.
 * Values should be positive, and all non-positive values are ignored.
 * On output, this array is arbitrarily permuted and should not be used.
 * @param options Further options.
 *
 * @return The suggested pseudo-count to control the log-transformation-induced bias below the specified threshold.
 */
template<typename Float_>
Float_ choose_pseudo_count_raw(std::size_t num, Float_* const size_factors, const ChoosePseudoCountOptions& options) {
    if (num <= 1) {
        return options.min_value;
    }

    // Avoid problems with zeros.
    decltype(I(num)) counter = 0;
    for (decltype(I(num)) i = 0; i < num; ++i) {
        const auto val = size_factors[i];
        if (std::isfinite(val) && val > 0) {
            if (i != counter) {
                size_factors[counter] = val;
            }
            ++counter;
        }
    }
    num = counter;

    if (num <= 1) {
        return options.min_value;
    }

    Float_ lower_sf, upper_sf;
    if (options.quantile == 0) {
        lower_sf = *std::min_element(size_factors, size_factors + num);
        upper_sf = *std::max_element(size_factors, size_factors + num);
    } else {
        lower_sf = internal::find_quantile(options.quantile, num, size_factors);
        upper_sf = internal::find_quantile(1 - options.quantile, num, size_factors);
    }

    // Very confusing formulation in Equation 3, but whatever.
    const Float_ pseudo_count = (1.0 / lower_sf - 1.0 / upper_sf) / (8 * options.max_bias);

    return std::max(options.min_value, pseudo_count);
}

/**
 * This function just wraps `choose_pseudo_count_raw()` with the automatic creation of a writeable buffer for the size factors.
 *
 * @tparam Float_ Floating-point type of the size factors.
 *
 * @param num Number of size factors.
 * @param[in] size_factors Pointer to an array of size factors of length `n`.
 * Values should be positive, and all non-positive values are ignored.
 * @param options Further options.
 *
 * @return The suggested pseudo-count to control the log-transformation-induced bias below the specified threshold.
 */
template<typename Float_>
Float_ choose_pseudo_count(const std::size_t num, const Float_* const size_factors, const ChoosePseudoCountOptions& options) {
    std::vector<Float_> buffer(size_factors, size_factors + num);
    return choose_pseudo_count_raw(num, buffer.data(), options);
}

}

#endif
