#ifndef SCRAN_VARIANCES_FIT_VARIANCE_TREND_H
#define SCRAN_VARIANCES_FIT_VARIANCE_TREND_H

#include <algorithm>
#include <vector>
#include <array>
#include <cstddef>

#include "WeightedLowess/WeightedLowess.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file fit_variance_trend.hpp
 * @brief Fit a mean-variance trend to log-count data.
 */

namespace scran_variances {

/**
 * @brief Options for `fit_variance_trend()`.
 */
struct FitVarianceTrendOptions {
    /**
     * Minimum mean log-expression for trend fitting.
     * Genes with lower means do not participate in the LOWESS fit, to ensure that windows are not skewed towards the majority of low-abundance genes.
     * Instead, the fitted values for these genes are defined by extrapolating the left edge of the fitted trend is extrapolated to the origin.
     * The default value is chosen based on the typical distribution of means of log-expression values across genes.
     * Only used if `FitVarianceTrendOptions::mean_filter = true`.
     */
    double minimum_mean = 0.1;

    /**
     * Should any filtering be performed on the mean log-expression of each gene (see `FitVarianceTrend::minimum_mean`)?
     * The default of `true` assumes that there is a bulk of low-abundance genes that are uninteresting and should be removed to avoid skewing the windows of the LOWESS smoother.
     */
    bool mean_filter = true;

    /**
     * Should any quarter-root transformation of the variances be performed prior to LOWESS smoothing?
     * This transformation is copied from `limma::voom()` and shrinks all values towards 1, flattening any sharp gradients in the trend for an easier fit.
     * The default of `true` assumes that the variances are computed from log-expression values, in which case there is typically a strong "hump" in the mean-variance relationship.
     */
    bool transform = true;

    /**
     * Span for the LOWESS smoother, as a proportion of the total number of points.
     * Larger values improve stability at the cost of sensitivity to changes in low-density regions.
     * This is only used if `FitVarianceTrendOptions::use_minimum_width = false`.
     */
    double span = 0.3;

    /**
     * Should a minimum width constraint be applied to the LOWESS smoother?
     * This replaces the proportion-based span for defining each window.
     * Instead, the window for each point must be of a minimum width (see `FitVarianceTrendOptions::minimum_width`) 
     * and is extended until it contains a minimum number of points (see `FitVarianceTrendOptions::minimum_window_count`).
     * Setting this to `true` ensures that sensitivity is maintained in the trend fit at low-density regions for the distribution of means, e.g., at high abundances.
     * It also avoids overfitting from very small windows in high-density intervals. 
     */
    bool use_minimum_width = false;

    /**
     * Minimum width of the window to use when `FitVarianceTrendOptions::use_minimum_width = true`.
     * This should be appropriate for the range of means used in `fit_variance_trend()`.
     * The default value is chosen based on the typical range in single-cell RNA-seq data.
     */
    double minimum_width = 1;

    /**
     * Minimum number of observations in each window when `FitVarianceTrendOptions::use_minimum_width = true`.
     * This ensures that each window contains at least a given number of observations for a stable fit.
     * If the minimum width window contains fewer observations, it is extended using the standard LOWESS logic until the minimum number is achieved.
     */
    int minimum_window_count = 200;

    /**
     * Number of threads to use in the LOWESS fit.
     * The parallelization scheme is defined by `WeightedLowess::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Workspace for `fit_variance_trend()`.
 *
 * This avoids repeated memory allocations for repeated calls to `fit_variance_trend()`.
 */
template<typename Float_>
struct FitVarianceTrendWorkspace {
    /**
     * @cond
     */
    WeightedLowess::SortBy sorter;

    std::vector<unsigned char> sort_workspace;

    std::vector<Float_> xbuffer, ybuffer;
    /**
     * @endcond
     */
};

/**
 * Fit a trend to the per-feature variances against the means, both of which are typically computed from log-normalized expression data.
 * We use a LOWESS smoother in several steps:
 *
 * 1. Filter out low-abundance genes, to ensure the span of the smoother is not skewed by many low-abundance genes.
 * 2. Take the quarter-root of the variances, to squeeze the trend towards 1.
 * This makes the trend more "linear" to improve the performance of the LOWESS smoother;
 * it also reduces the chance of obtaining negative fitted values.
 * 3. Apply the LOWESS smoother to the quarter-root variances.
 * This is done using the implementation in the [**WeightedLowess**](https://github.com/LTLA/CppWeightedLowess) library.
 * 4. Reverse the quarter-root transformation to obtain the fitted values for all non-low-abundance genes.
 * 5. Extrapolate linearly from the left-most fitted value to the origin to obtain fitted values for the previously filtered genes.
 * This is empirically justified by the observation that mean-variance trends of log-expression data are linear at very low abundances.
 *
 * @tparam Float_ Floating-point type of the statistics.
 *
 * @param n Number of features.
 * @param[in] mean Pointer to an array of length `n`, containing the means for all features.
 * @param[in] variance Pointer to an array of length `n`, containing the variances for all features.
 * @param[out] fitted Pointer to an array of length `n`, to store the fitted values.
 * @param[out] residuals Pointer to an array of length `n`, to store the residuals.
 * @param workspace Collection of temporary data structures.
 * This can be re-used across multiple `fit_variance_trend()` calls.
 * @param options Further options.
 */
template<typename Float_>
void fit_variance_trend(
    const std::size_t n,
    const Float_* const mean,
    const Float_* const variance,
    Float_* const fitted,
    Float_* const residuals,
    FitVarianceTrendWorkspace<Float_>& workspace,
    const FitVarianceTrendOptions& options)
{
    auto& xbuffer = workspace.xbuffer;
    sanisizer::resize(xbuffer, n);
    auto& ybuffer = workspace.ybuffer;
    sanisizer::resize(ybuffer, n);

    const auto quad = [](Float_ x) -> Float_ {
        return x * x * x * x;
    };

    std::size_t counter = 0;
    const Float_ min_mean = options.minimum_mean;
    for (decltype(I(n)) i = 0; i < n; ++i) {
        if (!options.mean_filter || mean[i] >= min_mean) {
            xbuffer[counter] = mean[i];
            if (options.transform) {
                ybuffer[counter] = std::pow(variance[i], 0.25); // Using the same quarter-root transform that limma::voom uses.
            } else {
                ybuffer[counter] = variance[i];
            }
            ++counter;
        }
    }

    if (counter < 2) {
        throw std::runtime_error("not enough observations above the minimum mean");
    }

    auto& sorter = workspace.sorter;
    sorter.set(counter, xbuffer.data());
    auto& work = workspace.sort_workspace;
    sorter.permute(std::array<Float_*, 2>{ xbuffer.data(), ybuffer.data() }, work);

    WeightedLowess::Options<Float_> smooth_opt;
    if (options.use_minimum_width) {
        smooth_opt.span = options.minimum_window_count;
        smooth_opt.span_as_proportion = false;
        smooth_opt.minimum_width = options.minimum_width;
    } else {
        smooth_opt.span = options.span;
    }
    smooth_opt.num_threads = options.num_threads;

    // Using the residual array to store the robustness weights as a placeholder;
    // we'll be overwriting this later.
    WeightedLowess::compute(counter, xbuffer.data(), ybuffer.data(), fitted, residuals, smooth_opt);

    // Determining the left edge before we unpermute.
    const Float_ left_x = xbuffer[0];
    const Float_ left_fitted = (options.transform ? quad(fitted[0]) : fitted[0]);

    sorter.unpermute(fitted, work);

    // Walking backwards to shift the elements back to their original position
    // (i.e., before filtering on the mean) on the same array. We need to walk
    // backwards to ensure that writing to the original position on this array
    // doesn't clobber the first 'counter' positions containing the fitted
    // values, at least not until each value is shifted to its original place.
    for (auto i = n; i > 0; --i) {
        auto j = i - 1;
        if (!options.mean_filter || mean[j] >= min_mean) {
            --counter;
            fitted[j] = (options.transform ? quad(fitted[counter]) : fitted[counter]);
        } else {
            fitted[j] = mean[j] / left_x * left_fitted; // draw a y = x line to the origin from the left of the fitted trend.
        }
    }

    for (decltype(I(n)) i = 0; i < n; ++i) {
        residuals[i] = variance[i] - fitted[i];
    }
    return;
}

/**
 * @brief Results of `fit_variance_trend()`.
 *
 * Meaningful instances of this object should generally be constructed by calling the `fit_variance_trend()` function.
 * Empty instances can be default-constructed as placeholders.
 *
 * @tparam Float_ Floating-point type of the statistics.
 */
template<typename Float_>
struct FitVarianceTrendResults {
    /**
     * @cond
     */
    FitVarianceTrendResults() {}

    FitVarianceTrendResults(const std::size_t n) :
        fitted(sanisizer::cast<decltype(I(fitted.size()))>(n)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        residuals(sanisizer::cast<decltype(I(residuals.size()))>(n)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        )
    {}
    /**
     * @endcond
     */

    /**
     * Vector of length equal to the number of features, containing fitted values from the trend.
     */
    std::vector<Float_> fitted;

    /**
     * Vector of length equal to the number of features, containing residuals from the trend.
     */
    std::vector<Float_> residuals;
};

/**
 * Overload of `fit_variance_trend()` that allocates the output vectors.
 *
 * @tparam Float_ Floating-point type of the statistics.
 *
 * @param n Number of features.
 * @param[in] mean Pointer to an array of length `n`, containing the means for all features.
 * @param[in] variance Pointer to an array of length `n`, containing the variances for all features.
 * @param options Further options.
 * 
 * @return Result of the trend fit, containing the fitted values and residuals for each gene. 
 */
template<typename Float_>
FitVarianceTrendResults<Float_> fit_variance_trend(const std::size_t n, const Float_* const mean, const Float_* const variance, const FitVarianceTrendOptions& options) {
    FitVarianceTrendResults<Float_> output(n);
    FitVarianceTrendWorkspace<Float_> work;
    fit_variance_trend(n, mean, variance, output.fitted.data(), output.residuals.data(), work, options);
    return output;
}

}

#endif
