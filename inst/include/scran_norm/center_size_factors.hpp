#ifndef SCRAN_NORM_CENTER_SIZE_FACTORS_HPP
#define SCRAN_NORM_CENTER_SIZE_FACTORS_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>
#include <cstddef>

#include "tatami_stats/tatami_stats.hpp"

#include "sanitize_size_factors.hpp"
#include "utils.hpp"

/**
 * @file center_size_factors.hpp
 * @brief Center size factors prior to scaling normalization.
 */

namespace scran_norm {

/**
 * @brief Options for `compute_mean_size_factor()` and `compute_mean_size_factor_blocked()`.
 */
struct ComputeMeanSizeFactorOptions {
    /**
     * Whether to ignore invalid size factors when computing the mean size factor.
     * Non-finite or non-positive size factors may occur in datasets that have not been properly filtered to remove low-quality cells.
     * If such values might be present, we can check for and ignore them during the mean calculations.
     * If users know that invalid size factors cannot be present, they can set this flag to false for greater efficiency.
     */
    bool ignore_invalid = true;

    /**
     * Pointer to diagnostics for invalid size factors.
     * On return of `compute_mean_size_factor()`, the referenced object is filled with diagnostics for invalid values in `size_factors`.
     * It can also be `NULL`, in which case it is ignored.
     *
     * Ignored if `ComputeMeanSizeFactorOptions::ignore_invalid = false`,
     */
    SizeFactorDiagnostics* diagnostics = NULL;
};

/**
 * Compute the mean size factor. 
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of cells.
 * @param[in] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * @param options Further options.
 *
 * @return The mean size factor.
 */
template<typename SizeFactor_>
SizeFactor_ compute_mean_size_factor(const std::size_t num, const SizeFactor_* const size_factors, const ComputeMeanSizeFactorOptions& options) {
    static_assert(std::is_floating_point<SizeFactor_>::value);
    SizeFactor_ mean = 0;
    I<decltype(num)> denom = 0;

    if (options.ignore_invalid) {
        SizeFactorDiagnostics tmpdiag;
        for (I<decltype(num)> i = 0; i < num; ++i) {
            const auto val = size_factors[i];
            if (!internal::is_invalid(val, tmpdiag)) {
                mean += val;
                ++denom;
            }
        }
        if (options.diagnostics != NULL) {
            *(options.diagnostics) = tmpdiag;
        }

    } else {
        mean = std::accumulate(size_factors, size_factors + num, static_cast<SizeFactor_>(0));
        denom = num;
    }

    if (denom) {
        return mean / denom;
    } else {
        return 0;
    }
}

/**
 * Compute the mean size factor for each block.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 * @tparam Block_ Integer type of the block assignments.
 *
 * @param num Number of cells.
 * @param[in] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * @param[in] block Pointer to an array of length `num`, containing the block assignment for each cell.
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
 * @param options Further options.
 *
 * @return Vector of length \f$N\f$ containing the mean size factor for each block.
 */
template<typename SizeFactor_, typename Block_>
std::vector<SizeFactor_> compute_mean_size_factor_blocked(
    const std::size_t num,
    const SizeFactor_* const size_factors,
    const Block_* const block,
    const ComputeMeanSizeFactorOptions& options
) {
    static_assert(std::is_floating_point<SizeFactor_>::value);
    const auto ngroups = tatami_stats::total_groups(block, num);
    auto group_mean = sanisizer::create<std::vector<SizeFactor_> >(ngroups);
    auto group_num = sanisizer::create<std::vector<I<decltype(num)>> >(ngroups);

    if (options.ignore_invalid) {
        SizeFactorDiagnostics tmpdiag;
        for (I<decltype(num)> i = 0; i < num; ++i) {
            const auto val = size_factors[i];
            if (!internal::is_invalid(val, tmpdiag)) {
                const auto b = block[i];
                group_mean[b] += val;
                ++(group_num[b]);
            }
        }
        if (options.diagnostics != NULL) {
            *(options.diagnostics) = tmpdiag;
        }

    } else {
        for (I<decltype(num)> i = 0; i < num; ++i) {
            const auto b = block[i];
            group_mean[b] += size_factors[i];
            ++(group_num[b]);
        }
    }

    for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
        if (group_num[g]) {
            group_mean[g] /= group_num[g];
        }
    }

    return group_mean;
}

/**
 * @brief Options for `center_size_factors()`. 
 */
struct CenterSizeFactorsOptions {
    /**
     * Whether to ignore invalid size factors when computing the mean size factor, see `ComputeMeanSizeFactorOptions::ignore_invalid` for details.
     *
     * Note that setting this option to `true` does not actually remove any of the invalid size factors.
     * If these are present, users should call `sanitize_size_factors()` after centering.
     * The `diagnostics` value in `center_size_factors()` and `center_size_factors_blocked()` can be used to determine whether such a call is necessary.
     * (In general, sanitization should be performed after centering so that the replacement size factors do not interfere with the mean calculations.)
     */
    bool ignore_invalid = true;

    /**
     * Mean of the size factors after centering.
     * This should almost always be 1, to ensure that the normalized expression values are on roughly the same scale as the original counts.
     * Nonetheless, expert users can change it to some non-unity value.
     */
    double center = 1;

    /**
     * Pointer to diagnostics for invalid size factors, passed to `ComputeMeanSizeFactorOptions::diagnostics`. 
     * Ignored if `CenterSizeFactorsOptions::ignore_invalid = false`.
     */
    SizeFactorDiagnostics* diagnostics = NULL;

    /**
     * Whether to report the final mean of the size factors, i.e., after centering.
     * If `false`, the mean of the input size factors is reported instead.
     */
    bool report_final = false;
};

/**
 * Centering the size factors involves scaling all size factors so that the mean across cells is equal to 1 (i.e., `CenterSizeFactorsOptions::center`).
 * The aim is to ensure that the normalized expression values are on roughly the same scale as the original counts.
 * This simplifies interpretation and ensures that any pseudo-count added prior to log-transformation has a predictable shrinkage effect.
 * In general, size factors should be centered before calling `normalize_counts()`.
 * 
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of cells.
 * @param[in,out] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * On output, this contains the centered size factors, scaled such that their mean is equal to `CenterSizeFactorsOptions::center`.
 * If the mean of the input size factors is zero, no scaling is performed.
 * @param options Further options.
 *
 * @return The mean of the input size factors (if `CenterSizeFactorsOptions::report_final = false`) or the mean of the size factors after centering (otherwise).
 */
template<typename SizeFactor_>
SizeFactor_ center_size_factors(const std::size_t num, SizeFactor_* const size_factors, const CenterSizeFactorsOptions& options) {
    ComputeMeanSizeFactorOptions copt;
    copt.ignore_invalid = options.ignore_invalid;
    copt.diagnostics = options.diagnostics;
    const auto mean = compute_mean_size_factor(num, size_factors, copt);

    if (mean == 0) {
        return 0;
    }

    const SizeFactor_ mult = options.center / mean;
    for (I<decltype(num)> i = 0; i < num; ++i){
        size_factors[i] *= mult;
    }

    if (options.report_final) {
        return options.center;
    } else {
        return mean;
    }
}

/**
 * Strategy for handling blocks when centering size factors, see `CenterSizeFactorsOptions::block_mode` for details.
 */
enum class CenterBlockMode : char { PER_BLOCK, LOWEST, CUSTOM };

/**
 * @brief Options for `center_size_factors_blocked()`.
 */
struct CenterSizeFactorsBlockedOptions {
    /**
     * Whether to ignore invalid size factors when computing the mean size factor, see `ComputeMeanSizeFactorOptions::ignore_invalid` for details.
     * Note that setting this option to `true` does not actually remove any of the invalid size factors, see comments at `CenterSizeFactorsOptions::ignore_invalid`.
     */
    bool ignore_invalid = true;

    /**
     * Pointer to diagnostics for invalid size factors, passed to `ComputeMeanSizeFactorOptions::diagnostics`. 
     * Ignored if `CenterSizeFactorsBlockedOptions::ignore_invalid = false`.
     */
    SizeFactorDiagnostics* diagnostics = NULL;

    /**
     * Strategy for handling blocks in `center_size_factors_blocked()`.
     *
     * With `PER_BLOCK`, size factors are scaled separately for each block so that they have a mean of 1 within each block.
     * The scaled size factors are identical to those obtained by separate invocations of `center_size_factors()` on the size factors for each block.
     * This can be desirable to ensure consistency with independent analyses of each block - otherwise, the centering would depend on the size factors in other blocks.
     * However, any systematic differences in the size factors between blocks are lost, i.e., systematic changes in coverage between blocks will not be normalized.
     * 
     * With `LOWEST`, we compute the mean size factor for each block and we divide all size factors in all blocks by the lowest of the per-block means.
     * Here, our normalization strategy involves downscaling all blocks to match the coverage of the lowest-coverage block.
     * This is useful for datasets with big differences in coverage between blocks as it avoids egregious upscaling of low-coverage blocks.
     * Specifically, strong upscaling allows the log-transformation to ignore any shrinkage from the pseudo-count.
     * This is problematic as it inflates differences between cells at log-values derived from low counts, increasing noise and overstating log-fold changes. 
     * Downscaling is safer as it allows the pseudo-count to shrink the log-differences between cells towards zero at low counts,
     * effectively sacrificing some information in the higher-coverage batches so that they can be compared to the low-coverage batches
     * (which is preferable to exaggerating the informativeness of the latter for comparison to the former).
     *
     * With `CUSTOM`, size factors are scaled such that the mean for each block is equal to that specified in `CenterSizeFactorsBlockedOptions::custom_centers`.
     * This is occasionally useful for ensuring that different sets of size factors are scaled to the same per-block mean,
     * e.g., to ensure that average abundances are comparable between spike-in transcripts and endogenous genes in `center_spike_in_factors_blocked()`.
     *
     * In all cases, if the mean of the input size factors for any block is zero, no centering is attempted for that block.
     */
    CenterBlockMode block_mode = CenterBlockMode::LOWEST;

    /**
     * Mean of the size factors after centering.
     * Only used if `CenterSizeFactorsBlockedOptions::block_mode = CenterBlockMode::CUSTOM`.
     */
    std::optional<std::vector<double> > custom_centers;

    /**
     * Whether to report the final mean of the size factors in each block, i.e., after centering.
     * If `false`, the means of the input size factors are reported instead.
     */
    bool report_final = false;
};

/**
 * Center size factors within each block to obtain interpretable values after normalization.
 * The rationale is the same as discussed in `center_size_factors()` but some additional work is required to account for experimental blocking, 
 * e.g., to accommodate systematic differences in sequencing depth between runs.
 * The exact strategy for adjusting size factors between blocks is controlled by `CenterSizeFactorsOptions::block_mode`.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 * @tparam Block_ Integer type of the block assignments.
 *
 * @param num Number of cells.
 * @param[in,out] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * On output, this contains size factors that are centered according to `CenterSizeFactorsOptions::block_mode`.
 * @param[in] block Pointer to an array of length `num`, containing the block assignment for each cell.
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
 * @param options Further options.
 *
 * @return Vector of length \f$N\f$ containing the mean size factor for each block, used to scale `size_factors` on output.
 */
template<typename SizeFactor_, typename Block_>
std::vector<SizeFactor_> center_size_factors_blocked(
    const std::size_t num,
    SizeFactor_* const size_factors,
    const Block_* const block,
    const CenterSizeFactorsBlockedOptions& options
) {
    ComputeMeanSizeFactorOptions copt;
    copt.ignore_invalid = options.ignore_invalid;
    copt.diagnostics = options.diagnostics;
    auto group_mean = compute_mean_size_factor_blocked(num, size_factors, block, copt);
    const auto ngroups = group_mean.size();

    if (options.block_mode == CenterBlockMode::PER_BLOCK) {
        std::vector<SizeFactor_> fac;
        fac.reserve(ngroups);
        for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
            const auto gm = group_mean[g];
            if (gm) {
                fac.emplace_back(1 / gm);
            } else {
                fac.emplace_back(1); // i.e., no-op.
            }
        }

        for (I<decltype(num)> i = 0; i < num; ++i) {
            size_factors[i] *= fac[block[i]];
        }

        if (options.report_final) {
            for (auto& gm : group_mean) {
                if (gm) {
                    gm = 1;
                }
            }
        }

        return group_mean;

    } else if (options.block_mode == CenterBlockMode::LOWEST) {
        SizeFactor_ min = 0;
        bool found = false;
        for (const auto m : group_mean) {
            // Ignore groups with means of zeros, either because they're full
            // of zeros themselves or they have no cells associated with them.
            if (m) {
                if (!found || m < min) {
                    min = m;
                    found = true;
                }
            }
        }

        if (min) {
            const SizeFactor_ mult = 1 / min;
            for (I<decltype(num)> i = 0; i < num; ++i) {
                size_factors[i] *= mult;
            }

            if (options.report_final) {
                for (auto& gm : group_mean) {
                    gm /= min;
                }
            }
        }

        return group_mean;

    } else { // i.e., options.block_mode == CenterBlockMode::CUSTOM
        if (!options.custom_centers.has_value()) {
            throw std::runtime_error("'custom_centers' should be set for custom block centers"); 
        }
        const auto& custom = *(options.custom_centers);
        const auto ngroups = group_mean.size();
        if (custom.size() != ngroups) {
            throw std::runtime_error("length of 'custom_centers' should be equal to the number of groups"); 
        }

        std::vector<SizeFactor_> fac;
        fac.reserve(ngroups);
        for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
            const auto gm = group_mean[g];
            if (gm) {
                fac.emplace_back(custom[g] / gm);
            } else {
                fac.emplace_back(1);
            }
        }

        for (I<decltype(num)> i = 0; i < num; ++i) {
            size_factors[i] *= fac[block[i]];
        }

        if (options.report_final) {
            for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
                auto& gm = group_mean[g];
                if (gm) {
                    gm = custom[g];
                }
            }
        }

        return group_mean;
    }
}

}

#endif
