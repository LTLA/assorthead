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
 * Strategy for handling blocks when centering size factors, see `CenterSizeFactorsOptions::block_mode` for details.
 */
enum class CenterBlockMode : char { PER_BLOCK, LOWEST };

/**
 * @brief Options for `center_size_factors()` and `center_size_factors_blocked()`.
 */
struct CenterSizeFactorsOptions {
    /**
     * Strategy for handling blocks in `center_size_factors_blocked()`.
     *
     * With the `PER_BLOCK` strategy, size factors are scaled separately for each block so that they have a mean of 1 within each block.
     * The scaled size factors are identical to those obtained by separate invocations of `center_size_factors()` on the size factors for each block.
     * This can be desirable to ensure consistency with independent analyses of each block - otherwise, the centering would depend on the size factors in other blocks.
     * However, any systematic differences in the size factors between blocks are lost, i.e., systematic changes in coverage between blocks will not be normalized.
     * 
     * With the `LOWEST` strategy, we compute the mean size factor for each block and we divide all size factors in all blocks by the lowest of the per-block means.
     * Here, our normalization strategy involves downscaling all blocks to match the coverage of the lowest-coverage block.
     * This is useful for datasets with big differences in coverage between blocks as it avoids egregious upscaling of low-coverage blocks.
     * Specifically, strong upscaling allows the log-transformation to ignore any shrinkage from the pseudo-count.
     * This is problematic as it inflates differences between cells at log-values derived from low counts, increasing noise and overstating log-fold changes. 
     * Downscaling is safer as it allows the pseudo-count to shrink the log-differences between cells towards zero at low counts,
     * effectively sacrificing some information in the higher-coverage batches so that they can be compared to the low-coverage batches
     * (which is preferable to exaggerating the informativeness of the latter for comparison to the former).
     */
    CenterBlockMode block_mode = CenterBlockMode::LOWEST;

    /**
     * Whether to ignore invalid size factors when computing the mean size factor.
     * Size factors of infinity and NaN or those with non-positive values may occur in datasets that have not been properly filtered to remove low-quality cells.
     * If such values might be present, we can check for and ignore them during the mean calculations.
     *
     * Note that this setting does not actually remove any of the invalid size factors.
     * If these are present, users should call `sanitize_size_factors()` after centering.
     * The `diagnostics` value in `center_size_factors()` and `center_size_factors_blocked()` can be used to determine whether such a call is necessary.
     * (In general, sanitization should be performed after centering so that the replacement size factors do not interfere with the mean calculations.)
     *
     * If users know that invalid size factors cannot be present, they can set this flag to false for greater efficiency.
     */
    bool ignore_invalid = true;
};

/**
 * Compute the mean size factor but do not scale the size factors themselves.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of cells.
 * @param[in] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * @param[out] diagnostics Diagnostics for invalid size factors.
 * This is only used if `CenterSizeFactorsOptions::ignore_invalid = true`, in which case it is filled with diagnostics for invalid values in `size_factors`.
 * It can also be `NULL`, in which case it is ignored.
 * @param options Further options.
 *
 * @return The mean size factor, to be used to divide each element of `size_factors`.
 */
template<typename SizeFactor_>
SizeFactor_ center_size_factors_mean(const std::size_t num, const SizeFactor_* const size_factors, SizeFactorDiagnostics* const diagnostics, const CenterSizeFactorsOptions& options) {
    static_assert(std::is_floating_point<SizeFactor_>::value);
    SizeFactor_ mean = 0;
    I<decltype(num)> denom = 0;

    if (options.ignore_invalid) {
        SizeFactorDiagnostics tmpdiag;
        auto& diag = (diagnostics == NULL ? tmpdiag : *diagnostics);
        for (I<decltype(num)> i = 0; i < num; ++i) {
            const auto val = size_factors[i];
            if (!internal::is_invalid(val, diag)) {
                mean += val;
                ++denom;
            }
        }
    } else {
        mean = std::accumulate(size_factors, size_factors + num, static_cast<SizeFactor_>(0));
        denom = num;
    }

    if (denom) {
        return mean/denom;
    } else {
        return 0;
    }
}

/**
 * Centering the size factors involves scaling all of the size factors so that the mean across cells is equal to 1.
 * The aim is to ensure that the normalized expression values are on roughly the same scale as the original counts.
 * This simplifies interpretation and ensures that any pseudo-count added prior to log-transformation has a predictable shrinkage effect.
 * In general, size factors should be centered before calling `normalize_counts()`.
 * 
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of cells.
 * @param[in,out] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * On output, this contains the centered size factors.
 * @param[out] diagnostics Diagnostics for invalid size factors.
 * This is only used if `CenterSizeFactorsOptions::ignore_invalid = true`, in which case it is filled with diagnostics for invalid values in `size_factors`.
 * It can also be `NULL`, in which case it is ignored.
 * @param options Further options.
 *
 * @return The mean size factor.
 */
template<typename SizeFactor_>
SizeFactor_ center_size_factors(const std::size_t num, SizeFactor_* const size_factors, SizeFactorDiagnostics* const diagnostics, const CenterSizeFactorsOptions& options) {
    const auto mean = center_size_factors_mean(num, size_factors, diagnostics, options);
    if (mean) {
        for (I<decltype(num)> i = 0; i < num; ++i){
            size_factors[i] /= mean;
        }
    }
    return mean;
}

/**
 * Compute the mean size factor for each block, but do not scale the size factors themselves.
 * This is the blocked version of `center_size_factors_mean()`.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param num Number of cells.
 * @param[in] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * @param[in] block Pointer to an array of length `num`, containing the block assignment for each cell.
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
 * @param[out] diagnostics Diagnostics for invalid size factors.
 * This is only used if `CenterSizeFactorsOptions::ignore_invalid = true`, in which case it is filled with diagnostics for invalid values in `size_factors`.
 * It can also be `NULL`, in which case it is ignored.
 * @param options Further options.
 *
 * @return Vector of length \f$N\f$ containing the mean size factor for each block,
 * to be used to scale the size factors in each block.
 */
template<typename SizeFactor_, typename Block_>
std::vector<SizeFactor_> center_size_factors_blocked_mean(
    const std::size_t num,
    const SizeFactor_* const size_factors,
    const Block_* const block,
    SizeFactorDiagnostics* const diagnostics,
    const CenterSizeFactorsOptions& options)
{
    static_assert(std::is_floating_point<SizeFactor_>::value);
    const auto ngroups = tatami_stats::total_groups(block, num);
    auto group_mean = sanisizer::create<std::vector<SizeFactor_> >(ngroups);
    auto group_num = sanisizer::create<std::vector<I<decltype(num)>> >(ngroups);

    if (options.ignore_invalid) {
        SizeFactorDiagnostics tmpdiag;
        auto& diag = (diagnostics == NULL ? tmpdiag : *diagnostics);
        for (I<decltype(num)> i = 0; i < num; ++i) {
            const auto val = size_factors[i];
            if (!internal::is_invalid(val, diag)) {
                const auto b = block[i];
                group_mean[b] += val;
                ++(group_num[b]);
            }
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
 * Center size factors within each block to obtain interpretable values after normalization, as discussed in `center_size_factors()`.
 * The exact strategy for handling blocks is controlled by `CenterSizeFactorsOptions::block_mode`.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param num Number of cells.
 * @param[in] size_factors Pointer to an array of length `num`, containing the size factor for each cell.
 * On output, this contains size factors that are centered according to `CenterSizeFactorsOptions::block_mode`.
 * @param[in] block Pointer to an array of length `num`, containing the block assignment for each cell.
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
 * @param[out] diagnostics Diagnostics for invalid size factors.
 * This is only used if `CenterSizeFactorsOptions::ignore_invalid = true`, in which case it is filled with invalid diagnostics for values in `size_factors`.
 * It can also be NULL, in which case it is ignored.
 * @param options Further options.
 *
 * @return Vector of length \f$N\f$ containing the mean size factor for each block.
 */
template<typename SizeFactor_, typename Block_>
std::vector<SizeFactor_> center_size_factors_blocked(
    const std::size_t num,
    SizeFactor_* const size_factors,
    const Block_* const block,
    SizeFactorDiagnostics* const diagnostics,
    const CenterSizeFactorsOptions& options)
{
    const auto group_mean = center_size_factors_blocked_mean(num, size_factors, block, diagnostics, options);

    if (options.block_mode == CenterBlockMode::PER_BLOCK) {
        for (I<decltype(num)> i = 0; i < num; ++i) {
            const auto& div = group_mean[block[i]];
            if (div) {
                size_factors[i] /= div;
            }
        }

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

        if (min > 0) {
            for (I<decltype(num)> i = 0; i < num; ++i) {
                size_factors[i] /= min;
            }
        }
    }

    return group_mean;
}

}

#endif
