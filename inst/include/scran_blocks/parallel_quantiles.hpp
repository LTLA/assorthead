#ifndef SCRAN_BLOCKS_PARALLEL_QUANTILES_HPP
#define SCRAN_BLOCKS_PARALLEL_QUANTILES_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <cmath>
#include <optional>
#include <cassert>

#include "utils.hpp"

#include "sanisizer/sanisizer.hpp"
#include "quickstats/quickstats.hpp"

/**
 * @file parallel_quantiles.hpp
 * @brief Quantile of parallel elements across arrays.
 */

namespace scran_blocks {

/**
 * Compute the quantile for parallel elements across multiple arrays.
 * This can be used as an alternative to `parallel_means()` to summarize statistics across blocks, e.g., by computing the median with `quantile = 0.5`.
 * The quantile is type 7, consistent with the default in R's `quantile` function. 
 *
 * @tparam Stat_ Type of the input statistic, typically floating point.
 * @tparam Output_ Floating-point type of the output quantile.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param quantile Quantile to compute, in \f$[0, 1]\f$.
 * @param[out] out Pointer to an output array of length `n`.
 * On completion, `out[i]` is filled with the quantile of `(in.front()[i], in[1][i], ..., in.back()[i])`.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are removed before computing the quantile.
 * If `false`, it is assumed that no NaNs are present.
 */
template<typename Stat_, typename Output_>
void parallel_quantiles(const std::size_t n, const std::vector<Stat_*>& in, const double quantile, Output_* const out, const bool skip_nan) {
    const auto nblocks = in.size();
    if (nblocks == 0) {
        std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
        return;
    } else if (nblocks == 1) {
        std::copy_n(in[0], n, out);
        return;
    }

    std::vector<Stat_> tmp_buffer;
    tmp_buffer.reserve(nblocks);

    if (skip_nan) {
        quickstats::SingleQuantileVariableNumber<Output_> calcs(sanisizer::cast<std::size_t>(nblocks), quantile);
        for (std::size_t g = 0; g < n; ++g) {
            tmp_buffer.clear();
            for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                const auto val = in[b][g];
                if (!std::isnan(val)) {
                    tmp_buffer.push_back(val);
                }
            }
            out[g] = calcs(tmp_buffer.size(), tmp_buffer.data());
        }

    } else {
        quickstats::SingleQuantileFixedNumber<Output_> calc(sanisizer::cast<std::size_t>(nblocks), quantile);
        for (std::size_t g = 0; g < n; ++g) {
            tmp_buffer.clear();
            for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                tmp_buffer.push_back(in[b][g]);
            }
            out[g] = calc(tmp_buffer.data());
        }
    }
}

/**
 * Overload of `parallel_quantiles()` that allocates memory for the output array.
 *
 * @tparam Output_ Floating-point type of the output quantile.
 * @tparam Stat_ Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param quantile Quantile to compute, in \f$[0, 1]\f$.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are removed before computing the quantile.
 * If `false`, it is assumed that no NaNs are present.
 *
 * @return Vector of length `n`, where the `i`-th element is the quantile of `(in.front()[i], in[1][i], ..., in.back()[i])`.
 */
template<typename Output_ = double, typename Stat_>
std::vector<Output_> parallel_quantiles(const std::size_t n, const std::vector<Stat_*>& in, const double quantile, const bool skip_nan) {
    auto out = sanisizer::create<std::vector<Output_> >(n);
    parallel_quantiles(n, in, quantile, out.data(), skip_nan);
    return out;
}

}

#endif

