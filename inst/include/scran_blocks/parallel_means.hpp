#ifndef SCRAN_BLOCKS_PARALLEL_MEANS_HPP
#define SCRAN_BLOCKS_PARALLEL_MEANS_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file parallel_means.hpp
 * @brief Mean of arallel elements across arrays.
 */

namespace scran_blocks {

/**
 * @cond
 */
template<bool weighted_, typename Stat_, typename Weight_, typename Output_>
void parallel_means_internal(const std::size_t n, std::vector<Stat_*> in, const Weight_* const w, Output_* const out, const bool skip_nan) {
    const auto nblocks = in.size();
    if (nblocks == 0) {
        std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
        return;
    } else if (nblocks == 1) {
        if constexpr(weighted_) {
            if (w[0] == 0) {
                std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
                return;
            }
        } 
        std::copy(in[0], in[0] + n, out);
        return;
    }

    if (skip_nan) {
        for (I<decltype(n)> i = 0; i < n; ++i) {
            Output_ prod = 0;
            Output_ denom = 0;

            for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                const auto val = in[b][i];
                if (!std::isnan(val)) {
                    if constexpr(weighted_) {
                        const auto curw = w[b];
                        prod += val * curw;
                        denom += curw;
                    } else {
                        prod += val;
                        denom += 1;
                    }
                }
            }

            out[i] = prod/denom;
        }

    } else {
        std::fill_n(out, n, 0);
        auto wcopy = w;
        for (const auto current : in) {
            if constexpr(weighted_) {
                const Weight_ weight = *(wcopy++);
                if (weight != 1) { 
                    for (I<decltype(n)> i = 0; i < n; ++i) {
                        out[i] += current[i] * weight;
                    }
                    continue;
                }
            }
            for (I<decltype(n)> i = 0; i < n; ++i) {
                out[i] += current[i];
            }
        }

        const Output_ denom = [&]() {
            if constexpr(weighted_) {
                return std::accumulate(w, w + in.size(), static_cast<Output_>(0));
            } else {
                return in.size();
            }
        }();
        for (I<decltype(n)> i = 0; i < n; ++i) {
            out[i] /= denom;
        }
    }
}
/**
 * @endcond
 */

/**
 * Mean of parallel elements across multiple arrays.
 * This is equivalent to calling `parallel_weighted_means()` with equal weights for each array.
 *
 * @tparam Stat_ Type of the input statistic, typically floating point.
 * @tparam Output_ Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[out] out Pointer to an output array of length `n`.
 * On completion, `out[i]` is filled with the mean of `(in.front()[i], in[1][i], ..., in.back()[i])`.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are removed before computing the mean.
 * If `false`, it is assumed that no NaNs are present.
 */
template<typename Stat_, typename Output_>
void parallel_means(const std::size_t n, std::vector<Stat_*> in, Output_* const out, const bool skip_nan) {
    parallel_means_internal<false>(n, std::move(in), static_cast<int*>(NULL), out, skip_nan);
    return;
}

/**
 * Overload of `parallel_means()` that allocates an output vector of averaged values.
 *
 * @tparam Output Floating-point output type.
 * @tparam Stat Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are removed before computing the mean.
 * If `false`, it is assumed that no NaNs are present.
 *
 * @return Vector of length `n`, where the `i`-th element is the mean of `(in.front()[i], in[1][i], ..., in.back()[i])`.
 */
template<typename Output_ = double, typename Stat_>
std::vector<Output_> parallel_means(const std::size_t n, std::vector<Stat_*> in, const bool skip_nan) {
    auto out = sanisizer::create<std::vector<Output_> >(n);
    parallel_means(n, std::move(in), out.data(), skip_nan);
    return out;
}

/**
 * Compute a weighted average of parallel elements across multiple arrays.
 * For example, we can average statistics across blocks using weights computed with `compute_weights()`.
 *
 * @tparam Stat_ Type of the input statistic, typically floating point.
 * @tparam Weight_ Type of the weight, typically floating point.
 * @tparam Output_ Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 * Weights should be non-negative and finite.
 * @param[out] out Pointer to an output array of length `n`.
 * On completion, `out[i]` is filled with the weighted mean of `(in.front()[i], in[1][i], ..., in.back()[i])`.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are removed before computing the mean.
 * If `false`, it is assumed that no NaNs are present.
 */
template<typename Stat_, typename Weight_, typename Output_>
void parallel_weighted_means(const std::size_t n, std::vector<Stat_*> in, const Weight_* const w, Output_* const out, const bool skip_nan) {
    if (!in.empty()) {
        bool same = true;
        const auto numin = in.size();
        for (I<decltype(numin)> i = 1; i < numin; ++i) {
            if (w[i] != w[0]) {
                same = false;
                break;
            }
        }

        if (same) {
            if (w[0] == 0) {
                std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
            } else {
                parallel_means(n, std::move(in), out, skip_nan);
            }
            return;
        }
    }

    parallel_means_internal<true>(n, std::move(in), w, out, skip_nan);
    return;
}

/**
 * Overload of `parallel_weighted_means()` that allocates an output vector of averaged values.
 *
 * @tparam Output_ Floating-point output type.
 * @tparam Weight_ Type of the weight, typically floating point.
 * @tparam Stat_ Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 * Weights should be non-negative and finite.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are removed before computing the mean.
 * If `false`, it is assumed that no NaNs are present.
 *
 * @return Vector of length `n`, where the `i`-th element is the weighted mean of `(in.front()[i], in[1][i], ..., in.back()[i])`.
 */
template<typename Output_ = double, typename Stat_, typename Weight_>
std::vector<Output_> parallel_weighted_means(const std::size_t n, std::vector<Stat_*> in, const Weight_* const w, const bool skip_nan) {
    auto out = sanisizer::create<std::vector<Output_> >(n);
    parallel_weighted_means(n, std::move(in), w, out.data(), skip_nan);
    return out;
}

/**
 * @cond
 */
// Methods for back-compatibility.
template<typename Stat_, typename Output_>
void average_vectors(const std::size_t n, std::vector<Stat_*> in, Output_* const out, const bool skip_nan) {
    parallel_means(n, in, out, skip_nan);
}

template<typename Output_ = double, typename Stat_>
std::vector<Output_> average_vectors(const std::size_t n, std::vector<Stat_*> in, const bool skip_nan) {
    return parallel_means(n, in, skip_nan);
}

template<typename Stat_, typename Weight_, typename Output_>
void average_vectors_weighted(const std::size_t n, std::vector<Stat_*> in, const Weight_* const w, Output_* const out, const bool skip_nan) {
    parallel_weighted_means(n, in, w, out, skip_nan);
}

template<typename Output_ = double, typename Stat_, typename Weight_>
std::vector<Output_> average_vectors_weighted(const std::size_t n, std::vector<Stat_*> in, const Weight_* const w, const bool skip_nan) {
    return parallel_weighted_means(n, in, w, skip_nan);
}
/**
 * @endcond
 */

}

#endif
