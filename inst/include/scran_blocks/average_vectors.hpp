#ifndef SCRAN_BLOCKS_AVERAGE_VECTORS_HPP
#define SCRAN_BLOCKS_AVERAGE_VECTORS_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

/**
 * @file average_vectors.hpp
 * @brief Average parallel elements across vectors.
 */

namespace scran_blocks {

/**
 * @cond
 */
namespace internal {

template<bool weighted_, typename Stat_, typename Weight_, typename Output_>
void average_vectors(std::size_t n, std::vector<Stat_*> in, const Weight_* w, Output_* out, bool skip_nan) {
    if (in.empty()) {
        std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
        return;
    } else if (in.size() == 1) {
        if constexpr(weighted_) {
            if (w[0] == 0) {
                std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
                return;
            }
        } 
        std::copy(in[0], in[0] + n, out);
        return;
    }

    std::fill_n(out, n, 0);
    std::vector<Weight_> accumulated(skip_nan ? n : 0);

    auto wcopy = w;
    for (auto current : in) {
        if constexpr(weighted_) {
            // Don't skip if weight = 0, as we need to still compute the product, e.g.,
            // if the value is Inf, we'd end up with 0 * Inf => NaN that can't be skipped.
            Weight_ weight = *(wcopy++);

            // Use the other loop and skip an unnecessary multiplication when the weight is 1.
            if (weight != 1) { 
                if (skip_nan) {
                    for (decltype(n) i = 0; i < n; ++i) {
                        auto x = current[i] * weight;
                         if (!std::isnan(x)) {
                            out[i] += x; 
                            accumulated[i] += weight;
                         }
                    }
                } else {
                    for (decltype(n) i = 0; i < n; ++i) {
                        out[i] += current[i] * weight;
                    }
                }
                continue;
            }
        }

        if (skip_nan) {
            for (decltype(n) i = 0; i < n; ++i) {
                auto x = current[i];
                if (!std::isnan(x)) {
                    out[i] += x; 
                    ++accumulated[i];
                }
            }
        } else {
            for (decltype(n) i = 0; i < n; ++i) {
                out[i] += current[i];
            }
        }
    }

    if (skip_nan) {
        for (decltype(n) i = 0; i < n; ++i) {
            out[i] /= accumulated[i];
        }
    } else {
        double denom = 1;
        if constexpr(weighted_) {
            denom /= std::accumulate(w, w + in.size(), 0.0);
        } else {
            denom /= in.size();
        }
        for (decltype(n) i = 0; i < n; ++i) {
            out[i] *= denom;
        }
    }
}

}
/**
 * @endcond
 */

/**
 * Average parallel elements across multiple arrays.
 *
 * @tparam Stat_ Type of the input statistic, typically floating point.
 * @tparam Output_ Floating-point output type.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param[out] out Pointer to an output array of length `n`.
 * On completion, `out` is filled with the average of all arrays in `in`.
 * Specifically, each element of `out` is set to the average of the corresponding elements across all `in` arrays.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are ignored in the average calculations for each element, at the cost of some efficiency.
 */
template<typename Stat_, typename Output_>
void average_vectors(size_t n, std::vector<Stat_*> in, Output_* out, bool skip_nan) {
    internal::average_vectors<false>(n, std::move(in), static_cast<int*>(NULL), out, skip_nan);
    return;
}

/**
 * Overload of `compute()` that allocates an output vector of averaged values.
 *
 * @tparam Output Floating-point output type.
 * @tparam Stat Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of length `n`.
 * @param skip_nan Whether to check for NaNs.
 *
 * @return A vector of length `n` is returned, containing the average of all arrays in `in`.
 */
template<typename Output_ = double, typename Stat_>
std::vector<Output_> average_vectors(size_t n, std::vector<Stat_*> in, bool skip_nan) {
    auto out = sanisizer::create<std::vector<Output_> >(n);
    average_vectors(n, std::move(in), out.data(), skip_nan);
    return out;
}

/**
 * Compute a weighted average of parallel elements across multiple arrays.
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
 * On output, `out` is filled with the weighted average of all arrays in `in`.
 * Specifically, each element of `out` is set to the weighted average of the corresponding elements across all `in` arrays.
 * @param skip_nan Whether to check for NaNs.
 * If `true`, NaNs are ignored in the average calculations for each element, at the cost of some efficiency.
 */
template<typename Stat_, typename Weight_, typename Output_>
void average_vectors_weighted(size_t n, std::vector<Stat_*> in, const Weight_* w, Output_* out, bool skip_nan) {
    if (!in.empty()) {
        bool same = true;
        auto numin = in.size();
        for (decltype(numin) i = 1; i < numin; ++i) {
            if (w[i] != w[0]) {
                same = false;
                break;
            }
        }

        if (same) {
            if (w[0] == 0) {
                std::fill_n(out, n, std::numeric_limits<Output_>::quiet_NaN());
            } else {
                average_vectors(n, std::move(in), out, skip_nan);
            }
            return;
        }
    }

    internal::average_vectors<true>(n, std::move(in), w, out, skip_nan);
    return;
}

/**
 * Overload of `compute_weighted()` that allocates an output vector of averaged values.
 *
 * @tparam Output_ Floating-point output type.
 * @tparam Weight_ Type of the weight, typically floating point.
 * @tparam Stat_ Type of the input statistic, typically floating point.
 *
 * @param n Length of each array.
 * @param[in] in Vector of pointers to input arrays of the same length.
 * @param[in] w Pointer to an array of length equal to `in.size()`, containing the weight to use for each input array.
 * Weights should be non-negative and finite.
 * @param skip_nan Whether to check for NaNs.
 *
 * @return A vector is returned containing with the average of all arrays in `in`.
 */
template<typename Output_ = double, typename Stat_, typename Weight_>
std::vector<Output_> average_vectors_weighted(std::size_t n, std::vector<Stat_*> in, const Weight_* w, bool skip_nan) {
    auto out = sanisizer::create<std::vector<Output_> >(n);
    average_vectors_weighted(n, std::move(in), w, out.data(), skip_nan);
    return out;
}

}

#endif
