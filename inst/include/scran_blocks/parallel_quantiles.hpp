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

/**
 * @file parallel_quantiles.hpp
 * @brief Quantile of parallel elements across arrays.
 */

namespace scran_blocks {

/**
 * @brief Calculate a single quantile from a container.
 *
 * @tparam Output_ Floating-point type of the output quantile.
 * @tparam Iterator_ Random-access writeable iterator to a container.
 *
 * This class should be constructed once, given the number of elements and the desired probability.
 * It can then be used to compute a quantile for containers of size equal to the pre-specified number of elements. 
 * The quantile is type 7, consistent with the default in R's `quantile` function. 
 */
template<typename Output_, class Iterator_>
class SingleQuantile {
public:
    /**
     * @param len Number of elements in the container, should be positive.
     * @param quantile Quantile to compute, in \f$[0, 1]\f$.
     */
    SingleQuantile(const std::size_t len, const double quantile) {
        assert(len > 0);
        sanisizer::can_ptrdiff<Iterator_>(len);
        const Output_ frac = static_cast<Output_>(len - 1) * static_cast<Output_>(quantile);
        const Output_ base = std::floor(frac);
        my_lower = base; // cast is known-safe if can_ptrdiff passes and 0 <= quantile <= 1.
        my_upper_fraction = frac - base;
        my_lower_fraction = static_cast<Output_>(1) - my_upper_fraction;
        my_skip_upper = my_upper_fraction == 0;
    }

private:
    std::size_t my_lower;
    Output_ my_upper_fraction;
    Output_ my_lower_fraction;
    bool my_skip_upper;

public:
    /**
     * @param begin Start of the container.
     * @param end End of the container.
     *
     * @return Quantile for the sequence of elements in `[begin, end)`.
     *
     * The range `[begin, end)` should have length equal to `n`, and should not contain any NaN values.
     * On output, the order of elements in `[begin, end)` may be rearranged.
     */
    Output_ operator()(Iterator_ begin, Iterator_ end) const {
        assert(sanisizer::is_less_than(my_lower, end - begin));
        auto target = begin + my_lower;
        std::nth_element(begin, target, end);

        // Avoid looking at target+1 if don't need it - in particular, we'd get a out-of-bounds access if quantile = 1.
        if (my_skip_upper) {
            return *target;
        } else {
            const auto next = std::min_element(target + 1, end); 
            return static_cast<Output_>(*target) * my_lower_fraction + static_cast<Output_>(*next) * my_upper_fraction;
        }
    }
};

/**
 * @brief Calculate a single quantile for containers of variable length.
 *
 * @tparam Output_ Floating-point type of the output quantile.
 * @tparam Iterator_ Random-access writeable iterator to a container.
 *
 * This class should be constructed once, given the maximum number of elements and the desired probability.
 * It can then be used to compute quantiles from containers of size less than or equal to the maximum.
 * See `SingleQuantile` for more details.
 */
template<typename Output_, class Iterator_>
class SingleQuantileVariable {
public:
    /**
     * @param max_len Maximum number of elements in the container.
     * Unlike `SingleQuantile`, this may be zero.
     * @param quantile Quantile to compute, in \f$[0, 1]\f$.
     */
    SingleQuantileVariable(const std::size_t max_len, const double quantile) : my_quantile(quantile) {
        if (max_len >= 2) {
            sanisizer::resize(my_choices, max_len - 1);
        }
    }

private:
    std::vector<std::optional<SingleQuantile<Output_, Iterator_> > > my_choices;
    double my_quantile;

public:
    /**
     * @param len Length of the container.
     * This should be equal to `end - begin` and no greater than `max_len`.
     * @param begin Start of the container.
     * @param end End of the container.
     *
     * @return Quantile for the sequence of elements in `[begin, end)`.
     * If this sequence is empty, NaN is returned.
     *
     * The range `[begin, end)` should have length equal to `len`, and should not contain any NaN values.
     * On output, the order of elements in `[begin, end)` may be rearranged.
     *
     * This method is not thread-safe.
     */
    Output_ operator()(const std::size_t len, Iterator_ begin, Iterator_ end) {
        assert(sanisizer::is_equal(len, end - begin));
        if (len == 0) {
            return std::numeric_limits<Output_>::quiet_NaN();
        } else if (len == 1) {
            return *begin;
        } else {
            auto& ocalc = my_choices[len - 2];
            if (!ocalc.has_value()) { // Instantiating them on demand.
                ocalc.emplace(len, my_quantile);
            }
            return (*ocalc)(begin, end);
        }
    }

    /**
     * @param begin Start of the container.
     * @param end End of the container.
     *
     * @return Quantile for the sequence of elements in `[begin, end)`.
     * If this sequence is empty, NaN is returned.
     *
     * The range `[begin, end)` should have length less than `max_length`, and should not contain any NaN values.
     * On output, the order of elements in `[begin, end)` may be rearranged.
     *
     * This method is not thread-safe.
     */
    Output_ operator()(Iterator_ begin, Iterator_ end) {
        return this->operator()(end - begin, begin, end);
    }
};

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
        SingleQuantileVariable<Output_, I<decltype(tmp_buffer.begin())> > calcs(nblocks, quantile);
        for (std::size_t g = 0; g < n; ++g) {
            tmp_buffer.clear();
            for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                const auto val = in[b][g];
                if (!std::isnan(val)) {
                    tmp_buffer.push_back(val);
                }
            }
            out[g] = calcs(tmp_buffer.size(), tmp_buffer.begin(), tmp_buffer.end());
        }

    } else {

        SingleQuantile<Output_, I<decltype(tmp_buffer.begin())> > calc(nblocks, quantile);
        for (std::size_t g = 0; g < n; ++g) {
            tmp_buffer.clear();
            for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                tmp_buffer.push_back(in[b][g]);
            }
            out[g] = calc(tmp_buffer.begin(), tmp_buffer.end());
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

