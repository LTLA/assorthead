#ifndef TATAMI_STATS_QUANTILE_HPP
#define TATAMI_STATS_QUANTILE_HPP

#include "utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "quickstats/quickstats.hpp"

/**
 * @file quantile.hpp
 *
 * @brief Compute row and column quantiles from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `quantile()`.
 */
struct QuantileOptions {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the quantile calculation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing quantiles across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * Compute quantiles for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the input values.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Floating-point type of the output value.
 * This should be capable of storing NaNs.
 *
 * @param row Whether to compute the quantile for each row.
 * If false, the quantile is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param prob Probability of the quantile to compute.
 * This should be in \f$[0, 1]\f$.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column quantiles.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_>
void quantile(
    const bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const double prob,
    Output_* const output,
    const QuantileOptions& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    if (otherdim == 0) {
        // Prevent initalize() from constructing with a fixed instance with otherdim == 0.
        std::fill_n(output, dim, std::numeric_limits<Output_>::quiet_NaN());
        return;
    }

    tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
        std::optional<quickstats::SingleQuantileFixedNumber<Output_> > qcalcs_fixed;
        std::optional<quickstats::SingleQuantileVariableNumber<Output_> > qcalcs_var;
        // Index_ is safe to cast to std::size_t as that's part of the tatami contract.
        nanable_ifelse<Value_>(
            opt.skip_nan,
            [&]() -> void {
                qcalcs_var.emplace(otherdim, prob);
            },
            [&]() -> void {
                qcalcs_fixed.emplace(otherdim, prob);
            }
        );

        if (mat.sparse()) {
            tatami::Options topt;
            topt.sparse_extract_index = false;
            topt.sparse_ordered_index = false; // we'll be sorting by value anyway.
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, topt);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto vbuffer = buffer.data();

            for (Index_ x = 0; x < l; ++x) {
                auto range = ext->fetch(vbuffer, NULL);
                tatami::copy_n(range.value, range.number, vbuffer);

                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        const auto new_non_zeros = shift_nans(vbuffer, range.number);
                        output[x + s] = (*qcalcs_var)(otherdim - (range.number - new_non_zeros), new_non_zeros, vbuffer);
                    },
                    [&]() -> void {
                        output[x + s] = (*qcalcs_fixed)(range.number, vbuffer);
                    }
                );
            }

        } else {
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);

            for (Index_ x = 0; x < l; ++x) {
                auto bufptr = buffer.data();
                auto raw = ext->fetch(bufptr);
                tatami::copy_n(raw, otherdim, bufptr);

                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        const auto new_total = shift_nans(bufptr, otherdim);
                        output[x + s] = (*qcalcs_var)(new_total, bufptr);
                    },
                    [&]() -> void {
                        output[x + s] = (*qcalcs_fixed)(bufptr);
                    }
                );
            }
        }
    }, dim, opt.num_threads);
}

/**
 * Overload of `quantile()` that allocates memory for the output quantiles.
 *
 * @tparam Value_ Numeric type of the input values.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Floating-point type of the output value.
 * This should be capable of storing NaNs.
 *
 * @param row Whether to compute the quantile for each row.
 * If false, the quantile is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param prob Probability of the quantile to compute.
 * This should be in \f$[0, 1]\f$.
 * @param opt Further options.
 *
 * @return Vector of length equal to the number of rows (if `row = true`) or columns (otherwise),
 * containing the row/column quantiles.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> quantile(
    const bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const double prob,
    const QuantileOptions& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    auto output = sanisizer::create<std::vector<Output_> >(dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    quantile(row, mat, prob, output.data(), opt);
    return output;
}

}

#endif
