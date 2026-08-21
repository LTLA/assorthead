#ifndef TATAMI_STATS_MEDIAN_HPP
#define TATAMI_STATS_MEDIAN_HPP

#include "utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "quickstats/quickstats.hpp"

/**
 * @file median.hpp
 *
 * @brief Compute row and column medians from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `median()`.
 */
struct MedianOptions {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the median calculation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing medians across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<typename Output_ = double, typename Value_, typename Index_>
Output_ median_direct(Value_* ptr, Index_ num, bool skip_nan) {
    nanable_ifelse<Value_>(
        skip_nan,
        [&]() -> void {
            num = shift_nans(ptr, num);
        },
        []() -> void {}
    );

    return quickstats::median<Output_>(num, ptr);
}

template<typename Output_ = double, typename Value_, typename Index_>
Output_ median_direct(Value_* value, Index_ num_nonzero, Index_ num_all, bool skip_nan) {
    nanable_ifelse<Value_>(
        skip_nan,
        [&]() -> void {
            auto new_nonzero = shift_nans(value, num_nonzero);
            num_all -= num_nonzero - new_nonzero;
            num_nonzero = new_nonzero;
        },
        []() -> void {}
    );

    return quickstats::median<Output_>(num_all, num_nonzero, value);
}
/**
 * @endcond
 */

/**
 * Compute medians for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the input values.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Floating-point type of the output value.
 * This should be capable of storing NaNs.
 *
 * @param row Whether to compute the median for each row.
 * If false, the median is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column medians.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_>
void median(const bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* const output, const MedianOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.sparse()) {
        tatami::Options topt;
        topt.sparse_extract_index = false;
        topt.sparse_ordered_index = false; // we'll be sorting by value anyway.

        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, topt);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto vbuffer = buffer.data();
            for (Index_ x = 0; x < l; ++x) {
                auto range = ext->fetch(vbuffer, NULL);
                tatami::copy_n(range.value, range.number, vbuffer);
                output[x + s] = median_direct<Output_>(vbuffer, range.number, otherdim, opt.skip_nan);
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
            for (Index_ x = 0; x < l; ++x) {
                auto ptr = ext->fetch(buffer.data());
                tatami::copy_n(ptr, otherdim, buffer.data());
                output[x + s] = median_direct<Output_>(buffer.data(), otherdim, opt.skip_nan);
            }
        }, dim, opt.num_threads);
    }
}

/**
 * Overload of `median()` that allocates memory for the output medians.
 *
 * @tparam Output_ Floating-point type of the output value.
 * This should be capable of storing NaNs.
 * @tparam Value_ Numeric type of the input values.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param row Whether to compute the median for each row.
 * If false, the median is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param opt Further options.
 *
 * @return Vector of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column medians.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> median(const bool row, const tatami::Matrix<Value_, Index_>& mat, const MedianOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    auto output = sanisizer::create<std::vector<Output_> >(dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    median(row, mat, output.data(), opt);
    return output;
}

}

#endif
