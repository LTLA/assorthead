#ifndef TATAMI_STATS_VARIANCE_HPP
#define TATAMI_STATS_VARIANCE_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "quickstats/quickstats.hpp"

#include "rss.hpp"
#include "skip_nan/rss.hpp"
#include "utils.hpp"

/**
 * @file variance.hpp
 *
 * @brief Compute row and column variances from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `variance()`.
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct VarianceOptions {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the variance calculation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing variances across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;

    /**
     * Placeholder value to use for the mean when the extent of the relevant dimension is zero.
     * This is NaN if supported by `Output_`, otherwise zero.
     */
    Output_ mean_placeholder = quickstats::nan_if_available_else_zero<Output_>();

    /**
     * Placeholder value to use for the variance when the extent of the relevant dimension is less than 2.
     * This is NaN if supported by `Output_`, otherwise zero.
     */
    Output_ variance_placeholder = quickstats::nan_if_available_else_zero<Output_>();
};

/**
 * @brief Result buffers for `variance()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_>
struct VarianceBuffers {
    /**
     * Pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `variance()`, this is filled with the sample mean of each row/column.
     */
    Output_* mean;

    /**
     * Pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `variance()`, this is filled with the sample variance of each row/column.
     */
    Output_* variance;
};

/**
 * Compute sample variances for each element of a chosen dimension of a `tatami::Matrix`.
 * This may use either Welford's method or the standard two-pass method,
 * depending on the dimension in `row` and the preferred access dimension of `p`.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Floating-point type of the output data.
 *
 * @param row Whether to compute the variance for each row.
 * If false, the variance is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Buffers to output arrays.
 * On output, this will contain the row/column variances.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_>
void variance(bool row, const tatami::Matrix<Value_, Index_>& mat, VarianceBuffers<Output_>& output, const VarianceOptions<Output_>& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    nanable_ifelse<Value_>(
        opt.skip_nan,

        [&]() -> void {
            skip_nan::RssBuffers<Output_, Index_> tmp;
            tmp.mean = output.mean;
            tmp.rss = output.variance;
            auto count = tatami::create_container_of_Index_size<std::vector<Index_> >(row ? mat.nrow() : mat.ncol());
            tmp.count = count.data();

            skip_nan::RssOptions ropt;
            ropt.num_threads = opt.num_threads;
            ropt.mean_placeholder = opt.mean_placeholder;
            skip_nan::rss(row, mat, tmp, ropt);

            for (Index_ i = 0; i < dim; ++i) {
                if (count[i] <= 1) {
                    output.variance[i] = opt.variance_placeholder;
                } else {
                    output.variance[i] /= count[i] - 1;
                }
            }
        },

        [&]() -> void {
            RssBuffers<Output_> tmp;
            tmp.mean = output.mean;
            tmp.rss = output.variance;

            RssOptions ropt;
            ropt.num_threads = opt.num_threads;
            ropt.mean_placeholder = opt.mean_placeholder;
            rss(row, mat, tmp, ropt);

            if (otherdim <= 1) {
                std::fill_n(output.variance, dim, opt.variance_placeholder);
            } else {
                for (Index_ i = 0; i < dim; ++i) {
                    output.variance[i] /= otherdim - 1;
                }
            }
        }
    );
}

/**
 * @brief Results of `variance()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_>
struct VarianceResult {
    /**
     * Vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample mean of each row/column.
     */
    std::vector<Output_> mean;

    /**
     * Vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample variance of each row/column.
     */
    std::vector<Output_> variance;
};

/**
 * Overload of `variance()` that allocates memory for the output arrays.
 *
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param row Whether to compute the variance for each row.
 * If false, the variance is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param opt Further options.
 *
 * @return The mean and variance of each row/column.
 */
template<typename Output_ = double, typename Value_, typename Index_>
VarianceResult<Output_> variance(bool row, const tatami::Matrix<Value_, Index_>& mat, const VarianceOptions<Output_>& opt) {
    VarianceResult<Output_> output;
    const auto dim = (row ? mat.nrow() : mat.ncol());
    tatami::resize_container_to_Index_size(output.mean, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    tatami::resize_container_to_Index_size(output.variance, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );

    VarianceBuffers<Output_> buffers;
    buffers.mean = output.mean.data();
    buffers.variance = output.variance.data();

    variance(row, mat, buffers, opt);
    return output;
}

}

#endif
