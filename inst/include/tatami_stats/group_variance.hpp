#ifndef TATAMI_STATS_GROUP_VARIANCE_HPP
#define TATAMI_STATS_GROUP_VARIANCE_HPP

#include <vector>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <cassert>
#include <limits>
#include <cmath>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "group_rss.hpp"
#include "skip_nan/group_rss.hpp"
#include "utils.hpp"

/**
 * @file group_variance.hpp
 *
 * @brief Compute group-wise variances from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `group_variance()`.
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct GroupVarianceOptions {
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
 * @brief Result buffers for `group_variance()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_>
struct GroupVarianceBuffers {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `group_variance()`, this is filled with the sample mean of each row/column for the corresponding group.
     */
    std::vector<Output_*> mean;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `group_variance()`, this is filled with the sample variance of each row/column for the corresponding group.
     */
    std::vector<Output_*> variance;
};

/**
 * Compute per-group variances for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 * @tparam Count_ Numeric type of the group sizes, typically integer.
 * @tparam Output_ Floating-point type of the output value.
 *
 * @param row Whether to compute variances for the rows.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * @param[in] group_size Pointer to an array of length equal to `num_groups`, containing the size of each group.
 * @param[out] output Buffers in which to store the results.
 * On output, each array stores the means and variances of the corresponding group.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Group_, typename Count_, typename Output_>
void group_variance(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    const Count_* const group_size,
    GroupVarianceBuffers<Output_>& output,
    const GroupVarianceOptions<Output_>& opt
) {
    assert(sanisizer::is_equal(num_groups, output.mean.size()));
    assert(sanisizer::is_equal(num_groups, output.variance.size()));
    const auto dim = (row ? mat.nrow() : mat.ncol());

    nanable_ifelse<Value_>(
        opt.skip_nan,

        [&]() -> void {
            skip_nan::GroupRssBuffers<Output_, Index_> tmp;
            tmp.mean = output.mean;
            tmp.rss = output.variance;

            auto count = sanisizer::create<std::vector<std::vector<Index_> > >(num_groups);
            tmp.count.reserve(num_groups);
            for (std::size_t g = 0; g < num_groups; ++g) {
                tatami::resize_container_to_Index_size(count[g], dim);
                tmp.count.push_back(count[g].data());
            }

            skip_nan::GroupRssOptions ropt;
            ropt.num_threads = opt.num_threads;
            skip_nan::group_rss(row, mat, group, num_groups, tmp, ropt);
            for (std::size_t g = 0; g < num_groups; ++g) {
                const auto outvar = output.variance[g];
                const auto curcounts = count[g];
                for (Index_ d = 0; d < dim; ++d) {
                    if (curcounts[d] <= 1) {
                        outvar[d] = opt.variance_placeholder;
                    } else {
                        outvar[d] /= curcounts[d] - 1;
                    }
                }
            }
        },

        [&]() -> void {
            GroupRssBuffers<Output_> tmp;
            tmp.mean = output.mean;
            tmp.rss = output.variance;

            GroupRssOptions ropt;
            ropt.num_threads = opt.num_threads;
            group_rss(row, mat, group, num_groups, group_size, tmp, ropt);

            for (std::size_t g = 0; g < num_groups; ++g) {
                const auto outvar = output.variance[g];
                const auto gsize = group_size[g];
                if (gsize <= 1) {
                    std::fill_n(outvar, dim, opt.variance_placeholder);
                } else {
                    for (Index_ d = 0; d < dim; ++d) {
                        outvar[d] /= gsize - 1;
                    }
                }
            }
        }
    );
}

/**
 * Overload that computes the group sizes before calling `group_variance()`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 * @tparam Output_ Floating-point type of the output value.
 *
 * @param row Whether to compute variances for the rows.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * @param[out] output Buffers in which to store the results.
 * On output, each array stores the means and variances of the corresponding group.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Group_, typename Output_>
void group_variance(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    GroupVarianceBuffers<Output_>& output,
    const GroupVarianceOptions<Output_>& opt
) {
    auto group_size = sanisizer::create<std::vector<Index_> >(num_groups);
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    for (Index_ o = 0; o < otherdim; ++o) {
        group_size[group[o]] += 1;
    }
    group_variance(row, mat, group, num_groups, group_size.data(), output, opt);
}

/**
 * @brief Results of `group_variance()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_>
struct GroupVarianceResult {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample mean of each row/column.
     */
    std::vector<std::vector<Output_> > mean;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample variance of each row/column.
     */
    std::vector<std::vector<Output_> > variance;
};

/**
 * Compute per-group variances for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 *
 * @param row Whether to compute variances for the rows.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * @param opt Further options.
 *
 * @return Variance and mean of each group for each row/column.
 */
template<typename Output_ = double, typename Value_, typename Index_, typename Group_> 
GroupVarianceResult<Output_> group_variance(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    const GroupVarianceOptions<Output_>& opt
) {
    GroupVarianceResult<Output_> output;
    sanisizer::resize(output.mean, num_groups);
    sanisizer::resize(output.variance, num_groups);

    GroupVarianceBuffers<Output_> buffers;
    sanisizer::resize(buffers.mean, num_groups);
    sanisizer::resize(buffers.variance, num_groups);
    const auto dim = (row ? mat.nrow() : mat.ncol());

    for (std::size_t g = 0; g < num_groups; ++g) {
        tatami::resize_container_to_Index_size(output.mean[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        buffers.mean[g] = output.mean[g].data();
        tatami::resize_container_to_Index_size(output.variance[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        buffers.variance[g] = output.variance[g].data();
    }

    group_variance(row, mat, group, num_groups, buffers, opt);
    return output;
}

}

#endif
