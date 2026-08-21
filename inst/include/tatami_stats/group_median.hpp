#ifndef TATAMI_STATS_GROUP_MEDIAN_HPP
#define TATAMI_STATS_GROUP_MEDIAN_HPP

#include "utils.hpp"
#include "median.hpp"

#include <vector>
#include <algorithm>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file group_median.hpp
 *
 * @brief Compute group-wise medians from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `group_median()`.
 */
struct GroupMedianOptions {
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
 * Compute per-group medians for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each column.
 * @tparam GroupSizes_ Vector-like class that has `size()` and `[` methods and contains integers.
 * @tparam Output_ Floating-point type of the output value, capable of storing averages or NaNs.
 *
 * @param row Whether to compute group-wise medians within each row.
 * If false, medians are computed in each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * @param[out] output Pointer to an array of pointers of length equal to the number of groups.
 * Each inner pointer should reference an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column medians for each group (indexed according to the assignment in `group`).
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Group_, typename Output_>
void group_median(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    std::vector<Output_*>& output,
    const GroupMedianOptions& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    auto group_sizes = sanisizer::create<std::vector<Index_> >(num_groups);
    for (Index_ i = 0; i < otherdim; ++i) {
        group_sizes[group[i]] += 1;
    }

    tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
        auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
        auto workspace = sanisizer::create<std::vector<std::vector<Value_> > >(num_groups);
        for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
            sanisizer::reserve(workspace[g], group_sizes[g]);
        }

        if (mat.sparse()) {
            tatami::Options topt;
            topt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, row, start, len, topt);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);

            for (Index_ i = 0; i < len; ++i) {
                auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                for (Index_ j = 0; j < range.number; ++j) {
                    workspace[group[range.index[j]]].push_back(range.value[j]);
                }

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    auto& w = workspace[g];
                    output[g][i + start] = median_direct<Output_, Value_, Index_>(w.data(), w.size(), group_sizes[g], opt.skip_nan);
                    w.clear();
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, row, start, len);
            for (Index_ i = 0; i < len; ++i) {
                auto ptr = ext->fetch(xbuffer.data());
                for (Index_ j = 0; j < otherdim; ++j) {
                    workspace[group[j]].push_back(ptr[j]);
                }

                for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
                    auto& w = workspace[g];
                    output[g][i + start] = median_direct<Output_, Value_, Index_>(w.data(), w.size(), opt.skip_nan);
                    w.clear();
                }
            }
        }
    }, dim, opt.num_threads);
}

/**
 * Overload of `group_median()` that allocates memory for the output medians.
 *
 * @tparam Output_ Floating-point type of the output value, capable of storing averages or NaNs.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each column.
 *
 * @param row Whether to compute group-wise medians within each row.
 * If false, medians are computed in each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * @param opt Further options.
 *
 * @return Vector of length equal to the number of groups.
 * Each element is a vector of length equal to the number of rows (if `row = true`) or columns (otherwise),
 * containing the row/column medians for the corresponding group.
 */
template<typename Output_ = double, typename Value_, typename Index_, typename Group_>
std::vector<std::vector<Output_> > group_median(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    const GroupMedianOptions& opt
) {
    auto output = sanisizer::create<std::vector<std::vector<Output_> > >(num_groups);
    auto outptrs = sanisizer::create<std::vector<Output_*> >(num_groups);
    const auto dim = (row ? mat.nrow() : mat.ncol());
    for (std::size_t g = 0; g < num_groups; ++g) {
        tatami::resize_container_to_Index_size(output[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        outptrs[g] = output[g].data();
    }
    group_median(row, mat, group, num_groups, outptrs, opt);
    return output;
}

}

#endif
