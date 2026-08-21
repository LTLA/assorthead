#ifndef TATAMI_STATS_SKIP_NAN_GROUP_RSS_HPP
#define TATAMI_STATS_SKIP_NAN_GROUP_RSS_HPP

#include <vector>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <cassert>
#include <limits>
#include <cmath>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "quickstats/quickstats.hpp"

#include "../utils.hpp"
#include "../group_rss.hpp"

/**
 * @file group_rss.hpp
 *
 * @brief Compute group-wise residual sum of squares while skipping NaNs.
 */

namespace tatami_stats {

namespace skip_nan {

/**
 * @brief Options for `skip_nan::group_rss()`.
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct GroupRssOptions {
    /**
     * Number of threads to use for iterating across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;

    /**
     * Placeholder value to use for the mean of empty groups.
     * This is NaN if supported by `Output_`, otherwise it is zero.
     */
    Output_ mean_placeholder = quickstats::nan_if_available_else_zero<Output_>();
};

/**
 * @brief Result buffers for `skip_nan::group_rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
 */
template<typename Output_, typename Count_>
struct GroupRssBuffers {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `skip_nan::group_rss()`, this is filled with the sample mean of each row/column for the corresponding group.
     */
    std::vector<Output_*> mean;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `skip_nan::group_rss()`, this is filled with the residual sum of squares of each row/column for the corresponding group.
     */
    std::vector<Output_*> rss;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `skip_nan::group_rss()`, this is filled with the number of unskipped observations in each row/column for the corresponding group.
     */
    std::vector<Count_*> count;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Group_, typename Output_, typename Count_>
void group_rss_direct(
    const bool row,
    const tatami::Matrix<Value_, Index_>& mat, 
    const Group_* const group, 
    const std::size_t num_groups, 
    GroupRssBuffers<Output_, Count_>& output,
    const GroupRssOptions<Output_>& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.sparse()) {
        auto full_group_sizes = sanisizer::create<std::vector<Index_> >(num_groups);
        for (Index_ i = 0; i < otherdim; ++i) {
            full_group_sizes[group[i]] += 1;
        }

        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);
            auto cur_means = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_rss = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_non_zeros = sanisizer::create<std::vector<Index_> >(num_groups);
            auto cur_sizes = sanisizer::create<std::vector<Index_> >(num_groups);

            for (Index_ x = 0; x < l; ++x) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());

                // Computing the mean first.
                for (Index_ i = 0; i < range.number; ++i) {
                    const auto val = range.value[i];
                    const auto b = group[range.index[i]];
                    if (!std::isnan(val)) {
                        ++cur_non_zeros[b];
                        cur_means[b] += val;
                    } else {
                        ++cur_sizes[b];
                    }
                }
                for (std::size_t g = 0; g < num_groups; ++g) {
                    const auto actual_size = full_group_sizes[g] - cur_sizes[g];
                    cur_sizes[g] = actual_size;
                    output.count[g][s + x] = actual_size;
                }
                group_rss_finish_means(num_groups, cur_sizes.data(), cur_means, static_cast<Index_>(s + x), output.mean, opt.mean_placeholder);

                // Now computing the RSS.
                for (Index_ i = 0; i < range.number; ++i) {
                    const auto val = range.value[i];
                    if (!std::isnan(val)) {
                        const auto g = group[range.index[i]];
                        const auto delta = val - cur_means[g];
                        cur_rss[g] += delta * delta;
                    }
                }
                for (std::size_t g = 0; g < num_groups; ++g) {
                    if (cur_sizes[g] > 0) { // preserve RSS = 0 for empty groups, otherwise placeholder mean might cause problems.
                        const Output_ my_rss = cur_rss[g] + cur_means[g] * cur_means[g] * (cur_sizes[g] - cur_non_zeros[g]);
                        output.rss[g][s + x] = my_rss;
                    } else {
                        output.rss[g][s + x] = 0;
                    }
                }

                std::fill(cur_means.begin(), cur_means.end(), 0);
                std::fill(cur_rss.begin(), cur_rss.end(), 0);
                std::fill(cur_non_zeros.begin(), cur_non_zeros.end(), 0);
                std::fill(cur_sizes.begin(), cur_sizes.end(), 0);
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto cur_means = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_rss = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_sizes = sanisizer::create<std::vector<Index_> >(num_groups);

            for (Index_ x = 0; x < l; ++x) {
                auto ptr = ext->fetch(buffer.data());

                // Computing the mean first.
                for (Index_ j = 0; j < otherdim; ++j) {
                    const auto val = ptr[j];
                    if (!std::isnan(val)) {
                        const auto g = group[j];
                        cur_means[g] += val;
                        ++cur_sizes[g];
                    }
                }
                for (std::size_t g = 0; g < num_groups; ++g) {
                    output.count[g][s + x] = cur_sizes[g];
                }
                group_rss_finish_means(num_groups, cur_sizes.data(), cur_means, static_cast<Index_>(s + x), output.mean, opt.mean_placeholder);

                // Now computing the RSS.
                for (Index_ j = 0; j < otherdim; ++j) {
                    const auto val = ptr[j];
                    if (!std::isnan(val)) {
                        const auto g = group[j];
                        const auto delta = val - cur_means[g];
                        cur_rss[g] += delta * delta;
                    }
                }
                for (std::size_t g = 0; g < num_groups; ++g) {
                    output.rss[g][s + x] = cur_rss[g];
                }

                std::fill(cur_means.begin(), cur_means.end(), 0);
                std::fill(cur_rss.begin(), cur_rss.end(), 0);
                std::fill(cur_sizes.begin(), cur_sizes.end(), 0);
            }
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Group_, typename Output_, typename Count_>
void group_rss_running(
    const bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group, 
    const std::size_t num_groups, 
    GroupRssBuffers<Output_, Count_>& output,
    const GroupRssOptions<Output_>& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    for (std::size_t g = 0; g < num_groups; ++g) {
        std::fill_n(output.rss[g], dim, 0);
        std::fill_n(output.count[g], dim, 0);
    }

    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    if (otherdim == 0) {
        for (std::size_t g = 0; g < num_groups; ++g) {
            std::fill_n(output.mean[g], dim, opt.mean_placeholder);
        }
        return; 
    } else {
        for (std::size_t g = 0; g < num_groups; ++g) {
            std::fill_n(output.mean[g], dim, 0);
        }
    }

    const bool do_parallel = opt.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > all_partial_mean, all_partial_rss;
    std::optional<std::vector<std::optional<std::vector<std::vector<Count_> > > > > all_partial_count;
    if (do_parallel) {
        // -1, as we'll repurpose the RSS output buffer to store the partial RSS of the first thread.
        all_partial_rss.emplace(sanisizer::cast<I<decltype(all_partial_rss->size())> >(opt.num_threads - 1));
        all_partial_mean.emplace(sanisizer::cast<I<decltype(all_partial_mean->size())> >(opt.num_threads));
        all_partial_count.emplace(sanisizer::cast<I<decltype(all_partial_count->size())> >(opt.num_threads));
    }

    const bool is_sparse = mat.is_sparse();
    const int nused = tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
        Output_** mean_ptrs;
        Output_** rss_ptrs;
        std::optional<std::vector<Output_*> > tmp_mean_ptrs, tmp_rss_ptrs;
        std::optional<std::vector<std::vector<Output_> > > cur_mean, cur_rss;
        Count_** count_ptrs;
        std::optional<std::vector<Count_*> > tmp_count_ptrs;
        std::optional<std::vector<std::vector<Count_> > > cur_count;

        if (!do_parallel) {
            // Storing mean and RSS directly in the output vector to cut down two allocations if we're not working in parallel.
            mean_ptrs = output.mean.data();
            rss_ptrs = output.rss.data();
            count_ptrs = output.count.data();

        } else {
            // Storing the partial RSS directly in the output vectors to save ourselves an allocation if we're in the first thread.
            // We can't do the same for the mean, though, as we need to keep the partial mean and the global mean separate for the reduction.
            cur_mean.emplace(sanisizer::cast<I<decltype(cur_mean->size())> >(num_groups));
            tmp_mean_ptrs.emplace(sanisizer::cast<I<decltype(tmp_mean_ptrs->size())> >(num_groups));
            for (std::size_t g = 0; g < num_groups; ++g) {
                tatami::resize_container_to_Index_size((*cur_mean)[g], dim);
                (*tmp_mean_ptrs)[g] = (*cur_mean)[g].data();
            }
            mean_ptrs = tmp_mean_ptrs->data();

            cur_count.emplace(sanisizer::cast<I<decltype(cur_count->size())> >(num_groups));
            tmp_count_ptrs.emplace(sanisizer::cast<I<decltype(tmp_count_ptrs->size())> >(num_groups));
            for (std::size_t g = 0; g < num_groups; ++g) {
                tatami::resize_container_to_Index_size((*cur_count)[g], dim);
                (*tmp_count_ptrs)[g] = (*cur_count)[g].data();
            }
            count_ptrs = tmp_count_ptrs->data();

            if (thread == 0) {
                rss_ptrs = output.rss.data();
            } else {
                cur_rss.emplace(sanisizer::cast<I<decltype(cur_rss->size())> >(num_groups));
                tmp_rss_ptrs.emplace(sanisizer::cast<I<decltype(tmp_rss_ptrs->size())> >(num_groups));
                for (std::size_t g = 0; g < num_groups; ++g) {
                    tatami::resize_container_to_Index_size((*cur_rss)[g], dim);
                    (*tmp_rss_ptrs)[g] = (*cur_rss)[g].data();
                }
                rss_ptrs = tmp_rss_ptrs->data();
            }
        }

        if (is_sparse) {
            auto ext = tatami::consecutive_extractor<true>(mat, !row, s, l);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);
            auto nonzeros = sanisizer::create<std::vector<std::vector<Count_> > >(num_groups);
            for (std::size_t g = 0; g < num_groups; ++g) {
                tatami::resize_container_to_Index_size(nonzeros[g], dim);
            }

            auto cur_group_size = sanisizer::create<std::vector<Count_> >(num_groups);
            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                const auto grp = group[s + x];
                const auto mptr = mean_ptrs[grp];
                const auto rptr = rss_ptrs[grp];
                const auto cptr = count_ptrs[grp];
                ++cur_group_size[grp];
                auto& nnz = nonzeros[grp];
                for (Index_ i = 0; i < out.number; ++i) {
                    const auto d = out.index[i];
                    const auto val = out.value[i];
                    if (!std::isnan(val)) {
                        quickstats::update_rss(mptr[d], rptr[d], val, ++nnz[d]); // increment is safe as 'nnz + 1 <= l' fits in an Index_.
                    } else {
                        ++cptr[d];
                    }
                }
            }

            for (std::size_t g = 0; g < num_groups; ++g) {
                const auto mptr = mean_ptrs[g];
                const auto rptr = rss_ptrs[g];
                const auto cptr = count_ptrs[g];
                const auto& nnz = nonzeros[g];
                const auto curtotal = cur_group_size[g];
                for (Index_ d = 0; d < dim; ++d) {
                    auto& unskipped_total = cptr[d];
                    unskipped_total = curtotal - unskipped_total; // could be zero, so the update with zeros needs to be safe.
                    quickstats::update_rss_with_zeros(mptr[d], rptr[d], static_cast<Count_>(unskipped_total - nnz[d]), unskipped_total);
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, !row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(buffer.data());
                const auto grp = group[s + x];
                const auto mptr = mean_ptrs[grp];
                const auto rptr = rss_ptrs[grp];
                const auto cptr = count_ptrs[grp];
                for (Index_ d = 0; d < dim; ++d) {
                    const auto val = out[d];
                    if (!std::isnan(val)) {
                        quickstats::update_rss(mptr[d], rptr[d], val, ++cptr[d]); // increment is safe as 'cptr[d] + 1 <= l' fits in an Index_.
                    }
                }
            }
        }

        if (do_parallel) {
            (*all_partial_count)[thread] = std::move(cur_count);
            (*all_partial_mean)[thread] = std::move(cur_mean);
            if (thread > 0) {
                (*all_partial_rss)[thread - 1] = std::move(cur_rss);
            }
        }
    }, otherdim, opt.num_threads);
    assert(nused > 0);

    if (do_parallel) {
        const auto& ap_mean = *all_partial_mean;
        const auto& ap_rss = *all_partial_rss;
        const auto& ap_count = *all_partial_count;

        for (std::size_t g = 0; g < num_groups; ++g) {
            const auto cur_global_count = output.count[g];
            for (int u = 0; u < nused; ++u) {
                const auto& cur_count = (*(ap_count[u]))[g];
                for (Index_ d = 0; d < dim; ++d) {
                    cur_global_count[d] += cur_count[d];
                }
            }
        }

        // Computing the global mean.
        for (std::size_t g = 0; g < num_groups; ++g) {
            const auto cur_global_count = output.count[g];
            const auto cur_global_mean = output.mean[g];

            for (int u = 0; u < nused; ++u) {
                const auto& cur_mean = (*(ap_mean[u]))[g];
                const auto& cur_count = (*(ap_count[u]))[g];
                for (Index_ d = 0; d < dim; ++d) {
                    if (cur_count[d] > 0) {
                        const auto mult = static_cast<Output_>(cur_count[d]) / static_cast<Output_>(cur_global_count[d]);
                        cur_global_mean[d] += cur_mean[d] * mult;
                    }
                }
            }
        }

        // Combining the RSS. We need to use the safe variant of recenter_rss(), just to protect against the
        // case where a group has no observations within a particular thread. 
        for (std::size_t g = 0; g < num_groups; ++g) {
            const auto cur_global_mean = output.mean[g];
            const auto cur_output = output.rss[g];
            for (int u = 0; u < nused; ++u) {
                const auto& cur_mean = (*(ap_mean[u]))[g];
                const auto& cur_count = (*(ap_count[u]))[g];
                if (u == 0) {
                    for (Index_ d = 0; d < dim; ++d) {
                        cur_output[d] = quickstats::recenter_rss(cur_count[d], cur_output[d], cur_mean[d], cur_global_mean[d]); 
                    }
                } else {
                    const auto& cur_rss = (*(ap_rss[u - 1]))[g];
                    for (Index_ d = 0; d < dim; ++d) {
                        cur_output[d] += quickstats::recenter_rss(cur_count[d], cur_rss[d], cur_mean[d], cur_global_mean[d]); 
                    }
                }
            }
        }
    }

    for (std::size_t g = 0; g < num_groups; ++g) {
        const auto mptr = output.mean[g];
        const auto cptr = output.count[g];
        for (Index_ d = 0; d < dim; ++d) {
            if (cptr[d] == 0) {
                mptr[d] = opt.mean_placeholder;
            }
        }
    }
}
/**
 * @endcond
 */

/**
 * Compute per-group variances for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
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
template<typename Value_, typename Index_, typename Group_, typename Output_, typename Count_>
void group_rss(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    GroupRssBuffers<Output_, Count_>& output,
    const GroupRssOptions<Output_>& opt
) {
    assert(sanisizer::is_equal(num_groups, output.mean.size()));
    assert(sanisizer::is_equal(num_groups, output.rss.size()));
    assert(sanisizer::is_equal(num_groups, output.count.size()));
    if (mat.prefer_rows() == row) {
        group_rss_direct(row, mat, group, num_groups, output, opt);
    } else {
        group_rss_running(row, mat, group, num_groups, output, opt);
    }
}

/**
 * @brief Results of `skip_nan::group_rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
 */
template<typename Output_, typename Count_>
struct GroupRssResult {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample mean of each row/column for the corresponding group.
     */
    std::vector<std::vector<Output_> > mean;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the residual sum of squares of each row/column for the corresponding group.
     */
    std::vector<std::vector<Output_> > rss;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the number of unskipped observations in each row/column for the corresponding group.
     */
    std::vector<std::vector<Count_> > count;
};

/**
 * Compute per-group variances for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
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
 * @return RSS and mean of each group for each row/column.
 */
template<typename Output_, typename Count_, typename Value_, typename Index_, typename Group_> 
GroupRssResult<Output_, Count_> group_rss(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    const GroupRssOptions<Output_>& opt
) {
    GroupRssResult<Output_, Count_> output;
    sanisizer::resize(output.mean, num_groups);
    sanisizer::resize(output.rss, num_groups);
    sanisizer::resize(output.count, num_groups);

    GroupRssBuffers<Output_, Count_> buffers;
    sanisizer::resize(buffers.mean, num_groups);
    sanisizer::resize(buffers.rss, num_groups);
    sanisizer::resize(buffers.count, num_groups);

    const auto dim = (row ? mat.nrow() : mat.ncol());
    for (std::size_t g = 0; g < num_groups; ++g) {
        tatami::resize_container_to_Index_size(output.mean[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        buffers.mean[g] = output.mean[g].data();

        tatami::resize_container_to_Index_size(output.rss[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        buffers.rss[g] = output.rss[g].data();

        tatami::resize_container_to_Index_size(output.count[g], dim
#ifdef TATAMI_STATS_TEST_DIRTY
            , -1
#endif
        );
        buffers.count[g] = output.count[g].data();
    }

    group_rss(row, mat, group, num_groups, buffers, opt);
    return output;
}

}

}

#endif
