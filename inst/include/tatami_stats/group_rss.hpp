#ifndef TATAMI_STATS_GROUP_RSS_HPP
#define TATAMI_STATS_GROUP_RSS_HPP

#include "utils.hpp"

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

/**
 * @file group_rss.hpp
 *
 * @brief Compute group-wise residual sum of squares from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `group_rss()`.
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct GroupRssOptions {
    /**
     * Number of threads to use when computing RSS values across a `tatami::Matrix`.
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
 * @brief Result buffers for `group_rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_>
struct GroupRssBuffers {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `group_rss()`, this is filled with the sample mean of each row/column for the corresponding group.
     */
    std::vector<Output_*> mean;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `group_rss()`, this is filled with the residual sum of squares of each row/column for the corresponding group.
     */
    std::vector<Output_*> rss;
};

/**
 * @cond
 */
template<typename Output_, typename Count_, typename Index_>
void group_rss_finish_means(
    const std::size_t num_groups,
    const Count_* const group_size,
    std::vector<Output_>& means,
    const Index_ i,
    std::vector<Output_*>& output_means,
    const Output_ placeholder
) {
    for (std::size_t b = 0; b < num_groups; ++b) {
        if (group_size[b]) {
            means[b] /= group_size[b];
        } else {
            means[b] = placeholder;
        }
        output_means[b][i] = means[b];
    }
}

template<typename Value_, typename Index_, typename Group_, typename Count_, typename Output_>
void group_rss_direct(
    const bool row,
    const tatami::Matrix<Value_, Index_>& mat, 
    const Group_* const group, 
    const std::size_t num_groups, 
    const Count_* const group_size,
    GroupRssBuffers<Output_>& output,
    const GroupRssOptions<Output_>& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.sparse()) {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);
            auto cur_means = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_rss = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_non_zeros = sanisizer::create<std::vector<Index_> >(num_groups);

            for (Index_ x = 0; x < l; ++x) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());

                // Computing the mean first.
                for (Index_ i = 0; i < range.number; ++i) {
                    const auto g = group[range.index[i]];
                    cur_means[g] += range.value[i];
                    ++cur_non_zeros[g];
                }
                group_rss_finish_means(num_groups, group_size, cur_means, static_cast<Index_>(s + x), output.mean, opt.mean_placeholder);

                // Now computing the RSS.
                for (Index_ i = 0; i < range.number; ++i) {
                    const auto g = group[range.index[i]];
                    const auto delta = range.value[i] - cur_means[g];
                    cur_rss[g] += delta * delta;
                }
                for (std::size_t g = 0; g < num_groups; ++g) {
                    if (group_size[g] > 0) { // preserve RSS = 0 if the group is empty, otherwise the placeholder mean might be a NaN that causes problems.
                        const Output_ my_rss = cur_rss[g] + cur_means[g] * cur_means[g] * (group_size[g] - cur_non_zeros[g]);
                        output.rss[g][s + x] = my_rss;
                    } else {
                        output.rss[g][s + x] = 0;
                    }
                }

                std::fill(cur_means.begin(), cur_means.end(), 0);
                std::fill(cur_rss.begin(), cur_rss.end(), 0);
                std::fill(cur_non_zeros.begin(), cur_non_zeros.end(), 0);
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto cur_means = sanisizer::create<std::vector<Output_> >(num_groups);
            auto cur_rss = sanisizer::create<std::vector<Output_> >(num_groups);

            for (Index_ x = 0; x < l; ++x) {
                auto ptr = ext->fetch(buffer.data());

                // Computing the mean first.
                for (Index_ j = 0; j < otherdim; ++j) {
                    cur_means[group[j]] += ptr[j];
                }
                group_rss_finish_means(num_groups, group_size, cur_means, static_cast<Index_>(s + x), output.mean, opt.mean_placeholder);

                // Now computing the RSS.
                for (Index_ j = 0; j < otherdim; ++j) {
                    const auto g = group[j];
                    const auto delta = ptr[j] - cur_means[g];
                    cur_rss[g] += delta * delta;
                }
                for (std::size_t g = 0; g < num_groups; ++g) {
                    output.rss[g][s + x] = cur_rss[g];
                }

                std::fill(cur_means.begin(), cur_means.end(), 0);
                std::fill(cur_rss.begin(), cur_rss.end(), 0);
            }
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Group_, typename Count_, typename Output_>
void group_rss_running_nonempty(
    const bool row,
    const Index_ dim,
    const Index_ otherdim,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group, 
    const std::size_t num_groups, 
    const Count_* const group_size,
    GroupRssBuffers<Output_>& output,
    const GroupRssOptions<Output_>& opt
) {
    const bool do_parallel = opt.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Output_> > > > > all_partial_mean, all_partial_rss;
    std::optional<std::vector<std::optional<std::vector<Count_> > > > all_partial_count;
    if (do_parallel) {
        // -1, as we'll repurpose the RSS output buffer to store the partial RSS of the first thread.
        all_partial_rss.emplace(sanisizer::cast<I<decltype(all_partial_rss->size())> >(opt.num_threads - 1));
        all_partial_mean.emplace(sanisizer::cast<I<decltype(all_partial_mean->size())> >(opt.num_threads));
        all_partial_count.emplace(sanisizer::cast<I<decltype(all_partial_count->size())> >(opt.num_threads));
    }

    // All groups are assumed to be non-empty at this point,
    // which allows us to skip some allocations.
    for (std::size_t g = 0; g < num_groups; ++g) {
        assert(group_size[g] > 0);
    }

    // We overwrite any existing value in the array in the do_parallel=true situation.
    // So, the initial value doesn't need to be zero.
    if (!do_parallel) {
        for (std::size_t g = 0; g < num_groups; ++g) {
            std::fill_n(output.mean[g], dim, 0);
        }
    }
    for (std::size_t g = 0; g < num_groups; ++g) {
        std::fill_n(output.rss[g], dim, 0);
    }

    const bool is_sparse = mat.is_sparse();
    const int nused = tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
        Output_** mean_ptrs;
        Output_** rss_ptrs;
        std::optional<std::vector<Output_*> > tmp_mean_ptrs, tmp_rss_ptrs;
        std::optional<std::vector<std::vector<Output_> > > cur_mean, cur_rss;

        if (!do_parallel) {
            // Storing mean and RSS directly in the output vector to cut down two allocations if we're not working in parallel.
            mean_ptrs = output.mean.data();
            rss_ptrs = output.rss.data();

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

        auto cur_count = sanisizer::create<std::vector<Count_> >(num_groups); 

        if (is_sparse) {
            auto ext = tatami::consecutive_extractor<true>(mat, !row, s, l);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);
            auto nonzeros = sanisizer::create<std::vector<std::vector<Index_> > >(num_groups);
            for (std::size_t g = 0; g < num_groups; ++g) {
                tatami::resize_container_to_Index_size(nonzeros[g], dim);
            }

            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                const auto grp = group[s + x];
                ++cur_count[grp]; // increment is safe as 'cur_count[grp] + 1 <= l' fits in an Index_.
                const auto mptr = mean_ptrs[grp];
                const auto rptr = rss_ptrs[grp];
                auto& nnz = nonzeros[grp];
                for (Index_ i = 0; i < out.number; ++i) {
                    const auto d = out.index[i];
                    quickstats::update_rss(mptr[d], rptr[d], out.value[i], ++nnz[d]); // increment is safe as 'nnz + 1 <= l' fits in an Index_.
                }
            }

            for (std::size_t g = 0; g < num_groups; ++g) {
                const auto curtotal = cur_count[g];
                if (curtotal) {
                    const auto mptr = mean_ptrs[g];
                    const auto rptr = rss_ptrs[g];
                    const auto& nnz = nonzeros[g];
                    for (Index_ d = 0; d < dim; ++d) {
                        // unsafe call is possible as we check for curtotal > 0.
                        quickstats::update_rss_with_zeros_unsafe(mptr[d], rptr[d], static_cast<Count_>(curtotal - nnz[d]), curtotal);
                    }
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, !row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(buffer.data());
                const auto grp = group[s + x];
                ++cur_count[grp]; // increment is safe as 'cur_count[grp] + 1 <= l' fits in an Index_.
                const auto mptr = mean_ptrs[grp];
                const auto rptr = rss_ptrs[grp];
                for (Index_ d = 0; d < dim; ++d) {
                    quickstats::update_rss(mptr[d], rptr[d], out[d], cur_count[grp]);
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

        for (std::size_t g = 0; g < num_groups; ++g) {
            const auto cur_output = output.mean[g];
            const auto cur_global_count = group_size[g];
            assert(cur_global_count > 0);
            bool initialized = false;

            for (int u = 0; u < nused; ++u) {
                const auto cur_count = (*((*all_partial_count)[u]))[g];
                if (cur_count == 0) {
                    continue;
                }

                const auto& cur_mean = (*(ap_mean[u]))[g];
                const Output_ mult = static_cast<Output_>(cur_count) / static_cast<Output_>(cur_global_count);
                if (!initialized) { // Don't use u == 0, as the first non-empty 'g' might not occur in the first thread.
                    for (Index_ d = 0; d < dim; ++d) {
                        cur_output[d] = cur_mean[d] * mult;
                    }
                    initialized = true;
                } else {
                    for (Index_ d = 0; d < dim; ++d) {
                        cur_output[d] += cur_mean[d] * mult;
                    }
                }
            }

            assert(initialized);
        }

        // Combining the RSS. 
        for (std::size_t g = 0; g < num_groups; ++g) {
            const auto& cur_global = output.mean[g];
            const auto cur_output = output.rss[g];
            bool initialized = false;

            for (int u = 0; u < nused; ++u) {
                const auto cur_count = (*((*all_partial_count)[u]))[g];
                if (cur_count == 0) { // This check allows us to use the unsafe RSS centering below.
                    continue;
                }

                const auto& cur_mean = (*(ap_mean[u]))[g];
                if (u == 0) { // Special case to avoid trying to access u - 1.
                    for (Index_ d = 0; d < dim; ++d) {
                        cur_output[d] = quickstats::recenter_rss_unsafe(cur_count, cur_output[d], cur_mean[d], cur_global[d]); 
                    }
                    initialized = true;
                } else {
                    const auto& cur_rss = (*(ap_rss[u - 1]))[g];
                    if (!initialized) { // Don't use u == 0, as the first non-empty 'g' might not occur in the first thread.
                        for (Index_ d = 0; d < dim; ++d) {
                            cur_output[d] = quickstats::recenter_rss_unsafe(cur_count, cur_rss[d], cur_mean[d], cur_global[d]); 
                        }
                        initialized = true;
                    } else {
                        for (Index_ d = 0; d < dim; ++d) {
                            cur_output[d] += quickstats::recenter_rss_unsafe(cur_count, cur_rss[d], cur_mean[d], cur_global[d]); 
                        }
                    }
                }
            }

            assert(initialized);
        }
    }
}

template<typename Value_, typename Index_, typename Group_, typename Count_, typename Output_>
void group_rss_running(
    const bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group, 
    const std::size_t num_groups, 
    const Count_* const group_size,
    GroupRssBuffers<Output_>& output,
    const GroupRssOptions<Output_>& opt
) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    if (otherdim == 0) {
        for (std::size_t g = 0; g < num_groups; ++g) {
            std::fill_n(output.mean[g], dim, opt.mean_placeholder);
            std::fill_n(output.rss[g], dim, 0);
        }
        return; 
    }

    std::size_t num_empty = 0;
    for (std::size_t g = 0; g < num_groups; ++g) {
        num_empty += (group_size[g] == 0);
    }

    // We strip out all empty groups so that we can skip some allocations in each thread. 
    std::optional<std::vector<Count_> > new_group_size_store;
    std::optional<GroupRssBuffers<Output_> > new_output_store;
    std::optional<std::vector<Group_> > new_group_store;
    std::size_t num_non_empty;
    const Count_* new_group_size;
    const Group_* new_group;
    GroupRssBuffers<Output_>* new_output;

    if (num_empty > 0) {
        num_non_empty = num_groups - num_empty; // must be positive, otherwise otherdim == 0 if all groups are empty.
        new_group_size_store.emplace();
        new_group_size_store->reserve(num_non_empty);
        new_output_store.emplace();
        new_output_store->mean.reserve(num_non_empty);
        new_output_store->rss.reserve(num_non_empty);

        auto mapping = sanisizer::create<std::vector<std::size_t> >(num_groups);
        for (std::size_t g = 0; g < num_groups; ++g) {
            if (group_size[g]) {
                mapping[g] = new_group_size_store->size();
                new_group_size_store->push_back(group_size[g]);
                new_output_store->mean.push_back(output.mean[g]);
                new_output_store->rss.push_back(output.rss[g]);
            } else {
                std::fill_n(output.mean[g], dim, opt.mean_placeholder);
                std::fill_n(output.rss[g], dim, 0);
            }
        }

        new_group_store.emplace(tatami::cast_Index_to_container_size<std::vector<Group_> >(otherdim));
        for (Index_ i = 0; i < otherdim; ++i) {
            (*new_group_store)[i] = mapping[group[i]];
        }

        new_group_size = new_group_size_store->data();
        new_output = &(*new_output_store);
        new_group = new_group_store->data();
    } else {
        num_non_empty = num_groups;
        new_group_size = group_size;
        new_output = &output;
        new_group = group;
    }

    group_rss_running_nonempty(
        row,
        dim,
        otherdim,
        mat,
        new_group,
        num_non_empty,
        new_group_size,
        *new_output,
        opt
    );
}
/**
 * @endcond
 */

/**
 * Compute per-group residual sums of squares (RSS) for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 * @tparam Count_ Numeric type of the group sizes, typically integer.
 * @tparam Output_ Floating-point type of the output value.
 *
 * @param row Whether to compute RSS values for the rows.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should be non-negative and less than `num_groups`.
 * @param num_groups Number of groups in `group`.
 * @param[in] group_size Pointer to an array of length equal to `num_groups`, containing the size of each group.
 * @param[out] output Buffers in which to store the results.
 * On output, each array stores the means and RSS values of the corresponding group.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Group_, typename Count_, typename Output_>
void group_rss(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    const Count_* const group_size,
    GroupRssBuffers<Output_>& output,
    const GroupRssOptions<Output_>& opt
) {
    assert(sanisizer::is_equal(num_groups, output.mean.size()));
    assert(sanisizer::is_equal(num_groups, output.rss.size()));
    if (mat.prefer_rows() == row) {
        group_rss_direct(row, mat, group, num_groups, group_size, output, opt);
    } else {
        group_rss_running(row, mat, group, num_groups, group_size, output, opt);
    }
}

/**
 * Overload that computes the group sizes before calling `group_rss()`.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Count_ Numeric type of the group sizes, typically integer.
 *
 * @param row Whether to compute RSS values for the rows.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should be non-negative and less than `num_groups`.
 * @param num_groups Number of groups in `group`.
 * @param[out] output Buffers in which to store the results.
 * On output, each array stores the means and RSS values of the corresponding group.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Group_, typename Output_>
void group_rss(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    GroupRssBuffers<Output_>& output,
    const GroupRssOptions<Output_>& opt
) {
    auto group_size = sanisizer::create<std::vector<Index_> >(num_groups);
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    for (Index_ o = 0; o < otherdim; ++o) {
        group_size[group[o]] += 1;
    }
    group_rss(row, mat, group, num_groups, group_size.data(), output, opt);
}

/**
 * @brief Results of `group_rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_>
struct GroupRssResult {
    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample mean of each row/column.
     */
    std::vector<std::vector<Output_> > mean;

    /**
     * Vector of length equal to the number of groups.
     * Each element is a vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the residual sum of squares of each row/column.
     */
    std::vector<std::vector<Output_> > rss;
};

/**
 * Overload of `group_rss()` that allocates memory for the results.
 *
 * @tparam Output_ Floating-point type of the output value.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Group_ Integer type of the group assignments for each row/column.
 *
 * @param row Whether to compute RSS values for the rows.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[in] group Pointer to an array of length equal to the number of columns (if `row = true`) or rows (otherwise).
 * Each value should be an integer that specifies the group assignment.
 * Values should lie in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param num_groups Number of groups, i.e., \f$N\f$.
 * @param opt Further options.
 *
 * @return RSS and mean of each group for each row/column.
 */
template<typename Output_, typename Value_, typename Index_, typename Group_> 
GroupRssResult<Output_> group_rss(
    bool row,
    const tatami::Matrix<Value_, Index_>& mat,
    const Group_* const group,
    const std::size_t num_groups,
    const GroupRssOptions<Output_>& opt
) {
    GroupRssResult<Output_> output;
    sanisizer::resize(output.mean, num_groups);
    sanisizer::resize(output.rss, num_groups);

    GroupRssBuffers<Output_> buffers;
    sanisizer::resize(buffers.mean, num_groups);
    sanisizer::resize(buffers.rss, num_groups);

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
    }

    group_rss(row, mat, group, num_groups, buffers, opt);
    return output;
}

}

#endif
