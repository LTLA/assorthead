#ifndef TATAMI_STATS_SKIP_NAN_RSS_HPP
#define TATAMI_STATS_SKIP_NAN_RSS_HPP

#include "../utils.hpp"

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

/**
 * @file rss.hpp
 *
 * @brief Compute row and column residual sum of squares after skipping NaNs.
 */

namespace tatami_stats {

namespace skip_nan {

/**
 * @brief Options for `skip_nan::rss()`.
 * @tparam Output_ Floating-point type of the output data.
 */
template<typename Output_ = double>
struct RssOptions {
    /**
     * Number of threads to use for iterating over a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;

    /**
     * Placeholder value to use for the mean when the extent of the relevant dimension is zero.
     * This is NaN if supported by `Output_`, otherwise it is zero.
     */
    Output_ mean_placeholder = quickstats::nan_if_available_else_zero<Output_>();
};

/**
 * @brief Result buffers for `skip_nan::rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
 */
template<typename Output_, typename Count_>
struct RssBuffers {
    /**
     * Pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `skip_nan::rss()`, this is filled with the sample mean of each row/column.
     */
    Output_* mean;

    /**
     * Pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `skip_nan::rss()`, this is filled with the RSS of each row/column.
     */
    Output_* rss;

    /**
     * Pointer to an array of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise).
     * After `skip_nan::rss()`, this is filled with the number of unskipped observations in each row/column.
     */
    Count_* count;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Output_, typename Count_>
void rss_direct(bool row, const tatami::Matrix<Value_, Index_>& mat, RssBuffers<Output_, Count_>& output, const RssOptions<Output_>& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    quickstats::RssOptions<Output_> ropt;
    ropt.mean_placeholder = opt.mean_placeholder;

    if (mat.sparse()) {
        tatami::Options topt;
        topt.sparse_extract_index = false;

        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, topt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            quickstats::RssWorkspace<Output_> work;
            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(vbuffer.data(), NULL);
                tatami::copy_n(out.value, out.number, vbuffer.data());
                const auto new_number = shift_nans(vbuffer.data(), out.number);
                const Index_ new_total = otherdim - (out.number - new_number);
                const auto res = quickstats::rss(new_total, new_number, vbuffer.data(), work, ropt);
                output.mean[x + s] = res.mean;
                output.rss[x + s] = res.rss;
                output.count[x + s] = new_total;
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            quickstats::RssWorkspace<Output_> work;
            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(buffer.data());
                tatami::copy_n(out, otherdim, buffer.data());
                const auto new_total = shift_nans(buffer.data(), otherdim);
                const auto res = quickstats::rss(new_total, buffer.data(), work, ropt);
                output.mean[x + s] = res.mean;
                output.rss[x + s] = res.rss;
                output.count[x + s] = new_total;
            }
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Output_, typename Count_>
void rss_running(bool row, const tatami::Matrix<Value_, Index_>& mat, RssBuffers<Output_, Count_>& output, const RssOptions<Output_>& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    const bool is_sparse = mat.is_sparse();

    std::fill_n(output.rss, dim, 0);
    std::fill_n(output.count, dim, 0);
    if (otherdim == 0) {
        std::fill_n(output.mean, dim, opt.mean_placeholder);
        return;
    } else {
        std::fill_n(output.mean, dim, 0);
    }

    assert(opt.num_threads > 0);
    const bool do_parallel = opt.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > all_partial_mean, all_partial_rss;
    std::optional<std::vector<std::optional<std::vector<Count_> > > > all_partial_count;
    if (do_parallel) {
        // -1, as we'll repurpose the output buffers to store the output of the first thread.
        all_partial_rss.emplace(sanisizer::cast<I<decltype(all_partial_rss->size())> >(opt.num_threads - 1));
        all_partial_mean.emplace(sanisizer::cast<I<decltype(all_partial_mean->size())> >(opt.num_threads));
        all_partial_count.emplace(sanisizer::cast<I<decltype(all_partial_mean->size())> >(opt.num_threads));
    }

    const int nused = tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
        Output_* rss_ptr;
        Output_* mean_ptr;
        Count_* count_ptr;
        std::optional<std::vector<Output_> > cur_rss, cur_mean;
        std::optional<std::vector<Count_> > cur_count;

        if (!do_parallel) {
            // Storing mean and RSS directly in the output vector to cut down two allocations if we're not working in parallel.
            rss_ptr = output.rss;
            mean_ptr = output.mean;
            count_ptr = output.count;
        } else {
            // Storing the partial RSS directly in the output vector to save ourselves an allocation if we're in the first thread.
            // We can't do the same for the mean, though, as we need to keep the partial mean and the global mean separate for the reduction.
            cur_mean.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(dim));
            mean_ptr = cur_mean->data();
            cur_count.emplace(tatami::cast_Index_to_container_size<std::vector<Count_> >(dim));
            count_ptr = cur_count->data();
            if (thread == 0) {
                rss_ptr = output.rss;
            } else {
                cur_rss.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(dim));
                rss_ptr = cur_rss->data();
            }
        }

        if (is_sparse) {
            tatami::Options topt;
            topt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, !row, s, l, topt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);
            auto nonzeros = tatami::create_container_of_Index_size<std::vector<Count_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                for (Index_ i = 0; i < out.number; ++i) {
                    const auto d = out.index[i];
                    const auto val = out.value[i];
                    if (!std::isnan(val)) {
                        auto& nnz = nonzeros[d];
                        quickstats::update_rss(mean_ptr[d], rss_ptr[d], val, ++nnz); // increment is safe as 'nnz + 1 <= l' fits in an Index_;
                    } else {
                        ++count_ptr[d];
                    }
                }
            }

            for (Index_ d = 0; d < dim; ++d) {
                auto& unskipped_total = count_ptr[d];
                unskipped_total = l - unskipped_total; // could be zero, so the update with zeros needs to be safe.
                quickstats::update_rss_with_zeros(mean_ptr[d], rss_ptr[d], static_cast<Count_>(unskipped_total - nonzeros[d]), unskipped_total);
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, !row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(buffer.data());
                for (Index_ d = 0; d < dim; ++d) {
                    const auto val = out[d];
                    if (!std::isnan(val)) {
                        quickstats::update_rss(mean_ptr[d], rss_ptr[d], val, ++count_ptr[d]); // increment is safe as 'count_ptr[d] + 1 <= l' fits in an index.
                    }
                }
            }
        }

        // Moving results to the main containers.
        if (do_parallel) {
            (*all_partial_mean)[thread] = std::move(cur_mean);
            (*all_partial_count)[thread] = std::move(cur_count);
            if (thread > 0) {
                (*all_partial_rss)[thread - 1] = std::move(cur_rss);
            }
        }
    }, otherdim, opt.num_threads);
    assert(nused > 0);

    // Don't check nused > 1, as it's possible for do_parallel = true with nused = 1 if not all threads are used.
    // This would cause us to leave output.mean and output.rss empty.
    if (do_parallel) {
        const auto& ap_mean = *all_partial_mean;
        const auto& ap_rss = *all_partial_rss;
        const auto& ap_count = *all_partial_count;

        // Computing the global total.
        for (int u = 0; u < nused; ++u) {
            const auto& cur_count = *(ap_count[u]);
            for (Index_ d = 0; d < dim; ++d) {
                output.count[d] += cur_count[d];
            }
        }

        // Computing the global mean from its components.
        for (int u = 0; u < nused; ++u) {
            const auto& cur_count = *(ap_count[u]);
            const auto& cur_mean = *(ap_mean[u]);
            for (Index_ d = 0; d < dim; ++d) {
                if (cur_count[d] > 0) { // protect against NaN means at a count of 0.
                    const auto mult = static_cast<Output_>(cur_count[d]) / static_cast<Output_>(output.count[d]);
                    output.mean[d] += cur_mean[d] * mult;
                }
            }
        }

        // Combining the RSS. This time, we need to use the safe version as we don't know whether all elements were skipped in a thread.
        for (int u = 0; u < nused; ++u) {
            const auto& cur_count = *(ap_count[u]);
            const auto& cur_mean = *(ap_mean[u]);
            if (u == 0) {
                for (Index_ d = 0; d < dim; ++d) {
                    output.rss[d] = quickstats::recenter_rss(cur_count[d], output.rss[d], cur_mean[d], output.mean[d]); 
                }
            } else {
                const auto& cur_rss = *(ap_rss[u - 1]);
                for (Index_ d = 0; d < dim; ++d) {
                    output.rss[d] += quickstats::recenter_rss(cur_count[d], cur_rss[d], cur_mean[d], output.mean[d]); 
                }
            }
        }
    }

    for (Index_ d = 0; d < dim; ++ d) {
        if (output.count[d] == 0) { 
            output.mean[d] = opt.mean_placeholder;
        }
    }
}
/**
 * @endcond
 */

/**
 * Compute residual sums of squares (RSS) for each element of a chosen dimension of a `tatami::Matrix`, after skipping any NaNs.
 * This may use either Welford's method or the standard two-pass method,
 * depending on the dimension in `row` and the preferred access dimension of `p`.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
 *
 * @param row Whether to compute the RSS for each row.
 * If false, the RSS is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Buffers to output arrays.
 * On output, this will contain the row/column RSSs.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_, typename Count_>
void rss(bool row, const tatami::Matrix<Value_, Index_>& mat, RssBuffers<Output_, Count_>& output, const RssOptions<Output_>& opt) {
    if (mat.prefer_rows() == row) {
        rss_direct(row, mat, output, opt);
    } else {
        rss_running(row, mat, output, opt);
    }
}

/**
 * @brief Results of `skip_nan::rss()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
 */
template<typename Output_, typename Count_>
struct RssResult {
    /**
     * Vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample mean of each row/column.
     */
    std::vector<Output_> mean;

    /**
     * Vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the sample variance of each row/column.
     */
    std::vector<Output_> rss;

    /**
     * Vector of length equal to the appropriate dimension extent (rows for `row = true`, columns otherwise),
     * containing the number of unskipped observations in each row/column.
     */
    std::vector<Count_> count;
};

/**
 * Overload of `skip_nan::rss()` that allocates memory for the output arrays.
 *
 * @tparam Output_ Floating-point type of the output data.
 * @tparam Count_ Numeric type of the non-NaN counts.
 * This is typically an integer type.
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param row Whether to compute the RSS for each row.
 * If false, the RSS is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param opt Further options.
 *
 * @return The mean and RSS of each row/column.
 */
template<typename Output_ = double, typename Count_, typename Value_, typename Index_>
RssResult<Output_, Count_> rss(bool row, const tatami::Matrix<Value_, Index_>& mat, const RssOptions<Output_>& opt) {
    RssResult<Output_, Count_> output;
    const auto dim = (row ? mat.nrow() : mat.ncol());

    tatami::resize_container_to_Index_size(output.mean, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    tatami::resize_container_to_Index_size(output.rss, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    tatami::resize_container_to_Index_size(output.count, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );

    RssBuffers<Output_, Count_> buffers;
    buffers.mean = output.mean.data();
    buffers.rss = output.rss.data();
    buffers.count = output.count.data();

    rss(row, mat, buffers, opt);
    return output;
}

}

}

#endif
