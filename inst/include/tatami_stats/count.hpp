#ifndef TATAMI_STATS_COUNT_HPP
#define TATAMI_STATS_COUNT_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file count.hpp
 *
 * @brief Compute row and column counts from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `count()`.
 */
struct CountOptions {
    /**
     * Number of threads to use when counting across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Output_, class Condition_>
void count_direct(const bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* const output, Condition_ condition, const CountOptions& opt) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    const Index_ otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.sparse()) {
        tatami::Options topt;
        topt.sparse_ordered_index = false;
        const bool count_zero = condition(0);

        tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, start, len, topt);
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);

            for (Index_ x = 0; x < len; ++x) {
                auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                Output_ target = 0;
                for (Index_ j = 0; j < range.number; ++j) {
                    target += condition(range.value[j]);
                }
                if (count_zero) {
                    target += otherdim - range.number;
                }
                output[x + start] = target;
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            auto ext = tatami::consecutive_extractor<false>(mat, row, start, len);

            for (Index_ x = 0; x < len; ++x) {
                auto ptr = ext->fetch(xbuffer.data());
                Output_ target = 0;
                for (Index_ j = 0; j < otherdim; ++j) {
                    target += condition(ptr[j]);
                }
                output[x + start] = target;
            }
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Output_, class Condition_>
void count_running(const bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* const output, Condition_ condition, const CountOptions& opt) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    const Index_ otherdim = (row ? mat.ncol() : mat.nrow());

    const bool do_parallel = opt.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > all_partial_count;
    if (do_parallel) {
        all_partial_count.emplace(sanisizer::cast<I<decltype(all_partial_count->size())> >(opt.num_threads - 1));
    }

    // Checking if we should count zeros in the sparse case.
    const bool is_sparse = mat.is_sparse();
    const bool count_zero = is_sparse && condition(0);

    std::fill_n(output, dim, 0);

    const int num_used = tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
        Output_* out_ptr; 
        std::optional<std::vector<Output_> > cur_count;
        if (!do_parallel) {
            out_ptr = output;
        } else {
            // Directly write the result to the output buffer for the first thread to save ourselves an allocation.
            // Everything else goes into these temporary vectors.
            if (thread == 0) {
                out_ptr = output;
            } else {
                cur_count.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(dim));
                out_ptr = cur_count->data();
            }
        }

        if (is_sparse) {
            tatami::Options topt;
            topt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, !row, start, len, topt);
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);
            auto nonzeros = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);

            for (Index_ x = 0; x < len; ++x) {
                auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                for (Index_ j = 0; j < range.number; ++j) {
                    auto idx = range.index[j];
                    out_ptr[idx] += condition(range.value[j]);
                    ++(nonzeros[idx]);
                }
            }

            if (count_zero) {
                for (Index_ d = 0; d < dim; ++d) {
                    out_ptr[d] += len - nonzeros[d];
                }
            }

        } else {
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ext = tatami::consecutive_extractor<false>(mat, !row, start, len);

            for (Index_ x = 0; x < len; ++x) {
                auto ptr = ext->fetch(xbuffer.data());
                for (Index_ d = 0; d < dim; ++d) {
                    out_ptr[d] += condition(ptr[d]);
                }
            }
        }

        if (thread) {
            (*all_partial_count)[thread - 1] = std::move(cur_count);
        }
    }, otherdim, opt.num_threads);

    if (do_parallel) {
        // Skip the first thread as we already put its counts in 'output'.
        for (int u = 1; u < num_used; ++u) {
            const auto& curout = *((*all_partial_count)[u - 1]);
            for (Index_ d = 0; d < dim; ++d) {
                output[d] += curout[d];
            }
        }
    }
}
/**
 * @endcond
 */

/**
 * Count the number of values that satisfy the `condition` in each element of a chosen dimension.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`.
 * @tparam Condition_ Function that accepts a single `Value_` and returns a `bool`.
 *
 * @param row Whether to perform the count within each row.
 * If false, the count is performed within each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column counts.
 * @param condition Function to indicate whether a value should be counted. 
 * This function is responsible for handling any NaNs that might be present in `p`.
 * This function should be thread-safe.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_, class Condition_>
void count(const bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* const output, Condition_ condition, const CountOptions& opt) {
    if (mat.prefer_rows() == row) {
        count_direct(row, mat, output, std::move(condition), opt);
    } else {
        count_running(row, mat, output, std::move(condition), opt);
    }
}

/**
 * Overload of `count()` that allocates memory for the output vector.
 *
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`.
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Condition_ Function that accepts a single `Value_` and returns a `bool`.
 *
 * @param row Whether to perform the count within each row.
 * If false, the count is performed within each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param condition Function to indicate whether a value should be counted. 
 * This function is also responsible for handling any NaNs that might be present in `p`.
 * This function should be thread-safe.
 * @param opt Further options.
 */
template<typename Output_, typename Value_, typename Index_, class Condition_>
std::vector<Output_> count(const bool row, const tatami::Matrix<Value_, Index_>& mat, Condition_ condition, const CountOptions& opt) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    auto output = sanisizer::create<std::vector<Output_> >(dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    count(row, mat, output.data(), std::move(condition), opt);
    return output;
}

}

#endif
