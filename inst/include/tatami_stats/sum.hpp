#ifndef TATAMI_STATS_SUM_HPP
#define TATAMI_STATS_SUM_HPP

#include "utils.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <optional>
#include <cmath>

#include "tatami/tatami.hpp"

/**
 * @file sum.hpp
 *
 * @brief Compute row and column sums from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `sum()`.
 */
struct SumOptions {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of summation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing sums across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Output_>
void sum_direct(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const SumOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.is_sparse()) {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            tatami::Options topt;
            topt.sparse_extract_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, topt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);

            nanable_ifelse<Value_>(
                opt.skip_nan,
                [&]() -> void {
                    for (Index_ x = 0; x < l; ++x) {
                        const auto out = ext->fetch(vbuffer.data(), NULL);
                        Output_ sum = 0;
                        for (Index_ i = 0; i < out.number; ++i) {
                            const auto val = out.value[i];
                            if (!std::isnan(val)) {
                                sum += val;
                            }
                        }
                        output[x + s] = sum;
                    }
                },
                [&]() -> void {
                    quickstats::PairwiseSumWorkspace<Output_> work;
                    for (Index_ x = 0; x < l; ++x) {
                        const auto out = ext->fetch(vbuffer.data(), NULL);
                        output[x + s] = quickstats::pairwise_sum(out.number, out.value, work); // Index_ -> size_t conversion is safe, as per tatami's contract.
                    }
                }
            );
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);

            nanable_ifelse<Value_>(
                opt.skip_nan,
                [&]() -> void {
                    for (Index_ x = 0; x < l; ++x) {
                        const auto ptr = ext->fetch(buffer.data());
                        Output_ sum = 0;
                        for (Index_ i = 0; i < otherdim; ++i) {
                            const auto val = ptr[i];
                            if (!std::isnan(val)) {
                                sum += val;
                            }
                        }
                        output[x + s] = sum;
                    }
                },
                [&]() -> void {
                    quickstats::PairwiseSumWorkspace<Output_> work;
                    for (Index_ x = 0; x < l; ++x) {
                        const auto ptr = ext->fetch(buffer.data());
                        output[x + s] = quickstats::pairwise_sum(otherdim, ptr, work); // Index_ -> size_t conversion is safe, as per tatami's contract.
                    }
                }
            );
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Output_>
void sum_running(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const SumOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    const bool do_parallel = (opt.num_threads > 1);
    std::optional<std::vector<std::optional<std::vector<Output_> > > > all_partial_sum;
    if (do_parallel) {
        all_partial_sum.emplace(sanisizer::cast<I<decltype(all_partial_sum->size())> >(opt.num_threads - 1));
    }

    std::fill_n(output, dim, 0);

    const auto nused = tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
        Output_* sum_ptr;
        std::optional<std::vector<Output_> > cur_sum;
        if (!do_parallel) {
            sum_ptr = output;
        } else {
            if (thread == 0) {
                sum_ptr = output;
            } else {
                cur_sum.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(dim));
                sum_ptr = cur_sum->data();
            }
        }

        if (mat.is_sparse()) {
            tatami::Options topt;
            topt.sparse_ordered_index = false; // ordering doesn't matter.
            auto ext = tatami::consecutive_extractor<true>(mat, !row, s, l, topt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                const auto out = ext->fetch(vbuffer.data(), ibuffer.data());
                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        for (Index_ i = 0; i < out.number; ++i) {
                            const auto val = out.value[i];
                            if (!std::isnan(val)) {
                                sum_ptr[out.index[i]] += val;
                            }
                        }
                    },
                    [&]() -> void {
                        for (Index_ i = 0; i < out.number; ++i) {
                            sum_ptr[out.index[i]] += out.value[i];
                        }
                    }
                );
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, !row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                const auto ptr = ext->fetch(buffer.data());
                nanable_ifelse<Value_>(
                    opt.skip_nan,
                    [&]() -> void {
                        for (Index_ i = 0; i < dim; ++i) {
                            const auto val = ptr[i];
                            if (!std::isnan(val)) {
                                sum_ptr[i] += val;
                            }
                        }
                    },
                    [&]() -> void {
                        for (Index_ i = 0; i < dim; ++i) {
                            sum_ptr[i] += ptr[i];
                        }
                    }
                );
            }
        }

        if (do_parallel) {
            if (thread > 0) {
                (*all_partial_sum)[thread - 1] = std::move(cur_sum);
            }
        }
    }, otherdim, opt.num_threads);

    if (do_parallel) {
        for (int u = 1; u < nused; ++u) {
            const auto& cur_sum = *((*all_partial_sum)[u - 1]);
            for (Index_ d = 0; d < dim; ++d) {
                output[d] += cur_sum[d];
            }
        }
    }
}
/**
 * @endcond
 */

/**
 * Compute sums for each element of a chosen dimension of a `tatami::Matrix`.
 * This may either use pairwise summation or direct accumulation,
 * depending on the requested dimension in `row`, the preferred dimension for access in `p` and whether NaNs are to be skipped.
 * It is best to use a sufficiently high-precision `Output_` to mitigate round-off errors.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 *
 * @param row Whether to compute the sum for each row.
 * If false, the sum is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column sums.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_>
void sum(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const SumOptions& opt) {
    if (mat.prefer_rows() == row) {
        sum_direct(row, mat, output, opt);
    } else {
        sum_running(row, mat, output, opt);
    }
}

/**
 * Overload of `sum()` that allocates memory for the output sums.
 *
 * @tparam Output_ Numeric type of the output value.
 * It is assumed that this is large enough to store the sums. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param row Whether to compute the sum for each row.
 * If false, the sum is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param opt Further options.
 *
 * @return Vector of length equal to the number of rows (if `row = true`) or columns (otherwise),
 * containing the row/column sums.
 */
template<typename Output_ = double, typename Value_, typename Index_>
std::vector<Output_> sum(bool row, const tatami::Matrix<Value_, Index_>& mat, const SumOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    auto output = sanisizer::create<std::vector<Output_> >(dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    sum(row, mat, output.data(), opt);
    return output;
}

}

#endif
