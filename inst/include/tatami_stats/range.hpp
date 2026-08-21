#ifndef TATAMI_STATS_RANGE_HPP
#define TATAMI_STATS_RANGE_HPP

#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <type_traits>

#include "tatami/tatami.hpp"

/**
 * @file range.hpp
 *
 * @brief Compute row and column ranges from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Options for `range()`.
 */
struct RangeOptions {
    /**
     * Whether to check for NaNs in the input, and skip them.
     * If false, NaNs are assumed to be absent, and the behavior of the range calculation in the presence of NaNs is undefined.
     */
    bool skip_nan = false;

    /**
     * Number of threads to use when computing ranges across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<typename Value_>
constexpr Value_ choose_minimum_placeholder() {
    // Placeholder value 'x' is such that 'x >= y' is always true for any non-NaN 'y'.
    if constexpr(std::numeric_limits<Value_>::has_infinity) {
        return std::numeric_limits<Value_>::infinity();
    } else {
        return std::numeric_limits<Value_>::max();
    }
}

template<typename Value_>
constexpr Value_ choose_maximum_placeholder() {
    // Placeholder value 'x' is such that 'x <= y' is always true for any non-NaN 'y'.
    if constexpr(std::numeric_limits<Value_>::has_infinity) {
        return -std::numeric_limits<Value_>::infinity();
    } else {
        return std::numeric_limits<Value_>::lowest();
    }
}

template<typename Value_, typename Index_>
Value_ min_direct(const Value_* const ptr, const Index_ num, bool skip_nan) {
    return nanable_ifelse_with_value<Value_>(
        skip_nan,
        [&]() -> Value_ {
            auto current = choose_minimum_placeholder<Value_>(); 
            for (Index_ i = 0; i < num; ++i) {
                auto val = ptr[i];
                if (val < current) { // no need to explicitly handle NaNs, as any comparison with NaNs is always false.
                    current = val;
                }
            }
            return current;
        },
        [&]() -> Value_ {
            if (num) {
                return *std::min_element(ptr, ptr + num);
            } else {
                return choose_minimum_placeholder<Value_>();
            }
        }
    );
}

template<typename Value_, typename Index_>
Value_ max_direct(const Value_* ptr, const Index_ num, bool skip_nan) {
    return nanable_ifelse_with_value<Value_>(
        skip_nan,
        [&]() -> Value_ {
            auto current = choose_maximum_placeholder<Value_>(); 
            for (Index_ i = 0; i < num; ++i) {
                auto val = ptr[i];
                if (val > current) { // again, no need to explicitly handle NaNs, as any comparison with NaNs is always false.
                    current = val;
                }
            }
            return current;
        },
        [&]() -> Value_ {
            if (num) {
                return *std::max_element(ptr, ptr + num);
            } else {
                return choose_maximum_placeholder<Value_>();
            }
        }
    );
}

template<typename Value_, typename Index_>
Value_ min_direct(const Value_* value, const Index_ num_nonzero, const Index_ num_all, bool skip_nan) {
    if (num_nonzero) {
        auto candidate = min_direct(value, num_nonzero, skip_nan);
        if (num_nonzero < num_all) {
            if (candidate > 0) {
                candidate = 0;
            }
        }
        return candidate;
    } else if (num_all) {
        return 0;
    } else {
        return choose_minimum_placeholder<Value_>();
    }
}

template<typename Value_, typename Index_>
Value_ max_direct(const Value_* value, const Index_ num_nonzero, const Index_ num_all, bool skip_nan) {
    if (num_nonzero) {
        auto candidate = max_direct(value, num_nonzero, skip_nan);
        if (num_nonzero < num_all) {
            if (candidate < 0) {
                candidate = 0;
            }
        }
        return candidate;
    } else if (num_all) {
        return 0;
    } else {
        return choose_maximum_placeholder<Value_>();
    }
}
/**
 * @endcond
 */

/**
 * @brief Result buffers for `range()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 */
template<typename Output_>
struct RangeBuffers {
    /**
     * Pointer to an array of length equal to the number of rows/columns (depending on `row`).
     * After calling `range()`, this is filled with the minimum value for each row/column.
     */
    Output_* minimum;

    /**
     * Pointer to an array of length equal to the number of rows/columns (depending on `row`).
     * After calling `range()`, this is filled with the maximum value for each row/column.
     */
    Output_* maximum;
};

/**
 * @cond
 */
template<typename Value_, typename Index_, typename Output_>
void range_direct(bool row, const tatami::Matrix<Value_, Index_>& mat, RangeBuffers<Output_>& output, const RangeOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());

    if (mat.is_sparse()) {
        tatami::Options topt;
        topt.sparse_extract_index = false;
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<true>(mat, row, s, l, topt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(vbuffer.data(), NULL);
                output.minimum[x + s] = min_direct(out.value, out.number, otherdim, opt.skip_nan);
                output.maximum[x + s] = max_direct(out.value, out.number, otherdim, opt.skip_nan);
            }
        }, dim, opt.num_threads);

    } else {
        tatami::parallelize([&](int, Index_ s, Index_ l) -> void {
            auto ext = tatami::consecutive_extractor<false>(mat, row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
            for (Index_ x = 0; x < l; ++x) {
                auto ptr = ext->fetch(buffer.data());
                output.minimum[x + s] = min_direct(ptr, otherdim, opt.skip_nan);
                output.maximum[x + s] = max_direct(ptr, otherdim, opt.skip_nan);
            }
        }, dim, opt.num_threads);
    }
}

template<typename Value_, typename Index_, typename Output_>
void range_running(bool row, const tatami::Matrix<Value_, Index_>& mat, RangeBuffers<Output_>& output, const RangeOptions& opt) {
    const auto dim = (row ? mat.nrow() : mat.ncol());
    const auto otherdim = (row ? mat.ncol() : mat.nrow());
    const bool is_sparse = mat.is_sparse();

    const bool do_parallel = opt.num_threads > 1; 
    std::optional<std::vector<std::optional<std::vector<Output_> > > > all_partial_min, all_partial_max;
    if (do_parallel) {
        all_partial_min.emplace(sanisizer::cast<I<decltype(all_partial_min->size())> >(opt.num_threads - 1));
        all_partial_max.emplace(sanisizer::cast<I<decltype(all_partial_max->size())> >(opt.num_threads - 1));
    }

    constexpr auto min_placeholder = choose_minimum_placeholder<Value_>();
    constexpr auto max_placeholder = choose_maximum_placeholder<Value_>();
    if (opt.skip_nan || otherdim == 0) {
        // If we're not skipping NaNs and we have at least one dimension element,
        // the output arrays will be fully populated when thread 0 processes the first dimension element.
        std::fill_n(output.minimum, dim, min_placeholder);
        std::fill_n(output.maximum, dim, max_placeholder);
    }

    const auto nused = tatami::parallelize([&](int thread, Index_ s, Index_ l) -> void {
        Output_* min_ptr;
        Output_* max_ptr;
        std::optional<std::vector<Output_> > cur_min, cur_max;
        if (!do_parallel) {
            min_ptr = output.minimum;
            max_ptr = output.maximum;
        } else {
            if (thread == 0) {
                min_ptr = output.minimum;
                max_ptr = output.maximum;
            } else {
                cur_min.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(dim));
                cur_max.emplace(tatami::cast_Index_to_container_size<std::vector<Output_> >(dim));
                min_ptr = cur_min->data();
                max_ptr = cur_max->data();
            }
        }

        if (is_sparse) {
            tatami::Options topt;
            topt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true>(mat, !row, s, l, topt);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);
            auto nonzeros = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                auto out = ext->fetch(vbuffer.data(), ibuffer.data());

                if (x == 0) {
                    // For the first observed vector in each thread,
                    // we can optimize it a little as we don't need to read existing min/max.
                    nanable_ifelse<Value_>(
                        opt.skip_nan,
                        [&]() -> void {
                            for (Index_ i = 0; i < out.number; ++i) {
                                const auto val = out.value[i];
                                if (!std::isnan(val)) {
                                    const auto idx = out.index[i];
                                    min_ptr[idx] = val;
                                    max_ptr[idx] = val;
                                    ++nonzeros[idx];
                                }
                            }
                        },
                        [&]() -> void {
                            // For non-main threads, our thread-local buffer is already zeroed so no need to do this. 
                            if (!do_parallel || thread == 0) {
                                std::fill_n(min_ptr, dim, 0);
                                std::fill_n(max_ptr, dim, 0);
                            }
                            for (Index_ i = 0; i < out.number; ++i) {
                                const auto val = out.value[i];
                                const auto idx = out.index[i];
                                min_ptr[idx] = val;
                                max_ptr[idx] = val;
                                ++nonzeros[idx];
                            }
                        }
                    );

                } else {
                    for (Index_ i = 0; i < out.number; ++i) {
                        const auto val = out.value[i];
                        const auto idx = out.index[i];
                        auto& min_current = min_ptr[idx];
                        if (val < min_current) { // this should implicitly skip val=NaNs, as any NaN comparison will be false.
                            min_current = val;
                        }
                        auto& max_current = max_ptr[idx];
                        if (val > max_current) { // this should implicitly skip val=NaNs, as any NaN comparison will be false.
                            max_current = val;
                        }
                        ++nonzeros[idx];
                    }
                }
            }

            for (Index_ d = 0; d < dim; ++d) {
                if (l > nonzeros[d]) {
                    auto& min_current = min_ptr[d];
                    if (min_current > 0) {
                        min_current = 0;
                    }
                    auto& max_current = max_ptr[d];
                    if (max_current < 0) {
                        max_current = 0;
                    }
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(mat, !row, s, l);
            auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);

            for (Index_ x = 0; x < l; ++x) {
                auto ptr = ext->fetch(buffer.data());

                if (x == 0) {
                    // For the first observed vector in each thread,
                    // we can optimize it a little as we don't need to read existing min/max.
                    nanable_ifelse<Value_>(
                        opt.skip_nan,
                        [&]() -> void {
                            for (Index_ i = 0; i < dim; ++i) {
                                const auto val = ptr[i];
                                if (!std::isnan(val)) {
                                    min_ptr[i] = val;
                                    max_ptr[i] = val;
                                }
                            }
                        },
                        [&]() -> void {
                            std::copy_n(ptr, dim, min_ptr);
                            std::copy_n(ptr, dim, max_ptr);
                        }
                    );

                } else {
                    for (Index_ i = 0; i < dim; ++i) {
                        const auto val = ptr[i];
                        auto& min_current = min_ptr[i];
                        if (val < min_current) { // this should implicitly skip val=NaNs, as any NaN comparison will be false.
                            min_current = val;
                        }
                        auto& max_current = max_ptr[i];
                        if (val > max_current) { // this should implicitly skip val=NaNs, as any NaN comparison will be false.
                            max_current = val;
                        }
                    }
                }
            }
        }

        if (do_parallel) {
            if (thread > 0) {
                (*all_partial_min)[thread - 1] = std::move(cur_min);
                (*all_partial_max)[thread - 1] = std::move(cur_max);
            }
        }
    }, otherdim, opt.num_threads);

    if (do_parallel) {
        for (int u = 1; u < nused; ++u) {
            const auto& cur_min = *((*all_partial_min)[u - 1]);
            const auto& cur_max = *((*all_partial_max)[u - 1]);
            for (Index_ d = 0; d < dim; ++d) {
                if (output.minimum[d] > cur_min[d]) {
                    output.minimum[d] = cur_min[d];
                }
                if (output.maximum[d] < cur_max[d]) {
                    output.maximum[d] = cur_max[d];
                }
            }
        }
    }
}
/**
 * @endcond
 */

/**
 * Compute ranges for each element of a chosen dimension of a `tatami::Matrix`.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 *
 * @param row Whether to compute the range for each row.
 * If false, the range is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Buffers to output arrays.
 * On output, this will contain the row/column variances.
 * @param opt Further options.
 */
template<typename Value_, typename Index_, typename Output_>
void range(bool row, const tatami::Matrix<Value_, Index_>& mat, RangeBuffers<Output_>& output, const RangeOptions& opt) {
    if (mat.prefer_rows() == row) {
        range_direct(row, mat, output, opt);
    } else {
        range_running(row, mat, output, opt);
    }
}

/**
 * @brief Results of `range()`.
 *
 * @tparam Output_ Floating-point type of the output data.
 * This should be capable of storing NaNs.
 */
template<typename Output_>
struct RangeResult {
    /**
     * Vector of length equal to the number of rows/columns (depending on `row`),
     * containing the minimum value for each row/column.
     */
    std::vector<Output_> minimum;

    /**
     * Vector of length equal to the number of rows/columns (depending on `row`),
     * containing the maximum value for each row/column.
     */
    std::vector<Output_> maximum;
};

/**
 * Overload of `range()` that allocates memory for the minimum/maximum.
 *
 * @tparam Value_ Numeric type of the input data.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output data.
 * It is assumed that this is large enough to store the maxima/minima. 
 *
 * @param row Whether to compute the range for each row.
 * If false, the range is computed for each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param opt Further options.
 *
 * @return Minimum and maximum for each row/column.
 */
template<typename Value_, typename Index_, typename Output_ = Value_>
RangeResult<Output_> range(bool row, const tatami::Matrix<Value_, Index_>& mat, const RangeOptions& opt) {
    RangeResult<Output_> output;
    const auto dim = (row ? mat.nrow() : mat.ncol());
    tatami::resize_container_to_Index_size(output.minimum, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );
    tatami::resize_container_to_Index_size(output.maximum, dim
#ifdef TATAMI_STATS_TEST_DIRTY
        , -1
#endif
    );

    RangeBuffers<Output_> buffers;
    buffers.minimum = output.minimum.data();
    buffers.maximum = output.maximum.data();
    range(row, mat, buffers, opt);

    return output;
}

}

#endif
