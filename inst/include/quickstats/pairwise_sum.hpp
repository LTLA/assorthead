#ifndef QUICKSTATS_PAIRWISE_SUM_HPP
#define QUICKSTATS_PAIRWISE_SUM_HPP

#include <vector>
#include <cstddef>
#include <optional>
#include <cassert>
#include <array>
#include <stdexcept>

/**
 * @file pairwise_sum.hpp
 * @brief Pairwise summation.
 */

namespace quickstats {

/**
 * @brief Re-usable workspace for `pairwise_sum()` and `pairwise_sum_abstract()`.
 *
 * @tparam Output_ Numeric type of the sum.
 */
template<typename Output_ = double>
struct PairwiseSumWorkspace {
    /**
     * @cond
     */
    struct State {
        State(const std::size_t left_end) : left_end(left_end) {}
        std::size_t left_end;
        std::optional<Output_> left_sum;
    };
    std::vector<State> states;
    /**
     * @endcond
     */
};

/**
 * @brief Options for `pairwise_sum()` and `pairwise_sum_abstract()`.
 */
struct PairwiseSumOptions {
    /**
     * Maximum number of elements in the subarray to be summed in sequence.
     * This should be positive and no less than `2 * accumulators_`.
     * Larger values reduce the recursion overhead at the cost of weakening the mitigations from round-off error.
     */
    std::size_t max_sum_length = 128;
};

/**
 * @cond
 */
// Mathematically equivalent to std::accumulate but reorders summations for greater instruction-level parallelism.
// See performance tests in https://github.com/tatami-inc/test-multiplication/tree/master/other/accumulators.
template<typename Output_, std::size_t width_>
Output_ recursive_sum(std::array<Output_, width_>& dots) {
    if constexpr(width_ == 1) {
        return dots[0];
    } else if constexpr(width_ == 2) {
        return dots[0] + dots[1];
    } else {
        constexpr auto half_width = width_ / 2;
        std::array<Output_, half_width> tmp;
        for (std::size_t s = 0; s < half_width; ++s) { // Increase potential for vectorization.
            tmp[s] = dots[s] + dots[s + half_width];
        }
        if constexpr(width_ % 2 == 1) {
            return recursive_sum(tmp) + dots[width_ - 1];
        } else {
            return recursive_sum(tmp);
        }
    }
}
/**
 * @endcond
 */

/**
 * Perform pairwise summation on an abstract array of numeric elements.
 * The array is recursively halved until each subarray is no greater than `PairwiseSumOptions::max_sum_length`.
 * Elements in each subarray are summed in sequence, and the subtotals are then added together to obtain the total sum.
 * Compared to naive summation, this reduces round-off error from floating-point imprecision with minimal loss of performance. 
 *
 * @tparam accumulators_ Number of accumulators to sum each subarray.
 * This should be positive and should be no greater than `PairwiseSumOptions::max_sum_length / 2`.
 * It is strongly recommended that this is set to a power of 2.
 * Larger values improve instruction parallelization at the cost of increased binary size and potential register spills.
 * @tparam Input_ Function that accepts a `std::size_t` and returns an input value (typically numeric).
 * @tparam Output_ Numeric type of the sum.
 *
 * @param num_total Total number of observations.
 * @param input Function that accesses an abstract array of length `num_total`.
 * Specifically, it accepts an integer index in `[0, num_total)` and returns an input value to be summed.
 * This will be called exacly once for each integer in `[0, num_total)`.
 * @param work Workspace that can be re-used across multiple `pairwise_sum_abstract()` calls.
 * @param options Further options.
 *
 * @return Sum of all `input(i)` values for `i` in `[0, num_total)`.
 */
template<std::size_t accumulators_ = 4, class Input_, typename Output_>
Output_ pairwise_sum_abstract(const std::size_t num_total, Input_ input, PairwiseSumWorkspace<Output_>& work, const PairwiseSumOptions& options) {
    static_assert(accumulators_ > 0);

    work.states.clear();
    if (num_total < accumulators_) {
        if constexpr(accumulators_ == 1) {
            return 0;
        } else {
            Output_ out = 0;
            for (std::size_t i = 0; i < num_total; ++i) {
                out += input(i);
            }
            return out;
        }
    }

    const std::size_t limit = options.max_sum_length;
    if (accumulators_ > limit / 2) {
        throw std::runtime_error("'max_sum_length' must be greater than '2 * accumulators_'");
    }

    std::size_t start = 0, end = num_total; 
    Output_ out = 0;
    while (1) {
        const std::size_t len = end - start;
        if (len > limit) {
            work.states.emplace_back(end);
            end = start + len / 2;
            continue;
        }

        // 'start + accumulators_ <= end' should always be true. 
        //
        // Let's start by considering the initial case where 'num_total <= limit'.
        // As we already know that 'num_total >= accumulators_', we get 'start + accumulators_ == accumulators_ <= num_total == end'.
        //
        // Alright, what about the left-hand-side of the recursion?
        // We defined 'end = start + len / 2', and we already know that 'len > limit' and 'limit / 2 >= accumulators_';
        // hence, we know that that 'start + len / 2 >= start + accumulators_'.
        //
        // The right-hand-side of the recursion is easier as we know its length is greater than or equal to 'len / 2' (as integer division is truncating).
        // So, if the LHS fulfills the requirement, then the RHS must definitely fulfill it.
        assert(start + accumulators_ <= end);

        Output_ tmp;
        if constexpr(accumulators_ == 1) {
            tmp = input(start); // We know that start < end, so we can skip one addition. 
            for (std::size_t i = start + 1; i < end; ++i) {
                tmp += input(i);
            }

        } else {
            // This accumulator logic was originally implemented in https://github.com/tatami-inc/tatami_mult.
            // We added peeling as we can guarantee that we have enough observations and thus can omit the conditional.
            std::array<Output_, accumulators_> partials; 
            for (std::size_t a = 0; a < accumulators_; ++a) { // peeling the first loop as we know that start + accumulators_ <= end.
                partials[a] = input(start + a);
            }

            const std::size_t num_cycles = len / accumulators_;
            const std::size_t remainder = len % accumulators_;
            for (std::size_t c = 1; c < num_cycles; ++c) {
                for (std::size_t a = 0; a < accumulators_; ++a) {
                    const std::size_t idx = start + c * accumulators_ + a;
                    partials[a] += input(idx);
                }
            }

            // Technically, we could structure the splits to reduce the number of calls to the epilogue loops.
            // However, this would weaken the symmetry of the splitting and compromise the precision improvements.
            tmp = 0;
            for (std::size_t i = 0; i < remainder; ++i) {
                const std::size_t idx = start + num_cycles * accumulators_ + i;
                tmp += input(idx);
            }

            tmp += recursive_sum(partials);
        }

        start = end;
        while (work.states.size() && work.states.back().left_sum.has_value()) {
            tmp += *(work.states.back().left_sum);
            start = work.states.back().left_end;
            work.states.pop_back();
        }

        if (work.states.empty()) {
            out = tmp;
            break;
        }

        work.states.back().left_sum = tmp;
        end = work.states.back().left_end;
    }

    return out;
}

/**
 * Perform pairwise summation on an array, see `pairwise_sum_abstract()` for more details.
 *
 * @tparam accumulators_ Maximum number of elements to sum in sequence, see `pairwise_sum_abstract()` for details.
 * @tparam Input_ Numeric type of the input data.
 * @tparam Output_ Numeric type of the sum.
 *
 * @param num_total Total number of observations.
 * @param[in] ptr Pointer to an array of length `num_total`, containing the elements to be summed.
 * @param work Workspace that can be re-used across multiple `pairwise_sum()` calls.
 * @param options Further options.
 *
 * @return Sum of all elements in `[ptr, ptr + num_total)`.
 */
template<std::size_t accumulators_ = 4, typename Input_, typename Output_>
Output_ pairwise_sum(const std::size_t num_total, const Input_* const ptr, PairwiseSumWorkspace<Output_>& work, const PairwiseSumOptions& options) {
    return pairwise_sum_abstract<accumulators_>(
        num_total,
        [&](const std::size_t i) -> auto {
            return ptr[i];
        },
        work,
        options
    );
}

/* COMMENTS:
 * I tried to write a multi-threaded version of this where each direct summation was submitted to a separate worker until all workers were occupied,
 * and then added the results once they became available from each worker. 
 * This worksharing is fine-grained but imposes a high cost for inter-thread communication relative to the summation for small `limit_`. 
 * As a consequence, the performance of this multi-threaded version is worse than its serial counterpart.
 *
 * I could have implemented alternative approaches that involve less communication but require more memory.
 * For example, we could split elements into the subarrays ahead of time, distribute the summations to threads once, and then sum the results once all workers are done.
 * This greatly reduces the cross-talk between threads but requires an extra allocation to store the results.
 *
 * TBH, the easiest and most performant approach to parallelization is to just split your input array into one subarray per worker,
 * perform the sum within each worker for that subarray, and then add the sums afterwards.
 * This won't give exactly the same result as serial execution but we've crossed that bridge already.
 * (If exact results are required, we can split it into ceil(log2(num_workers)) subarrays,
 * which allows us to follow the same halving as pairwise_sum() to get the exact same result at the cost of suboptimal worksharing.)
 *
 * In any case, summation is already so fast that I don't think we need to spend a lot of effort in thinking about parallelization.
 * Especially given that, in real applications, the other threads will typically be occupied elsewhere.
 * Indeed, we don't deal with parallelization in other parts of this library, so it would be odd to implement it here. 
 *
 * I also have a sneaking suspicion that the serial code is already pseudo-parallelized via out-of-order execution,
 * where the next summation starts before the first one has ended.
 * I say this because pairwise_sum() somehow manages to be slightly faster than std::accumulate() in our R bindings.
 */

/**
 * @cond
 */
// For back-compatibility.
template<std::size_t limit_ = 128, std::size_t accumulators_ = 4, class Input_, class Modifier_, typename Output_>
Output_ pairwise_sum(const std::size_t num_total, const Input_* const ptr, Modifier_ mod, PairwiseSumWorkspace<Output_>& work) {
    return pairwise_sum_abstract<accumulators_>(
        num_total,
        [&](const std::size_t i) -> auto { 
            return mod(i, ptr[i]);
        },
        work,
        [&]{
            PairwiseSumOptions opt;
            opt.max_sum_length = limit_;
            return opt;
        }()
    );
}

template<std::size_t limit_ = 128, std::size_t accumulators_ = 4, typename Input_, typename Output_>
Output_ pairwise_sum(const std::size_t num_total, const Input_* const ptr, PairwiseSumWorkspace<Output_>& work) {
    return pairwise_sum<accumulators_>(
        num_total,
        ptr,
        work,
        [&]{
            PairwiseSumOptions opt;
            opt.max_sum_length = limit_;
            return opt;
        }()
    );
}
/**
 * @endcond
 */

}

#endif
