#ifndef TATAMI_STATS__UTILS_HPP
#define TATAMI_STATS__UTILS_HPP

#include <vector>
#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

/**
 * @file utils.hpp
 *
 * @brief Utilities for computing matrix statistics.
 */

namespace tatami_stats {

/**
 * Count the total number of groups, typically for per-group memory allocations.
 *
 * @tparam Group_ Integer type for the group assignments.
 *
 * @param[in] group Pointer to an array of group assignments per observation.
 * Each assignment should be an integer in \f$[0, G)\f$ where \f$G\f$ is the total number of groups.
 * @param n Number of observations, i.e., the length of the array referenced by `group`.
 *
 * @return Total number of groups, i.e., \f$G\f$.
 * Note that not all groups may actually have non-zero occurrences in `group`.
 */
template<typename Group_>
std::size_t total_groups(const Group_* group, std::size_t n) {
    if (n) {
        return sanisizer::sum<std::size_t>(*std::max_element(group, group + n), 1);
    } else {
        return 0;
    }
}

/**
 * Count the occurrences of each group.
 *
 * @tparam Group_ Integer type for the group assignments.
 * @tparam Size_ Integer type for the number of observations.
 *
 * @param[in] group Pointer to an array of group assignments per observation.
 * Each assignment should be an integer in \f$[0, G)\f$ where \f$G\f$ is the total number of groups.
 * @param n Number of observations, i.e., the length of the array referenced by `group`.
 *
 * @return Vector of length equal to \f$G\f$, containing the number of occurrences of each group.
 */
template<typename Group_, typename Size_>
std::vector<Size_> tabulate_groups(const Group_* group, Size_ n) {
    auto ngroups = total_groups(group, n);
    std::vector<Size_> group_sizes(ngroups);
    for (Size_ r = 0; r < n; ++r) {
        ++(group_sizes[group[r]]);
    }
    return group_sizes;
}

/**
 * @brief Local output buffer for running calculations.
 *
 * A common parallelization scheme involves dividing the set of objective vectors into contiguous blocks, where each thread operates on a block at a time.
 * However, in running calculations, an entire block's statistics are updated when its corresponding thread processes an observed vector.
 * If these statistics are stored in a global output buffer, false sharing at the boundaries of the blocks can degrade performance.
 *
 * To mitigate false sharing, we create a separate `std::vector` in each thread to store its output statistics.
 * The aim is to give the memory allocator an opportunity to store each thread's vector contents at non-contiguous addresses on the heap.
 * (While not guaranteed, well-separated addresses are observed on many compiler/architecture combinations, presumably due to the use of multiple arenas -
 * see https://github.com/tatami-inc/tatami_stats/issues/9 for testing.)
 * Once the calculations are finished, each thread can transfer its statistics to the global buffer.
 *
 * The `LocalOutputBuffer` is just a wrapper around a `std::vector` with some special behavior for the first thread.
 * Specifically, the first thread is allowed to directly write to the global buffer.
 * This avoids any extra allocation in the serial case where there is no need to protect against false sharing.
 *
 * @tparam Output_ Type of the result.
 */
template<typename Output_>
class LocalOutputBuffer {
public:
    /**
     * @tparam Index_ Type of the start index and length.
     * @param thread Identity of the thread, starting from zero to the total number of threads.
     * @param start Index of the first objective vector in the contiguous block for this thread.
     * @param length Number of objective vectors in the contiguous block for this thread.
     * @param[out] output Pointer to the global output buffer.
     * @param fill Initial value to fill the buffer.
     */
    template<typename Index_>
    LocalOutputBuffer(int thread, Index_ start, Index_ length, Output_* output, Output_ fill) : 
        my_output(output + sanisizer::cast<std::size_t>(start)),
        use_local(thread > 0),
        my_buffer(use_local ? sanisizer::cast<decltype(my_buffer.size())>(length) : static_cast<decltype(my_buffer.size())>(0), fill)
    {
        if (!use_local) {
            // Setting to zero to match the initial behavior of 'my_buffer' when 'use_local = true'.
            std::fill_n(my_output, length, fill);
        }
    }

    /**
     * Overloaded constructor that sets the default `fill = 0`.
     *
     * @tparam Index_ Type of the start index and length.
     * @param thread Identity of the thread, starting from zero to the total number of threads.
     * @param start Index of the first objective vector in the contiguous block for this thread.
     * @param length Number of objective vectors in the contiguous block for this thread.
     * @param[out] output Pointer to the global output buffer.
     */
    template<typename Index_>
    LocalOutputBuffer(int thread, Index_ start, Index_ length, Output_* output) : LocalOutputBuffer(thread, start, length, output, 0) {}

    /**
     * Default constructor.
     */
    LocalOutputBuffer() = default;

    /**
     * @return Pointer to an output buffer to use for this thread.
     * This contains at least `length` addressable elements (see the argument of the same name in the constructor). 
     * For `thread = 0`, this will be equal to `output + start`.
     */
    Output_* data() {
        return (use_local ? my_buffer.data() : my_output);
    }

    /**
     * @return Const pointer to an output buffer to use for this thread.
     * This contains at least `length` addressable elements (see the argument of the same name in the constructor). 
     * For `thread = 0`, this will be equal to `output + start`.
     */
    const Output_* data() const {
        return (use_local ? my_buffer.data() : my_output);
    }

    /**
     * Transfer results from the local buffer to the global buffer (i.e., `output` in the constructor).
     * For `thread = 0`, this will be a no-op.
     */
    void transfer() {
        if (use_local) {
            std::copy(my_buffer.begin(), my_buffer.end(), my_output);
        }
    }

private:
    Output_* my_output = NULL;
    bool use_local = false;
    std::vector<Output_> my_buffer;
};

/**
 * @brief Local output buffers for running calculations.
 *
 * Zero, one or more local output buffers to be created in each thread to avoid false sharing.
 * This class is equivalent to a vector of `LocalOutputBuffer` instances, but is easier to initialize and more memory-efficient.
 * In particular, no vector is created at all for the first thread, avoiding an unnecessary allocation in the serial case.
 *
 * @tparam Output_ Type of the result.
 * @tparam GetOutput_ Functor object that returns a pointer to the output buffer.
 */
template<typename Output_, class GetOutput_>
class LocalOutputBuffers {
public:
    /**
     * @tparam Number_ Integer type of the number of buffers.
     * @tparam Index_ Integer type of the start index and length.
     *
     * @param thread Identity of the thread, starting from zero to the total number of threads.
     * @param number Number of output buffers.
     * @param start Index of the first objective vector in the contiguous block for this thread.
     * @param length Number of objective vectors in the contiguous block for this thread.
     * @param outfun Function that accepts an `std::size_t` specifying the index of an output buffer in `[0, number)` and returns a `Output_*` pointer to that buffer.
     * @param fill Initial value to fill the buffer.
     */
    template<typename Number_, typename Index_>
    LocalOutputBuffers(int thread, Number_ number, Index_ start, Index_ length, GetOutput_ outfun, Output_ fill) : 
        my_number(sanisizer::cast<decltype(my_number)>(number)),
        my_start(sanisizer::cast<decltype(my_start)>(start)),
        my_use_local(thread > 0),
        my_getter(std::move(outfun))
    {
        if (thread == 0) {
            for (decltype(my_number) i = 0; i < my_number; ++i) {
                // Setting to the fill to match the initial behavior of 'my_buffer' when 'thread > 0'.
                std::fill_n(my_getter(i) + my_start, length, fill);
            }
        } else {
            my_buffers.reserve(my_number);
            for (decltype(my_number) i = 0; i < my_number; ++i) {
                my_buffers.emplace_back(tatami::can_cast_Index_to_container_size<typename decltype(my_buffers)::value_type>(length), fill);
            }
        }
    }

    /**
     * Overloaded constructor that sets the default `fill = 0`.
     *
     * @tparam Number_ Integer type of the number of buffers.
     * @tparam Index_ Integer type of the start index and length.
     *
     * @param thread Identity of the thread, starting from zero to the total number of threads.
     * @param number Number of output buffers.
     * @param start Index of the first objective vector in the contiguous block for this thread.
     * @param length Number of objective vectors in the contiguous block for this thread.
     * @param outfun Function that accepts an `Index_` specifying the index of an output buffer in `[0, number)` and returns a `Output_*` pointer to that buffer.
     */
    template<typename Number_, typename Index_>
    LocalOutputBuffers(int thread, Number_ number, Index_ start, Index_ length, GetOutput_ outfun) :
        LocalOutputBuffers(thread, number, start, length, std::move(outfun), 0) {}

    /**
     * Default constructor.
     */
    LocalOutputBuffers() = default;

    /**
     * @return Number of output buffers.
     * This is the same as `number` in the constructor.
     */
    std::size_t size() const {
        return my_number;
    }

    /**
     * @param i Index of the output buffer.
     * @return Pointer to the `i`-th output buffer to use in this thread.
     * This contains at least `length` addressable elements (see the argument of the same name in the constructor). 
     * For `thread = 0`, this will be equal to `outfun(i) + start`.
     */
    Output_* data(std::size_t i) {
        return (my_use_local ? my_buffers[i].data() : my_getter(i) + my_start);
    }

    /**
     * @param i Index of the output buffer.
     * @return Const pointer to the `i`-th output buffer to use for this thread.
     * This contains at least `length` addressable elements (see the argument of the same name in the constructor). 
     * For `thread = 0`, this will be equal to `outfun(i) + start`.
     */
    const Output_* data(std::size_t i) const {
        return (my_use_local ? my_buffers[i].data() : my_getter(i) + my_start);
    }

    /**
     * Transfer results from the local buffer to the global buffer (i.e., `outfun(i)` for `i` in `[0, number)` from the constructor).
     * For `thread = 0`, this will be a no-op.
     */
    void transfer() {
        if (my_use_local) {
            for (decltype(my_number) i = 0; i < my_number; ++i) {
                const auto& current = my_buffers[i];
                std::copy(current.begin(), current.end(), my_getter(i) + my_start);
            }
        }
    }

private:
    std::size_t my_number = 0;
    std::size_t my_start = 0;
    bool my_use_local = true;
    std::vector<std::vector<Output_> > my_buffers;
    GetOutput_ my_getter;
};

/**
 * @cond
 */
namespace internal {

template<typename Value_, class If_, class Else_>
void nanable_ifelse(bool skip_nan, If_ iffun, Else_ elsefun) {
    if constexpr(std::numeric_limits<Value_>::has_quiet_NaN) {
        if (skip_nan) {
            iffun();
            return;
        }
    }
    elsefun();
}

template<typename Value_, class If_, class Else_>
auto nanable_ifelse_with_value(bool skip_nan, If_ iffun, Else_ elsefun) {
    if constexpr(std::numeric_limits<Value_>::has_quiet_NaN) {
        if (skip_nan) {
            return iffun();
        }
    }
    return elsefun();
}

}
/**
 * @endcond
 */

}

#endif
