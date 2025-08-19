#ifndef SCRAN_QC_FIND_MEDIAN_MAD_H
#define SCRAN_QC_FIND_MEDIAN_MAD_H

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstddef>

#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file find_median_mad.hpp
 * @brief Compute the median and MAD from an array of values.
 */

namespace scran_qc {

/**
 * @brief Options for `find_median_mad()`.
 */
struct FindMedianMadOptions {
    /**
     * Whether to compute the median and MAD after log-transformation of the values.
     * This is useful for defining thresholds based on fold changes from the center.
     * If `true`, all values are assumed to be non-negative.
     */
    bool log = false;

    /**
     * Whether to only compute the median.
     * If true, `FindMedianMadResults::mad` will be set to NaN.
     */
    bool median_only = false;
};

/**
 * @brief Results of `find_median_mad()`.
 * @tparam Float_ Floating-point type. 
 */
template<typename Float_>
struct FindMedianMadResults {
    /**
     * @cond
     */
    FindMedianMadResults(Float_ m1, Float_ m2) : median(m1), mad(m2) {}
    FindMedianMadResults() = default;
    /**
     * @endcond
     */

    /**
     * Median.
     */
    Float_ median = 0;

    /**
     * Median absolute deviation.
     */
    Float_ mad = 0;
};

/**
 * Pretty much as it says on the can; calculates the median of an array of values first,
 * and uses the median to then compute the median absolute deviation (MAD) from that array.
 *
 * @tparam Float_ Floating-point type for input and output.
 *
 * @param num Number of observations.
 * @param[in] metrics Pointer to an array of observations of length `num`.
 * NaNs are ignored.
 * Array contents are arbitrarily modified on function return and should not be used afterwards.
 * @param options Further options.
 *
 * @return Median and MAD for `metrics`, possibly after log-transformation.
 */
template<typename Float_> 
FindMedianMadResults<Float_> find_median_mad(std::size_t num, Float_* metrics, const FindMedianMadOptions& options) {
    static_assert(std::is_floating_point<Float_>::value);

    // Rotate all the NaNs to the front of the buffer and ignore them.
    decltype(num) lost = 0;
    for (decltype(num) i = 0; i < num; ++i) {
        if (std::isnan(metrics[i])) {
            std::swap(metrics[i], metrics[lost]);
            ++lost;
        }
    }
    metrics += lost;
    num -= lost;

    if (options.log) {
        for (decltype(num) i = 0; i < num; ++i) {
            auto& val = metrics[i];
            if (val > 0) {
                val = std::log(val);
            } else if (val == 0) {
                val = -std::numeric_limits<double>::infinity();
            } else {
                throw std::runtime_error("cannot log-transform negative values");
            }
        }
    }

    // No need to skip the NaNs again.
    auto median = tatami_stats::medians::direct<Float_>(metrics, num, /* skip_nan = */ false);

    if (options.median_only || std::isnan(median)) {
        // Giving up.
        return FindMedianMadResults<Float_>(median, std::numeric_limits<Float_>::quiet_NaN());
    } else if (std::isinf(median)) {
        // MADs should be no-ops when added/subtracted from infinity. Any
        // finite value will do here, so might as well keep it simple.
        return FindMedianMadResults<Float_>(median, static_cast<Float_>(0));
    }

    // As an aside, there's no way to avoid passing in 'metrics' as a Float_,
    // even if the original values were integers, because we need to do this
    // subtraction here that could cast integers to floats. So at some point we
    // will need a floating-point buffer, and so we might as well just pass the
    // metrics in as floats in the first place. Technically the first sort
    // could be done with an integer buffer but then we'd need an extra argument.

    for (decltype(num) i = 0; i < num; ++i) {
        metrics[i] = std::abs(metrics[i] - median);
    }
    auto mad = tatami_stats::medians::direct<Float_>(metrics, num, /* skip_nan = */ false);
    mad *= 1.4826; // for equivalence with the standard deviation under normality.

    return FindMedianMadResults<Float_>(median, mad);
}

/**
 * Overload of `find_median_mad()` that uses an auxiliary buffer to avoid mutating the input array of values.
 *
 * @tparam Value_ Type for the input.
 * @tparam Float_ Floating-point type for output.
 *
 * @param num Number of observations.
 * @param[in] metrics Pointer to an array of observations of length `num`.
 * NaNs are ignored.
 * Array contents are arbitrarily modified on function return and should not be used afterwards.
 * @param[out] buffer Pointer to an array of length `num`, containing a buffer to use for storing intermediate results.
 * This can also be NULL in which case a buffer is allocated.
 * @param options Further options.
 *
 * @return Median and MAD for `metrics`, possibly after log-transformation.
 */
template<typename Float_ = double, typename Value_> 
FindMedianMadResults<Float_> find_median_mad(std::size_t num, const Value_* metrics, Float_* buffer, const FindMedianMadOptions& options) {
    std::vector<Float_> xbuffer;
    if (buffer == NULL) {
        xbuffer.resize(sanisizer::cast<decltype(xbuffer.size())>(num)
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        buffer = xbuffer.data();
    }
    std::copy_n(metrics, num, buffer);
    return find_median_mad(num, buffer, options);
}

/**
 * @brief Temporary data structures for `find_median_mad_blocked()`.
 *
 * This can be re-used across multiple `find_median_mad_blocked()` calls to avoid reallocation.
 *
 * @tparam Float_ Floating-point type for buffering.
 */
template<typename Float_>
class FindMedianMadWorkspace {
public:
    /**
     * @tparam Block_ Integer type for the block identifiers.
     * @param num Number of observations.
     * @param[in] block Pointer to an array of block identifiers. 
     * The array should be of length equal to `num`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     */
    template<typename Block_>
    FindMedianMadWorkspace(std::size_t num, const Block_* block) : my_buffer(num) {
        set(num, block);
    }

    /**
     * Default constructor.
     */
    FindMedianMadWorkspace() = default;

    /**
     * @tparam Block_ Integer type for the block identifiers.
     * @param num Number of observations.
     * @param[in] block Pointer to an array of block identifiers.
     * The array should be of length equal to `num`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     */
    template<typename Block_>
    void set(std::size_t num, const Block_* block) {
        my_block_starts.clear();

        if (block) { 
            for (decltype(num) i = 0; i < num; ++i) {
                auto candidate = block[i];
                if (sanisizer::is_greater_than_or_equal(candidate, my_block_starts.size())) {
                    my_block_starts.resize(sanisizer::sum<decltype(my_block_starts.size())>(candidate, 1));
                }
                ++my_block_starts[candidate];
            }

            std::size_t sofar = 0;
            for (auto& s : my_block_starts) {
                auto last = sofar;
                sofar += s;
                s = last;
            }
        }

        my_buffer.resize(sanisizer::cast<decltype(my_buffer.size())>(num)
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
        my_block_ends.resize(my_block_starts.size()
#ifdef SCRAN_QC_TEST_INIT
            , SCRAN_QC_TEST_INIT
#endif
        );
    }

/**
 * @cond
 */
public:
    // Can't figure out how to make compute_blocked() a friend,
    // so these puppies are public for simplicity.
    std::vector<Float_> my_buffer;
    std::vector<std::size_t> my_block_starts;
    std::vector<std::size_t> my_block_ends;
/**
 * @endcond
 */
};

/**
 * For blocked datasets, this function computes the median and MAD for each block.
 * It is equivalent to calling `find_median_mad()` separately on all observations from each block.
 *
 * @tparam Output_ Floating-point type for the output.
 * @tparam Block_ Integer type, containing the block IDs.
 * @tparam Value_ Numeric type for the input.
 *
 * @param num Number of observations.
 * @param[in] metrics Pointer to an array of observations of length `num`.
 * NaNs are ignored.
 * @param[in] block Optional pointer to an array of block identifiers.
 * If provided, the array should be of length equal to `num`.
 * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
 * If a null pointer is supplied, all observations are assumed to belong to the same block.
 * @param workspace Pointer to a workspace object, either (i) constructed on `num` and `block` or (ii) configured using `FindMedianMadWorkspace::set()` on `num` and `block`.
 * The same object can be re-used across multiple calls to `find_median_mad_blocked()` with the same `num` and `block`.
 * This can also be NULL in which case a new workspace is allocated. 
 * @param options Further options.
 *
 * @return Vector of length \f$N\f$, where each entry contains the median and MAD for each block in `block`.
 */
template<typename Output_ = double, typename Value_, typename Block_>
std::vector<FindMedianMadResults<Output_> > find_median_mad_blocked(
    std::size_t num,
    const Value_* metrics, 
    const Block_* block,
    FindMedianMadWorkspace<Output_>* workspace,
    const FindMedianMadOptions& options)
{
    std::unique_ptr<FindMedianMadWorkspace<Output_> > xworkspace;
    if (workspace == NULL) {
        xworkspace = std::make_unique<FindMedianMadWorkspace<Output_> >(num, block);
        workspace = xworkspace.get();
    }

    std::vector<FindMedianMadResults<Output_> > output;

    auto& buffer = workspace->my_buffer;
    if (!block) {
        std::copy_n(metrics, num, buffer.begin());
        output.push_back(find_median_mad(num, buffer.data(), options));
        return output;
    }

    const auto& starts = workspace->my_block_starts;
    auto& ends = workspace->my_block_ends;
    std::copy(starts.begin(), starts.end(), ends.begin());
    for (decltype(num) i = 0; i < num; ++i) {
        auto& pos = ends[block[i]];
        buffer[pos] = metrics[i];
        ++pos;
    }

    // Using the ranges on the buffer.
    auto nblocks = starts.size();
    output.reserve(nblocks);
    for (decltype(nblocks) g = 0; g < nblocks; ++g) {
        output.push_back(find_median_mad(ends[g] - starts[g], buffer.data() + starts[g], options));
    }

    return output;
}

}

#endif
