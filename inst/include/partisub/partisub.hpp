#ifndef PARTISUB_HPP
#define PARTISUB_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <random>

#include "sanisizer/sanisizer.hpp"
#include "aarand/aarand.hpp"

/**
 * @file partisub.hpp
 * @brief Subsampling in partitions.
 */

/**
 * @namespace partisub 
 * @brief Subsampling in partitions.
 */
namespace partisub {

/**
 * @brief Options for `compute()`.
 */
struct Options {
    /**
     * Whether to force non-empty partitions to be sampled at least once.
     * If `false`, there is no guarantee that small non-empty partitions will be represented by an observation in the selected subset.
     */
    bool force_non_empty = true;

    /**
     * Seed for the random number generator.
     */
    unsigned long long seed = 12345;
};

/**
 * Subsample observations within each partition.
 *
 * Consider a dataset where observations are grouped into discrete partitions, e.g., clusters, factors.
 * We would like to sample a subset of observations for further analysis, typically in time-consuming steps where the full dataset would be too large.
 * `compute()` creates a subset of the specified `target` size while ensuring that each partition is represented.
 * Specifically:
 *
 * - Each non-empty partition will always be represented by at least one of its constituent observations in the sampled subset.
 *   This ensures that even small partitions will be present in the subset.
 *   As a result, though, the reported number of observations may exceed `target` if there are many small partitions.
 *   Can be disabled by setting `Options::force_non_empty = false`.
 * - The number of observations sampled from each partition is roughly proportional to the size of the partition.
 *   More specifically, the sampling is done within each partition to minimize the effect of sampling noise on the relative partition frequencies.
 *   This aims to preserve differences in frequencies across partitions so that the subset accurately reflects the full dataset.
 *   Otherwise, any discrepancies may make it difficult to extrapolate the subset's results to the full dataset.
 * - All observations are returned if the requested `target` is greater than or equal to `num_obs`.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Partition_ Integer type of the partition assignments.
 *
 * @param num_obs Number of observations, should be non-negative.
 * @param[in] partition Pointer to an array of length `num_obs` containing partition assignments.
 * @param target Desired number of observations in the subset.
 * Note that the actual number of observations returned in `output` may be different.
 * @param options Further options.
 * @param[out] output Vector of indices for observations selected in the subset.
 * Indices are guaranteed to be unique and sorted.
 */
template<typename Index_, typename Partition_>
void compute(const Index_ num_obs, const Partition_* partition, const Index_ target, const Options& options, std::vector<Index_>& output) {
    if (target >= num_obs) {
        sanisizer::resize(output, num_obs);
        std::iota(output.begin(), output.end(), static_cast<Index_>(0));
        return;
    }

    // num_obs >= 0 at this point otherwise target >= num_obs would be true.
    const Partition_ num_partitions = sanisizer::sum<Partition_>(*std::max_element(partition, partition + num_obs), 1);

    auto partition_count = sanisizer::create<std::vector<Index_> >(num_partitions);
    for (Index_ o = 0; o < num_obs; ++o) {
        const auto part = partition[o];
        partition_count[part] += 1;
    }

    std::mt19937_64 rng(options.seed);

    // We compute the number of observations to take from each partition.
    // This is mostly straightforward as it should just be a ratio between the target and full number of observations.
    // However, this leaves us with some fractional observations in some partitions.
    // To make use of the fractional part, we perform weighted sampling to distribute observations across those partitions.
    //
    // We use the magical Efraimidis and Spirakis algorithm to do a one-pass weighted sampling without replacement based on the fractional parts.
    // See Algorithm A-Res at https://en.wikipedia.org/wiki/Reservoir_sampling
    // and also https://stackoverflow.com/questions/15113650/faster-weighted-sampling-without-replacement
    auto to_sample = sanisizer::create<std::vector<Index_> >(num_partitions);
    {
        std::vector<std::pair<double, Partition_> > probabilities;
        probabilities.reserve(num_partitions);

        const double ratio = static_cast<double>(target) / static_cast<double>(num_obs);
        for (Partition_ p = 0; p < num_partitions; ++p) {
            const double expected = static_cast<double>(partition_count[p]) * ratio;

            if (expected == 0) {
                ;
            } else if (expected < 1 && options.force_non_empty) {
                to_sample[p] = 1;
            } else {
                const double minimum = std::floor(expected);
                to_sample[p] = minimum;
                if (expected > minimum) {
                    probabilities.emplace_back(expected - minimum, p);
                }
            }
        }

        const Index_ already_used = std::accumulate(to_sample.begin(), to_sample.end(), static_cast<Index_>(0));

        if (already_used < target) {
            // The calculation of the weird random variate here is where the magic happens.
            //
            // The probability of 'unif()^(1/a)' being greater than 'unif()^(1/b)' is 'a/(a+b)'.
            // So, if we sort by the random variate and only keep the larger values, we enforce this pairwise difference in selection probability.
            // (In practice, we log-transform these random variates so that higher weights lead to lower values, for numeric precision.
            // Infinities from log-transformed zeros are fine here as they sort as expected.)
            //
            // To convince ourselves, we can go do some painful integrations to compute the probability of orderings that lead to a particular combination being selected.
            // This probability is equal to that of a naive weighted selection algorithm,
            // where the probability of selection of a particular value is equal to the ratio of the weight of the sum of remaining weights in the denominator.
            for (auto& prob : probabilities) {
                prob.first = - std::log(aarand::standard_uniform(rng)) / prob.first;
            }

            const Index_ leftovers = target - already_used;
            std::nth_element(probabilities.begin(), probabilities.begin() + leftovers, probabilities.end());

            for (Index_ i = 0; i < leftovers; ++i) {
                ++to_sample[probabilities[i].second];
            }
        }
    }

    // Alright, actually doing the sampling with replacement now.
    output.clear();
    for (Index_ i = 0; i < num_obs; ++i) {
        const auto part = partition[i];
        auto& needed = to_sample[part];
        if (needed == 0) {
            continue;
        }

        auto& available = partition_count[part];
        if (available <= needed || aarand::standard_uniform(rng) * static_cast<double>(available) <= static_cast<double>(needed)) {
            output.push_back(i);
            needed -= 1;                                
        }

        available -= 1;
    }
}

/**
 * Overload of `compute()` that allocates the output vector.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Partition_ Integer type of the partition assignments.
 *
 * @param num_obs Number of observations, should be non-negative.
 * @param[in] partition Pointer to an array of length `num_obs` containing partition assignments.
 * @param target Desired number of observations in the subset.
 * Note that the actual number of observations returned in `output` may be different.
 * @param options Further options.
 *
 * @return Vector of indices for observations selected in the subset.
 * Indices are guaranteed to be unique and sorted.
 */
template<typename Index_, typename Partition_>
std::vector<Index_> compute(const Index_ num_obs, const Partition_* partition, const Index_ target, const Options& options) {
    std::vector<Index_> output;
    compute(num_obs, partition, target, options, output);
    return output;
}

}

#endif
