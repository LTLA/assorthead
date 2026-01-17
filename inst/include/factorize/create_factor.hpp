#ifndef FACTORIZE_CLEAN_FACTOR_HPP
#define FACTORIZE_CLEAN_FACTOR_HPP

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file create_factor.hpp
 * @brief Create a factor from a categorical variable.
 */

namespace factorize {

/**
 * Convert a categorical variable into a factor.
 * Factors are defined in a similar manner as in the R programming language,
 * i.e., an array of integer codes, each of which reference into an array of unique levels.
 *
 * @tparam Input_ Type of the categorical variable.
 * Any type may be used here as long as it is hashable and has an equality operator.
 * @tparam Code_ Integer type for the output factor codes.
 *
 * @param n Number of observations. 
 * @param[in] input Pointer to an array of length `n` containing the input categorical variable.
 * @param[out] codes Pointer to an array of length `n` in which the factor codes are to be stored.
 * All values are integers in \f$[0, N)\f$ where \f$N\f$ is the length of the output vector;
 * all integers in this range are guaranteed to be present at least once in `cleaned`.
 *
 * @return A vector of the unique and sorted values of `input`, i.e., the factor levels.
 * For any observation `i`, it is guaranteed that `output[codes[i]] == input[i]`.
 */
template<typename Input_, typename Code_>
std::vector<Input_> create_factor(const std::size_t n, const Input_* const input, Code_* const codes) {
    auto unique = [&]{ // scoping this in an IIFE to release map memory sooner.
        std::unordered_map<Input_, Code_> mapping;
        for (decltype(I(n)) i = 0; i < n; ++i) {
            const auto current = input[i];
            const auto mIt = mapping.find(current);
            if (mIt != mapping.end()) {
                codes[i] = mIt->second;
            } else {
                Code_ alt = mapping.size();
                mapping[current] = alt;
                codes[i] = alt;
            }
        }
        return std::vector<std::pair<Input_, Code_> >(mapping.begin(), mapping.end());
    }();

    // Remapping to a sorted set.
    std::sort(unique.begin(), unique.end());
    const auto nuniq = unique.size();
    auto remapping = sanisizer::create<std::vector<Code_> >(nuniq);
    auto output = sanisizer::create<std::vector<Input_> >(nuniq);
    for (decltype(I(nuniq)) u = 0; u < nuniq; ++u) {
        remapping[unique[u].second] = u;
        output[u] = unique[u].first;
    }

    // Mapping each cell to its sorted factor.
    for (decltype(I(n)) i = 0; i < n; ++i) {
        codes[i] = remapping[codes[i]];
    }

    return output;
}

}

#endif
