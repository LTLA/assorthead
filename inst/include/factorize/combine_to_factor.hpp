#ifndef FACTORIZE_COMBINE_TO_FACTOR_HPP
#define FACTORIZE_COMBINE_TO_FACTOR_HPP

#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "create_factor.hpp"
#include "utils.hpp"

/**
 * @file combine_to_factor.hpp
 * @brief Combine categorical variables into a single factor. 
 */

namespace factorize {

/**
 * @tparam Input_ Type of the categorical variables to be combined.
 * Any type may be used here as long as it implements the comparison operators.
 * @tparam Code_ Integer type of the codes of the combined factor.
 * This should be large enough to hold the number of unique combinations.
 *
 * @param n Number of observations (i.e., cells).
 * @param[in] inputs Vector of pointers to arrays of length `n`, each containing a different categorical variable.
 * @param[out] codes Pointer to an array of length `n` in which the codes of the combined factor are to be stored.
 * On output, the code for observation `i` refers to the factor level defined by indexing into the inner vectors of the output vector,
 * i.e., for `j := codes[i]`, the factor level is defined by the combination `(output[0][j], output[1][j], ...)`.
 *
 * @return 
 * Vector of vectors containing the levels of the combined factor. 
 * Each inner vector corresponds to a variables in `inputs`, and all inner vectors have the same length.
 * Corresponding entries of the inner vectors represent a level of the combined factor, in the form of a combination of values from the input variables,
 * i.e., the first level is defined as `(output[0][0], output[1][0], ...)`,
 * the second level is defined as `(output[0][1], output[1][1], ...)`, and so on.
 * Each entry `output[i][j]` is guaranteed to be a value in `inputs[i]`.
 * Combinations are guaranteed to be unique and lexicographically sorted (i.e., by the first variable, then the second, and so on).
 */
template<typename Input_, typename Code_>
std::vector<std::vector<Input_> > combine_to_factor(const std::size_t n, const std::vector<const Input_*>& inputs, Code_* const codes) {
    const auto ninputs = inputs.size();
    auto output = sanisizer::create<std::vector<std::vector<Input_> > >(ninputs);

    // Handling the special cases.
    if (ninputs == 0) {
        std::fill_n(codes, n, 0);
        return output;
    }
    if (ninputs == 1) {
        output[0] = create_factor(n, inputs.front(), codes);
        return output;
    }

    // Creating a hashmap on the combinations of each factor.
    struct Combination {
        Combination(const std::size_t i) : index(i) {}
        std::size_t index;
    };

    auto unique = [&]{ // scoping this in an IIFE to release map memory sooner.
        // Using a map with a custom comparator that uses the index
        // of first occurrence of each factor as the key. Currently using a map
        // to (i) avoid issues with collisions of combined hashes and (ii)
        // avoid having to write more code for sorting a vector of arrays.
        auto cmp = [&](const Combination& left, const Combination& right) -> bool {
            for (auto curf : inputs) {
                if (curf[left.index] < curf[right.index]) {
                    return true;
                } else if (curf[left.index] > curf[right.index]) {
                    return false;
                }
            }
            return false;
        };
        std::map<Combination, Code_, decltype(I(cmp))> mapping(std::move(cmp));

        const auto eq = [&](const Combination& left, const Combination& right) -> bool {
            for (auto curf : inputs) {
                if (curf[left.index] != curf[right.index]) {
                    return false;
                }
            }
            return true;
        };

        for (decltype(I(n)) i = 0; i < n; ++i) {
            Combination current(i);
            const auto mIt = mapping.find(current);
            if (mIt == mapping.end() || !eq(mIt->first, current)) {
                Code_ alt = mapping.size();
                mapping.insert(mIt, std::make_pair(current, alt));
                codes[i] = alt;
            } else {
                codes[i] = mIt->second;
            }
        }

        return std::vector<std::pair<Combination, Code_> >(mapping.begin(), mapping.end());
    }();

    // Remapping to a sorted set.
    const auto nuniq = unique.size();
    for (auto& ofac : output) {
        ofac.reserve(nuniq);
    }
    auto remapping = sanisizer::create<std::vector<Code_> >(nuniq);
    for (decltype(I(nuniq)) u = 0; u < nuniq; ++u) {
        const auto ix = unique[u].first.index;
        for (decltype(I(ninputs)) f = 0; f < ninputs; ++f) {
            output[f].push_back(inputs[f][ix]);
        }
        remapping[unique[u].second] = u;
    }

    // Mapping each cell to its sorted combination.
    for (decltype(I(n)) i = 0; i < n; ++i) {
        codes[i] = remapping[codes[i]];
    }

    return output;
}

/**
 * This function is a variation of `combine_to_factor()` that considers unobserved combinations of variables.
 *
 * @tparam Input_ Factor type.
 * Any type may be used here as long as it is comparable.
 * @tparam Number_ Integer type for the number of unique values in each variable.
 * @tparam Code_ Integer type for the combined factor.
 * This should be large enough to hold the number of unique (possibly unused) combinations.
 *
 * @param n Number of observations (i.e., cells).
 * @param[in] inputs Vector of pairs, each of which corresponds to a categorical variable.
 * The first element of the pair is a pointer to an array of length `n`, containing the values of the variable for each observation.
 * The second element is the total number of unique values for this variable, which may be greater than the largest observed level.
 * @param[out] codes Pointer to an array of length `n` in which the codes of the combined factor are to be stored.
 * On output, each entry determines the corresponding observation's combination of levels by indexing into the inner vectors of the returned object;
 * see the argument of the same name in `combine_to_factor()` for more details.
 *
 * @return 
 * Vector of vectors containing all unique and sorted combinations of the input variables.
 * This has the same structure as the output of `combine_to_factor()`,
 * with the only difference being that unobserved combinations are also reported.
 */
template<typename Input_, typename Number_, typename Code_>
std::vector<std::vector<Input_> > combine_to_factor_unused(const std::size_t n, const std::vector<std::pair<const Input_*, Number_> >& inputs, Code_* const codes) {
    const auto ninputs = inputs.size();
    auto output = sanisizer::create<std::vector<std::vector<Input_> > >(ninputs);

    // Handling the special cases.
    if (ninputs == 0) {
        std::fill_n(codes, n, 0);
        return output;
    }
    if (ninputs == 1) {
        sanisizer::resize(output[0], inputs[0].second);
        std::iota(output[0].begin(), output[0].end(), static_cast<Code_>(0));
        std::copy_n(inputs[0].first, n, codes);
        return output;
    }

    // We iterate from back to front, where the first factor is the slowest changing.
    std::copy_n(inputs[ninputs - 1].first, n, codes); 
    Code_ ncombos = inputs[ninputs - 1].second;
    for (decltype(I(ninputs)) f = ninputs - 1; f > 0; --f) {
        const auto& finfo = inputs[f - 1];
        const auto next_ncombos = sanisizer::product<Code_>(ncombos, finfo.second);
        const auto ff = finfo.first;
        for (decltype(I(n)) i = 0; i < n; ++i) {
            // Product is safe as it is obviously less than 'next_combos' for 'ff[i] < finfo.second'.
            // Addition is also safe as it will be less than 'next_combos', though this is less obvious.
            codes[i] += sanisizer::product_unsafe<Code_>(ncombos, ff[i]);
        }
        ncombos = next_ncombos;
    }

    sanisizer::cast<decltype(I(output[0].size()))>(ncombos); // check that we can actually make the output vectors.
    Code_ outer_repeats = ncombos;
    Code_ inner_repeats = 1;
    for (decltype(I(ninputs)) f = ninputs; f > 0; --f) {
        auto& out = output[f - 1];
        out.reserve(ncombos);

        const auto& finfo = inputs[f - 1];
        const Code_ initial_size = inner_repeats * finfo.second;
        out.resize(initial_size);

        if (inner_repeats == 1) {
            std::iota(out.begin(), out.end(), static_cast<Code_>(0));
        } else {
            auto oIt = out.begin();
            for (Number_ l = 0; l < finfo.second; ++l) {
                std::fill_n(oIt, inner_repeats, l);
                oIt += inner_repeats;
            }
        }
        inner_repeats = initial_size;

        outer_repeats /= finfo.second;
        for (Code_ r = 1; r < outer_repeats; ++r) {
            // Referencing to 'begin()' while inserting is safe, as we reserved the full length at the start.
            // Thus, there shouldn't be any allocations.
            out.insert(out.end(), out.begin(), out.begin() + initial_size);
        }
    }

    return output;
}

}

#endif
