#ifndef SINGLEPP_SUBSET_REMAPPER_HPP
#define SINGLEPP_SUBSET_REMAPPER_HPP

#include "scaled_ranks.hpp"

#include <vector>
#include <limits>
#include <cstddef>
#include <type_traits>
#include <cassert>

namespace singlepp {

/*
 * This class remaps the indices in the RankVector to the subset of interest.
 * For example, if our subset of features of interest is:
 *
 *    [a, c, g, e]
 *
 * ... the user should call .add(a), .add(c), .add(e), etc. Then, when we
 * receive a rank vector like:
 *
 * [(A, a), (B, b), (C, c), (D, d), (E, e), (F, f), (G, g)],
 *
 * ... the .remap() method filters out the entries that weren't added by
 * .add(), and then remaps the remaining indices to their position on the
 * subset vector, yielding:
 *
 * [(A, 0), (C, 1), (E, 3), (G, 2)]
 *
 * The idea is to adjust the indices so it appears as if we had been working
 * with the subset of features all along, allowing us to call scaled_ranks() to
 * perform the rest of the analysis on the subsets only. This is primarily
 * intended for use on the reference rank vectors during fine-tuning, given that
 * data extracted from the reference/test matrices is already subsetted when
 * returned by SubsetSanitizer::fill_ranks().
 */
template<typename Index_>
class SubsetRemapper {
private:
    Index_ my_capacity;

    // This uses a vector instead of an unordered_map for fast remap()
    // inside the inner loop of the fine-tuning iterations.
    std::vector<Index_> my_mapping;
    std::vector<Index_> my_used;

public:
    SubsetRemapper(const Index_ capacity) : my_capacity(capacity) {
        sanisizer::resize(my_mapping, capacity, capacity);
        sanisizer::reserve(my_used, capacity);
    }

    void add(Index_ i) {
        assert(i < my_capacity);
        if (my_mapping[i] == my_capacity) {
            my_mapping[i] = my_used.size();
            my_used.push_back(i);
        }
    }

    void clear() {
        for (auto u : my_used) {
            my_mapping[u] = my_capacity;
        }
        my_used.clear();
    }

    Index_ size() const {
        return my_used.size();
    }

    Index_ capacity() const {
        return my_capacity;
    }

public:
    template<typename Stat_>
    void remap(typename RankedVector<Stat_, Index_>::const_iterator begin, typename RankedVector<Stat_, Index_>::const_iterator end, RankedVector<Stat_, Index_>& output) const {
        output.clear();
        for (; begin != end; ++begin) {
            assert(sanisizer::is_less_than(begin->second, my_capacity));
            const auto& target = my_mapping[begin->second];
            if (target != my_capacity) {
                output.emplace_back(begin->first, target);
            }
        }
    }

    template<typename Stat_>
    void remap(const RankedVector<Stat_, Index_>& input, RankedVector<Stat_, Index_>& output) const {
        remap(input.begin(), input.end(), output);
    }
};

}

#endif
