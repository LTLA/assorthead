#ifndef SINGLEPP_SUBSET_REMAPPER_HPP
#define SINGLEPP_SUBSET_REMAPPER_HPP

#include "scaled_ranks.hpp"

#include <vector>
#include <limits>
#include <cstddef>
#include <type_traits>

namespace singlepp {

namespace internal {

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
    // This uses a vector instead of an unordered_map for fast remap()
    // inside the inner loop of the fine-tuning iterations.
    std::vector<std::pair<bool, Index_> > my_mapping;
    std::vector<Index_> my_used;
    Index_ my_counter = 0;

public:
    void add(Index_ i) {
        if (static_cast<typename std::make_unsigned<Index_>::type>(i) >= my_mapping.size()) {
            my_mapping.resize(i + 1);
        }
        if (!my_mapping[i].first) {
            my_mapping[i].first = true;
            my_mapping[i].second = my_counter;
            my_used.push_back(i);
            ++my_counter;
        }
    }

    void clear() {
        my_counter = 0;
        for (auto u : my_used) {
            my_mapping[u].first = false;
        }
        my_used.clear();
    }

    void reserve(typename decltype(my_mapping)::size_type n) {
        my_mapping.reserve(n);
    }

public:
    template<typename Stat_>
    void remap(const RankedVector<Stat_, Index_>& input, RankedVector<Stat_, Index_>& output) const {
        output.clear();

        auto mapsize = my_mapping.size();
        if (static_cast<typename std::make_unsigned<Index_>::type>(std::numeric_limits<Index_>::max()) < mapsize) {
            // Avoid unnecessary check if the size is already greater than the largest possible index.
            // This also avoids the need to cast indices to size_t for comparison to my_mapping.size().
            for (const auto& x : input) {
                const auto& target = my_mapping[x.second];
                if (target.first) {
                    output.emplace_back(x.first, target.second);
                }
            }

        } else {
            // Otherwise, it is safe to cast the size to Index_ outside the
            // loop so that we don't need to cast x.second to size_t inside the loop.
            Index_ maxed = mapsize;
            for (const auto& x : input) {
                if (maxed > x.second) {
                    const auto& target = my_mapping[x.second];
                    if (target.first) {
                        output.emplace_back(x.first, target.second);
                    }
                }
            }
        }
    }
};

}

}

#endif
