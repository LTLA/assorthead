#ifndef SINGLEPP_SUBSET_SANITIZER_HPP
#define SINGLEPP_SUBSET_SANITIZER_HPP

#include "utils.hpp"
#include "scaled_ranks.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include <algorithm>
#include <vector>
#include <type_traits>
#include <cassert>

namespace singlepp {

template<bool sparse_, typename Index_>
class SubsetNoop {
private:
    const std::vector<Index_>& my_original_subset;
    typename std::conditional<sparse_, std::vector<Index_>, bool>::type my_remapping;
    typename std::conditional<sparse_, Index_, bool>::type my_remap_start = 0;

public:
    SubsetNoop(const std::vector<Index_>& sub) : my_original_subset(sub) {
        assert(is_sorted_unique(sub.size(), sub.data()));
        if constexpr(sparse_) {
            const auto num_subset = sub.size();
            if (num_subset) {
                my_remap_start = sub[0];
                sanisizer::resize(my_remapping, sub[num_subset - 1] - my_remap_start + 1);
                for (I<decltype(num_subset)> s = 0; s < num_subset; ++s) {
                    my_remapping[sub[s] - my_remap_start] = s;
                }
            }
        }
    }

public:
    const std::vector<Index_>& extraction_subset() const {
        return my_original_subset;
    }

    template<typename Input_, typename Stat_>
    void fill_ranks(const Input_& input, RankedVector<Stat_, Index_>& vec) const {
        // The indices in the output 'vec' refer to positions on the subset
        // vector, as if the input data was already subsetted. 
        vec.clear();

        if constexpr(sparse_) {
            for (I<decltype(input.number)> s = 0; s < input.number; ++s) {
                vec.emplace_back(input.value[s], my_remapping[input.index[s] - my_remap_start]);
            }

        } else {
            const auto num = my_original_subset.size();
            for (I<decltype(num)> s = 0; s < num; ++s) {
                vec.emplace_back(input[s], s);
            }
        }

        // RankedVector's whole schtick is that it's sorted, so we make sure of it here.
        std::sort(vec.begin(), vec.end());
    }
};

/*
 * This class sanitizes any user-provided subsets so that we can provide a
 * sorted subset to the tatami extractor. We then undo the sorting to use the
 * original indices in the rank filler. This entire thing is necessary as the
 * behavior of the subsets isn't something that the user can easily control
 * (e.g., if the reference/test datasets do not use the same feature ordering,
 * in which case the subset is necessarily unsorted).
 *
 * Regardless of whether the input subset is sorted, it should be unique.
 */
template<bool sparse_, typename Index_>
class SubsetSanitizer {
private:
    std::vector<Index_> my_sorted_subset;
    typename std::conditional<sparse_, bool, std::vector<Index_> >::type my_permutation;
    typename std::conditional<sparse_, std::vector<Index_>, bool>::type my_remapping;
    typename std::conditional<sparse_, Index_, bool>::type my_remap_start = 0;

public:
    SubsetSanitizer(const std::vector<Index_>& sub) {
        const auto num_subset = sub.size();
        std::vector<std::pair<Index_, I<decltype(num_subset)> > > store;
        sanisizer::reserve(store, num_subset);
        for (I<decltype(num_subset)> i = 0; i < num_subset; ++i) {
            store.emplace_back(sub[i], i);
        }

        // No need to consider the second element, as all elements of 'sub' should be unique.
        sort_by_first(store);

        sanisizer::reserve(my_sorted_subset, num_subset);
        if constexpr(sparse_) {
            my_remap_start = store.front().first;
            sanisizer::resize(my_remapping, store.back().first - my_remap_start + 1);
        } else {
            sanisizer::reserve(my_permutation, num_subset);
        }

        for (const auto& s : store) {
            // There should not be any duplicates here!
            assert(my_sorted_subset.empty() || my_sorted_subset.back() != s.first);

            my_sorted_subset.push_back(s.first);
            if constexpr(sparse_) {
                my_remapping[s.first - my_remap_start] = s.second;
            } else {
                my_permutation.push_back(s.second);
            }
        }
    }

public:
    const std::vector<Index_>& extraction_subset() const {
        return my_sorted_subset;
    }

    template<typename Input_, typename Stat_>
    void fill_ranks(const Input_& input, RankedVector<Stat_, Index_>& vec) const {
        // The indices in the output 'vec' refer to positions on the subset
        // vector, as if the input data was already subsetted. 
        vec.clear();

        if constexpr(sparse_) {
            for (I<decltype(input.number)> s = 0; s < input.number; ++s) {
                vec.emplace_back(input.value[s], my_remapping[input.index[s] - my_remap_start]);
            }
        } else {
            const auto num = my_permutation.size();
            for (I<decltype(num)> s = 0; s < num; ++s) {
                vec.emplace_back(input[s], my_permutation[s]);
            }
        }

        // RankedVector's whole schtick is that it's sorted, so we make sure of it here.
        std::sort(vec.begin(), vec.end());
    }
};

}

#endif
