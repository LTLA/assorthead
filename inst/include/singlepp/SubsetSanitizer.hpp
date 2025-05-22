#ifndef SINGLEPP_SUBSET_SANITIZER_HPP
#define SINGLEPP_SUBSET_SANITIZER_HPP

#include "scaled_ranks.hpp"

#include <algorithm>
#include <vector>
#include <cstddef>

namespace singlepp {

namespace internal {

/*
 * This class sanitizes any user-provided subsets so that we can provide a
 * sorted and unique subset to the tatami extractor. We then undo the sorting
 * to use the original indices in the rank filler. This entire thing is
 * necessary as the behavior of the subsets isn't something that the user can
 * easily control (e.g., if the reference/test datasets do not use the same
 * feature ordering, in which case the subset is necessarily unsorted).
 */
template<typename Index_>
class SubsetSanitizer {
private:
    bool my_use_sorted_subset = false;
    const std::vector<Index_>& my_original_subset;
    std::vector<Index_> my_sorted_subset;

    typedef typename std::vector<Index_>::size_type Size; // index type of the input subset vector in the constructor.
    std::vector<Size> my_original_indices;

public:
    SubsetSanitizer(const std::vector<Index_>& sub) : my_original_subset(sub) {
        auto num_subset = sub.size();
        for (decltype(num_subset) i = 1; i < num_subset; ++i) {
            if (sub[i] <= sub[i-1]) {
                my_use_sorted_subset = true;
                break;
            }
        }

        if (my_use_sorted_subset) {
            std::vector<std::pair<Index_, Size> > store;
            store.reserve(num_subset);
            for (decltype(num_subset) i = 0; i < num_subset; ++i) {
                store.emplace_back(sub[i], i);
            }

            std::sort(store.begin(), store.end());
            my_sorted_subset.reserve(num_subset);
            my_original_indices.resize(num_subset);
            for (const auto& s : store) {
                if (my_sorted_subset.empty() || my_sorted_subset.back() != s.first) {
                    my_sorted_subset.push_back(s.first);
                }
                my_original_indices[s.second] = my_sorted_subset.size() - 1;
            }
        }
    }

public:
    const std::vector<Index_>& extraction_subset() const {
        if (my_use_sorted_subset) {
            return my_sorted_subset;
        } else {
            return my_original_subset;
        }
    }

    template<typename Stat_>
    void fill_ranks(const Stat_* ptr, RankedVector<Stat_, Index_>& vec) const {
        // The indices in the output 'vec' refer to positions on the subset
        // vector, as if the input data was already subsetted. 
        vec.clear();
        if (my_use_sorted_subset) {
            auto num = my_original_indices.size();
            for (decltype(num) s = 0; s < num; ++s) {
                vec.emplace_back(ptr[my_original_indices[s]], s);
            }
        } else {
            auto num = my_original_subset.size();
            for (decltype(num) s = 0; s < num; ++s) {
                vec.emplace_back(ptr[s], s);
            }
        }
        std::sort(vec.begin(), vec.end());
    }
};

}

}

#endif
