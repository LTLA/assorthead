#ifndef SINGLEPP_SUBSET_TO_MARKERS_HPP
#define SINGLEPP_SUBSET_TO_MARKERS_HPP

#include "Markers.hpp"
#include "Intersection.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <optional>
#include <cstddef>

namespace singlepp {

// Use this method when the feature spaces are already identical.
template<typename Index_>
std::vector<Index_> subset_to_markers(const Index_ ref_nrow, PairwiseMarkers<Index_>& markers) {
    // Using ref_nrow as a missing-value placeholder, as all indices should be less than it.
    auto available = sanisizer::create<std::vector<Index_> >(ref_nrow, ref_nrow);
    std::vector<Index_> subset;
    for (const auto& mrk : markers) {
        for (const auto& mm : mrk) {
            for (const auto y : mm) {
                auto& av = available[y];
                if (av == ref_nrow) {
                    av = 0; // any value != ref_nrow will do here, and ref_nrow > 0 at this point (otherwise we'd segfault).
                    subset.emplace_back(y);
                }
            }
        }
    }

    // Output is sorted by the row indices, which are the same in both test and reference matrices.
    std::sort(subset.begin(), subset.end());
    const auto nsubset = subset.size();
    for (I<decltype(nsubset)> s = 0; s < nsubset; ++s) {
        available[subset[s]] = s;
    }

    for (auto& mrk : markers) {
        for (auto& mm : mrk) {
            for (auto& y : mm) {
                y = available[y];
            }
        }
    }

    return subset;
}

template<typename Index_>
std::pair<std::vector<Index_>, std::vector<Index_> > subset_to_markers(
    const Index_ test_nrow,
    const Intersection<Index_>& intersection,
    const Index_ ref_nrow,
    PairwiseMarkers<Index_>& markers
) {
    // Again, using ref_nrow and test_nrow as the respective missing-value placeholders.
    auto in_inter = sanisizer::create<std::vector<Index_> >(ref_nrow, test_nrow);
    for (const auto& in : intersection) {
        in_inter[in.second] = in.first;
    }

    auto available = sanisizer::create<std::vector<Index_> >(ref_nrow, ref_nrow);
    std::vector<std::pair<Index_, Index_> > subset;
    for (auto& mrk : markers) {
        for (auto& mm : mrk) {
            const auto num_markers = mm.size();
            I<decltype(num_markers)> used = 0;
            for (I<decltype(num_markers)> m = 0; m < num_markers; ++m) {
                const auto y = mm[m];
                const auto t = in_inter[y];
                if (t != test_nrow) {
                    auto& av = available[y];
                    if (av == ref_nrow) {
                        av = 0; // any value != ref_nrow will do here.
                        subset.emplace_back(t, y);
                    }
                    mm[used] = y;
                    ++used;
                }
            }
            mm.resize(used);
        }
    }

    // Output is sorted by the test indices, to favor more efficient extraction
    // from the test matrix after the training is complete.
    std::sort(subset.begin(), subset.end());
    std::pair<std::vector<Index_>, std::vector<Index_> > output;
    const auto nsubset = subset.size();
    sanisizer::reserve(output.first, nsubset);
    sanisizer::reserve(output.second, nsubset);
    for (I<decltype(nsubset)> s = 0; s < nsubset; ++s) {
        output.first.push_back(subset[s].first);
        output.second.push_back(subset[s].second);
        available[subset[s].second] = s;
    }

    for (auto& mrk : markers) {
        for (auto& mm : mrk) {
            for (auto& y : mm) {
                y = available[y];
            }
        }
    }

    return output;
}

}

#endif
