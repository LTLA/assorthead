#ifndef SINGLEPP_FIND_BEST_AND_DELTA_HPP
#define SINGLEPP_FIND_BEST_AND_DELTA_HPP

#include <algorithm>
#include <limits>
#include <vector>

namespace singlepp {

template<typename Label_, typename Float_>
std::pair<Label_, Float_> find_best_and_delta(const std::vector<Float_>& scores) {
    if (scores.size() <= 1) {
        return std::pair<Label_, Float_>(0, std::numeric_limits<Float_>::quiet_NaN());
    }

    auto top = std::max_element(scores.begin(), scores.end());
    decltype(scores.size()) best_idx = top - scores.begin();

    Float_ topscore = scores[best_idx];
    Float_ second;
    if (best_idx == 0) {
        second = *std::max_element(scores.begin() + 1, scores.end());
    } else if (best_idx + 1 == scores.size()) {
        second = *std::max_element(scores.begin(), scores.begin() + best_idx);
    } else {
        second = std::max(
            *std::max_element(scores.begin(), scores.begin() + best_idx),
            *std::max_element(scores.begin() + best_idx + 1, scores.end())
        );
    }
    
    return std::pair<Label_, Float_>(best_idx, topscore - second);
}

}

#endif
