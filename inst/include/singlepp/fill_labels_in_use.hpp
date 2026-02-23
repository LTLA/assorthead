#ifndef SINGLEPP_FILL_LABELS_IN_USE_HPP
#define SINGLEPP_FILL_LABELS_IN_USE_HPP

#include <vector>
#include <algorithm>
#include <type_traits>

namespace singlepp {

template<typename Stat_, typename Label_>
std::pair<Label_, Stat_> fill_labels_in_use(const std::vector<Stat_>& scores, Stat_ threshold, std::vector<Label_>& in_use) {
    static_assert(std::is_floating_point<Stat_>::value);
    static_assert(std::is_integral<Label_>::value);

    in_use.clear();
    if (scores.size() <= 1) {
        if (!scores.empty()) {
            in_use.push_back(0);
        }
        return std::pair<Label_, Stat_>(0, std::numeric_limits<Stat_>::quiet_NaN());
    } 

    auto it = std::max_element(scores.begin(), scores.end());
    Label_ best_label = it - scores.begin();
    Stat_ max_score = *it;

    constexpr Stat_ DUMMY = -1000; // should be lower than any conceivable correlation.
    Stat_ next_score = DUMMY;
    const Stat_ bound = max_score - threshold;

    Label_  nscores = scores.size();
    for (Label_ i = 0; i < nscores; ++i) {
        auto val = scores[i];
        if (val >= bound) {
            in_use.push_back(i);
        }
        if (i != best_label && next_score < val) {
            next_score = val;
        }
    }

    return std::make_pair(best_label, max_score - next_score); 
}

template<typename Stat_, typename Label_>
std::pair<Label_, Stat_> update_labels_in_use(const std::vector<Stat_>& scores, Stat_ threshold, std::vector<Label_>& in_use) {
    static_assert(std::is_floating_point<Stat_>::value);
    static_assert(std::is_integral<Label_>::value);

    auto it = std::max_element(scores.begin(), scores.end());
    auto nscores = scores.size();
    decltype(nscores) best_index = it - scores.begin();
    Stat_ max_score = *it;

    Label_ best_label = in_use[best_index];
    decltype(in_use.size()) counter = 0;

    constexpr Stat_ DUMMY = -1000;
    Stat_ next_score = DUMMY;
    const Stat_ bound = max_score - threshold;

    for (decltype(nscores) i = 0; i < nscores; ++i) {
        const auto& val = scores[i];
        if (val >= bound) {
            in_use[counter] = in_use[i];
            ++counter;
        }
        if (i != best_index && next_score < val) {
            next_score = val;
        }
    }

    in_use.resize(counter);
    return std::make_pair(best_label, max_score - next_score); 
}

}

#endif
