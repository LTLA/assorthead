#ifndef SINGLEPP_CORRELATIONS_TO_SCORE_HPP
#define SINGLEPP_CORRELATIONS_TO_SCORE_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

namespace singlepp {

namespace internal {

template<typename Float_>
Float_ correlations_to_score(std::vector<Float_>& correlations, Float_ quantile) {
    static_assert(std::is_floating_point<Float_>::value);

    auto ncells = correlations.size();
    if (ncells == 0) {
        return std::numeric_limits<Float_>::quiet_NaN();
    }

    if (quantile == 1 || ncells == 1) {
        return *std::max_element(correlations.begin(), correlations.end());
    }
    
    const Float_ denom = ncells - 1; 
    const Float_ prod = denom * quantile;
    const decltype(ncells) left = std::floor(prod);
    const decltype(ncells) right = std::ceil(prod);

    std::nth_element(correlations.begin(), correlations.begin() + right, correlations.end());
    const Float_ rightval = correlations[right];
    if (right == left) {
        return rightval;
    }

    // After nth_element(), all elements before 'right' are now less than or
    // equal to the value at 'right'. So if we want to get 'left', we can just
    // find the maximum value rather than sorting again.
    const Float_ leftval = *std::max_element(correlations.begin(), correlations.begin() + right);

    // `quantile - left / denom` represents the gap to the smaller quantile,
    // while `right / denom - quantile` represents the gap from the larger quantile.
    // The size of the gap is used as the weight for the _other_ quantile, i.e., 
    // the closer you are to a quantile, the higher the weight.
    // We convert these into proportions by dividing by their sum, i.e., `1/denom`.
    const Float_ leftweight = right - prod;
    const Float_ rightweight = prod - left;

    return rightval * rightweight + leftval * leftweight;
}

template<typename Float_, typename Stat_>
Float_ distance_to_correlation(const std::vector<Stat_>& p1, const std::vector<Stat_>& p2) {
    static_assert(std::is_floating_point<Float_>::value);
    auto n = p1.size();

    Float_ d2 = 0;
    for (decltype(n) i = 0; i < n; ++i) {
        auto tmp = static_cast<Float_>(p1[i]) - static_cast<Float_>(p2[i]);
        d2 += tmp * tmp;
    }
    return 1 - 2 * d2;
}

}

}

#endif
