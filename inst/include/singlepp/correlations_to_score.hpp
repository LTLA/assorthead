#ifndef SINGLEPP_CORRELATIONS_TO_SCORE_HPP
#define SINGLEPP_CORRELATIONS_TO_SCORE_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

#include "utils.hpp"

#include "sanisizer/sanisizer.hpp"

namespace singlepp {

template<typename Float_>
Float_ l2_to_correlation(const Float_ l2) {
    auto val = static_cast<Float_>(1) - static_cast<Float_>(2) * l2;
    return std::max(static_cast<Float_>(-1), std::min(static_cast<Float_>(1), val));
}

template<typename Index_, typename Float_>
struct PrecomputedQuantileDetails {
    Index_ right_index;
    bool find_left; // if find_left = true, left_index = right_index - 1, otherwise left_index = right_index.
    Float_ right_prop; // No need to store left_prop, as this is just 1 - right_prop.
};

template<typename Index_, typename Float_>
PrecomputedQuantileDetails<Index_, Float_> precompute_quantile_details(const Index_ num, const Float_ quantile) {
    // We should not have any situations where the number of samples is zero,
    // otherwise build_reference_raw() should have failed.
    assert(num > 0);

    // Check that we can safely cast to/from Index_ and Float_.
    const Float_ denom = sanisizer::to_float<Float_>(num - 1); 
    const Float_ fractional_index = denom * (1 - quantile);
    const Float_ left_index = std::floor(fractional_index);
    const Float_ right_index = std::ceil(fractional_index);

    PrecomputedQuantileDetails<Index_, Float_> output;
    output.right_index = right_index; // cast back to Index_ is safe.
    output.find_left = (left_index != right_index);
    output.right_prop = fractional_index - left_index;
    return output;
}

template<typename Float_, typename Index_>
Float_ l2_to_score(std::vector<Float_>& l2, const PrecomputedQuantileDetails<Index_, Float_>& deets) {
    static_assert(std::is_floating_point<Float_>::value);
    assert(sanisizer::is_greater_than(l2.size(), deets.right_index));

    std::nth_element(l2.begin(), l2.begin() + deets.right_index, l2.end());
    const Float_ right_val = l2_to_correlation(l2[deets.right_index]);
    if (!deets.find_left) {
        return right_val;
    }

    // After nth_element(), all elements before 'right_index' are now less than or equal to the value at 'right_index'.
    // So if we want to get the value at 'left_index', we can just find the maximum value rather than sorting again.
    const Float_ left_val = l2_to_correlation(*std::max_element(l2.begin(), l2.begin() + deets.right_index));

    // Here we compute the type 7 quantile, as done by default in R's stats::quantile() function.
    // This basically interpolates between the two observations flanking the quantile.
    // The calculation below is equivalent to 'right_prop * right_val + (1 - right_prop) * left_val'.
    return left_val + (right_val - left_val) * deets.right_prop;
}

}

#endif
