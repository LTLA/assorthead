#ifndef SINGLER_CLASSIC_MARKERS_NUMBER_HPP
#define SINGLER_CLASSIC_MARKERS_NUMBER_HPP

#include <cmath>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

namespace singler_classic_markers {

/**
 * Default number of markers in `choose()` and `choose_blocked()`.
 *
 * The exact expression is defined as \f$500 (\frac{2}{3})^{\log_2{L}}\f$ for \f$L\f$ labels,
 * which steadily decreases the markers per comparison as the number of labels increases.
 * This aims to avoid an excessive number of features when dealing with references with many labels.
 * At \f$L=0\f$, the number of markers is set to zero.
 *
 * @param num_labels Number of labels in the reference(s).
 *
 * @return An appropriate number of markers for each pairwise comparison.
 */
inline std::size_t default_number(std::size_t num_labels) {
    if (num_labels == 0) {
        return 0;
    } else {
        return sanisizer::from_float<std::size_t>(std::round(500.0 * std::pow(2.0/3.0, std::log2(static_cast<double>(num_labels)))));
    }
}

/**
 * @cond
 */
template<typename Index_>
Index_ get_num_keep(const std::size_t ngroups, const std::optional<std::size_t>& request) {
    std::size_t num_keep;
    if (request.has_value()) {
        num_keep = *request;
    } else {
        num_keep = default_number(ngroups);
    }
    return sanisizer::cap<Index_>(num_keep);
}
/**
 * @endcond
 */

}

#endif
