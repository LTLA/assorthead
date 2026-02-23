#ifndef SINGLER_CLASSIC_MARKERS_UTILS_HPP
#define SINGLER_CLASSIC_MARKERS_UTILS_HPP

#include <type_traits>
#include <vector>

namespace singler_classic_markers {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

template<bool include_stats_, typename Index_, typename Stat_>
using Markers = std::vector<std::vector<std::vector<typename std::conditional<include_stats_, std::pair<Index_, Stat_>, Index_>::type> > >;

}

#endif
