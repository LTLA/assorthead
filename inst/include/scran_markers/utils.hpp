#ifndef SCRAN_MARKERS_UTILS_HPP
#define SCRAN_MARKERS_UTILS_HPP

#include <type_traits>

#include "sanisizer/sanisizer.hpp"

namespace scran_markers {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

template<typename Group_, typename Index_>
std::vector<Index_> tabulate_groups(const Index_ num_cells, const Group_* group, const std::size_t num_groups) {
    auto output = sanisizer::create<std::vector<Index_> >(num_groups);
    for (Index_ i = 0; i < num_cells; ++i) {
        ++output[group[i]];
    }
    return output;
}

}

#endif
