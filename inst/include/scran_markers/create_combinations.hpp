#ifndef SCRAN_MARKERS_CREATE_COMBINATIONS_HPP
#define SCRAN_MARKERS_CREATE_COMBINATIONS_HPP

#include <vector>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

#include "utils.hpp"

namespace scran_markers {

template<typename Index_>
struct BlockedCombinations {
    BlockedCombinations(const Index_ num_cells, const std::size_t num_combinations) :
        num_combinations(num_combinations),
        combinations(tatami::cast_Index_to_container_size<I<decltype(combinations)> >(num_cells)),
        frequencies(sanisizer::cast<I<decltype(frequencies.size())> >(num_combinations))
    {}

    std::size_t num_combinations;

    std::vector<std::size_t> combinations;

    std::vector<Index_> frequencies;
};

// Wwe combine 'group' and 'block' into a single 'combinations' factor.
// The resulting combinations index into a 2-dimensional array of dimension 'ngroups * nblocks' where the group is the faster-changing dimension.
// This 2D array layout is used for all 'combo_*'-prefixed arrays like 'combo_weights', 'combo_means', etc.
template<typename Index_, typename Group_, typename Block_>
BlockedCombinations<Index_> create_combinations(
    const Index_ num_cells,
    const Group_* const group,
    const std::size_t num_groups,
    const Block_* const block,
    const std::size_t num_blocks
) {
    const auto num_combos = sanisizer::product<std::size_t>(num_groups, num_blocks);
    BlockedCombinations<Index_> output(num_cells, num_combos);
    for (Index_ c = 0; c < num_cells; ++c) {
        const auto combo = sanisizer::nd_offset<std::size_t>(group[c], num_groups, block[c]); // group is the faster changing dimension.
        output.combinations[c] = combo;
        ++output.frequencies[combo];
    }
    return output;
}

}

#endif
