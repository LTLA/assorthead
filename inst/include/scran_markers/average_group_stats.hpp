#ifndef SCRAN_MARKERS_AVERAGE_GROUP_STATS_HPP
#define SCRAN_MARKERS_AVERAGE_GROUP_STATS_HPP

#include <vector>
#include <limits>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

namespace scran_markers {

namespace internal {

template<typename Weight_>
std::vector<Weight_> compute_total_weight_per_group(std::size_t ngroups, std::size_t nblocks, const Weight_* combo_weights) {
    auto output = sanisizer::create<std::vector<Weight_> >(ngroups);
    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        for (decltype(ngroups) g = 0; g < ngroups; ++g) {
            output[g] += combo_weights[sanisizer::nd_offset<std::size_t>(g, ngroups, b)];
        }
    }
    return output;
}

template<typename Gene_, typename Stat_, typename Weight_>
void average_group_stats(
    Gene_ gene,
    std::size_t ngroups,
    std::size_t nblocks,
    const Stat_* tmp_means,
    const Stat_* tmp_detected,
    const Weight_* combo_weights,
    const Weight_* total_weights,
    const std::vector<Stat_*>& means,
    const std::vector<Stat_*>& detected)
{
    for (decltype(ngroups) g = 0; g < ngroups; ++g) {
        auto& gmean = means[g][gene];
        auto& gdet = detected[g][gene];

        auto total_weight = total_weights[g];
        if (total_weight == 0) {
            gdet = std::numeric_limits<Stat_>::quiet_NaN();
            gmean = std::numeric_limits<Stat_>::quiet_NaN();
            continue;
        }

        gmean = 0;
        gdet = 0;

        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            // Remember, blocks are the slower changing dimension, so we need to jump by 'ngroups'.
            auto offset = sanisizer::nd_offset<std::size_t>(g, ngroups, b);
            const auto& curweight = combo_weights[offset];
            if (curweight) {
                gmean += curweight * tmp_means[offset];
                gdet += curweight * tmp_detected[offset];
            } 
        }

        gmean /= total_weight;
        gdet /= total_weight;
    }
}

template<typename Gene_, typename Stat_>
void fill_average_results(
    Gene_ ngenes,
    std::size_t ngroups,
    std::vector<std::vector<Stat_> >& mean_res, 
    std::vector<std::vector<Stat_> >& detected_res, 
    std::vector<Stat_*>& mean_ptrs,
    std::vector<Stat_*>& detected_ptrs)
{
    mean_res.reserve(ngroups);
    auto mrngenes = sanisizer::cast<decltype(mean_res.front().size())>(ngenes);
    detected_res.reserve(ngroups);
    auto drngenes = sanisizer::cast<decltype(detected_res.front().size())>(ngenes);
    mean_ptrs.reserve(ngroups);
    detected_ptrs.reserve(ngroups);

    for (decltype(ngroups) g = 0; g < ngroups; ++g) {
        mean_res.emplace_back(mrngenes
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        detected_res.emplace_back(drngenes
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        mean_ptrs.emplace_back(mean_res.back().data());
        detected_ptrs.emplace_back(detected_res.back().data());
    }
}

}

}

#endif
