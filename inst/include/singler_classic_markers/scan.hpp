#ifndef MEDIAN_MARKERS_SCAN_MATRIX_HPP 
#define MEDIAN_MARKERS_SCAN_MATRIX_HPP 

#include <vector>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"

namespace singler_classic_markers {

template<typename Stat_, typename Value_, typename Index_, typename Combo_, class Setup_, class Function_, class Finalize_>
int scan_matrix(
    const tatami::Matrix<Value_, Index_>& matrix,
    const std::size_t ncombos,
    const Combo_* combo,
    const std::vector<Index_>& combo_sizes,
    Setup_ setup,
    Function_ fun,
    Finalize_ finalize,
    const int num_threads
) {
    const auto NR = matrix.nrow();
    const auto NC = matrix.ncol();

    return tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto customwork = setup();

        auto medians = sanisizer::create<std::vector<Stat_> >(ncombos);
        auto workspace = sanisizer::create<std::vector<std::vector<Value_> > >(ncombos);
        for (std::size_t c = 0; c < ncombos; ++c) {
            workspace[c].reserve(combo_sizes[c]);
        }

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
            auto ext = tatami::consecutive_extractor<true>(matrix, true, start, length);
            auto tmp_index = sanisizer::create<std::vector<Index_> >(ncombos);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (Index_ j = 0; j < range.number; ++j) {
                    workspace[combo[range.index[j]]].push_back(range.value[j]);
                }

                for (std::size_t c = 0; c < ncombos; ++c) {
                    auto& w = workspace[c];
                    medians[c] = tatami_stats::medians::direct<Stat_, Value_, Index_>(w.data(), w.size(), combo_sizes[c], /* skip_nan = */ false);
                    w.clear();
                }

                fun(r, medians, customwork);
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, true, start, length);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                const auto ptr = ext->fetch(vbuffer.data());
                for (Index_ j = 0; j < NC; ++j) {
                    workspace[combo[j]].push_back(ptr[j]);
                }

                for (std::size_t c = 0; c < ncombos; ++c) {
                    auto& w = workspace[c];
                    medians[c] = tatami_stats::medians::direct<Stat_, Value_, Index_>(w.data(), w.size(), /* skip_nan = */ false);
                    w.clear();
                }

                fun(r, medians, customwork);
            }
        }

        finalize(t, customwork);
    }, NR, num_threads);
}

}
#endif
