#ifndef TATAMI_MULT_SPARSE_MATRIX_UTILS_HPP
#define TATAMI_MULT_SPARSE_MATRIX_UTILS_HPP

#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../utils.hpp"

namespace tatami_mult {

template<typename Value_, typename Index_>
void populate_sparse_buffers(
    const bool row,
    const Index_ primary,
    const Index_ secondary,
    const tatami::Matrix<Value_, Index_>& matrix,
    std::vector<std::vector<Value_> >& all_vbuffers,
    std::vector<std::vector<Index_> >& all_ibuffers,
    std::vector<tatami::SparseRange<Value_, Index_> >& all_ranges,
    int num_threads
) {
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto tmp_v = tatami::create_container_of_Index_size<std::vector<Value_> >(secondary);
        auto tmp_i = tatami::create_container_of_Index_size<std::vector<Index_> >(secondary);
        auto ext = tatami::consecutive_extractor<true>(matrix, row, start, length);
        for (Index_ i = start, end = start + length; i < end; ++i) {
            tmp_v.resize(secondary); // creation was type-safe, so should resize.
            tmp_i.resize(secondary);

            auto range = ext->fetch(tmp_v.data(), tmp_i.data());
            if (range.value == tmp_v.data()) {
                all_vbuffers[i].swap(tmp_v);
                range.value = all_vbuffers[i].data();
            }
            if (range.index == tmp_i.data()) {
                all_ibuffers[i].swap(tmp_i);
                range.index = all_ibuffers[i].data();
            }
            all_ranges[i] = range;
        }
    }, primary, num_threads);
}

template<typename Value_, typename Index_, class Zero_>
std::optional<std::vector<Index_> > filter_non_empty_sparse(
    const std::vector<tatami::SparseRange<Value_, Index_> >& all_ranges,
    Zero_ zerofun
) {
    bool all_non_empty = true;
    for (const auto& range : all_ranges) {
        if (range.number == 0) {
            all_non_empty = false;
            break;
        }
    }

    std::optional<std::vector<Index_> > output;
    if (!all_non_empty) {
        const auto num = all_ranges.size();
        output.emplace();
        output->reserve(num);
        for (I<decltype(num)> i = 0; i < num; ++i) {
            if (all_ranges[i].number > 0) {
                output->push_back(i);
            } else {
                zerofun(i);
            }
        }
    }

    return output;
}

template<typename Index_, class Zero_>
std::optional<std::vector<Index_> > filter_non_empty_sparse(
    const std::vector<std::vector<Index_> >& all_index,
    Zero_ zerofun
) {
    bool all_non_empty = true;
    for (const auto& indices : all_index) {
        if (indices.empty() == 0) {
            all_non_empty = false;
            break;
        }
    }

    std::optional<std::vector<Index_> > output;
    if (!all_non_empty) {
        const auto num = all_index.size();
        output.emplace();
        output->reserve(num);
        for (I<decltype(num)> i = 0; i < num; ++i) {
            if (all_index[i].size() > 0) {
                output->push_back(i);
            } else {
                zerofun(i);
            }
        }
    }

    return output;
}

}

#endif
