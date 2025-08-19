#ifndef SCRAN_MARKERS_SCAN_MATRIX_HPP 
#define SCRAN_MARKERS_SCAN_MATRIX_HPP

#include "cohens_d.hpp"
#include "auc.hpp"
#include "simple_diff.hpp"

#include <vector>
#include <cassert>
#include <algorithm>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

namespace scran_markers {

namespace internal {

template<typename Value_, typename Group_, typename Index_, typename Stat_>
struct AucScanWorkspace {
    std::vector<AucWorkspace<Value_, Group_, Stat_> > block_workspaces;
    std::vector<std::vector<Index_> > block_num_zeros;
    std::vector<std::vector<Index_> > block_totals;
    std::vector<std::vector<Stat_> > block_scale;
    std::vector<Stat_> common_buffer;
    std::vector<Stat_> full_weight;
};

template<typename Value_, typename Group_, typename Index_, typename Stat_, typename Weight_>
void initialize_auc_workspace(
    AucScanWorkspace<Value_, Group_, Index_, Stat_>& work,
    std::size_t ngroups,
    std::size_t nblocks,
    const std::vector<Index_>& combo_size,
    const std::vector<Weight_>& combo_weight) 
{
    const auto ngroups2 = sanisizer::product<typename std::vector<Stat_>::size_type>(ngroups, ngroups);
    work.common_buffer.resize(ngroups2
#ifdef SCRAN_MARKERS_TEST_INIT
        , SCRAN_MARKERS_TEST_INIT
#endif
    );

    work.block_workspaces.reserve(nblocks);
    work.block_num_zeros.reserve(nblocks);
    auto nzngroups = sanisizer::cast<decltype(work.block_num_zeros.front().size())>(ngroups);
    work.block_totals.reserve(nblocks);
    auto btngroups = sanisizer::cast<decltype(work.block_totals.front().size())>(ngroups);

    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        // All workspaces just re-use the same buffer for the AUCs, so make sure to run compute_pairwise_auc() for only one block at a time.
        work.block_workspaces.emplace_back(ngroups, work.common_buffer.data()); 
        work.block_num_zeros.emplace_back(nzngroups
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        work.block_totals.emplace_back(btngroups
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
    }

    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        for (decltype(ngroups) g = 0; g < ngroups; ++g) {
            work.block_totals[b][g] = combo_size[sanisizer::nd_offset<std::size_t>(g, ngroups, b)]; // remember that the groups are the fastest changing dimension in this array.
        }
    }

    work.block_scale.reserve(nblocks);
    work.full_weight.resize(ngroups2);
    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        work.block_scale.emplace_back(ngroups2);
        auto& cur_scale = work.block_scale[b];
        auto& cur_totals = work.block_totals[b];

        for (decltype(ngroups) g1 = 1; g1 < ngroups; ++g1) {
            auto w1 = combo_weight[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)];
            Stat_ denom1 = cur_totals[g1];

            for (decltype(g1) g2 = 0; g2 < g1; ++g2) {
                Stat_ block_denom = denom1 * static_cast<Stat_>(cur_totals[g2]);
                if (block_denom == 0) {
                    continue;
                }

                Stat_ block_weight = w1 * combo_weight[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
                Stat_ block_scaling = block_denom / block_weight;

                auto pair_offset1 = sanisizer::nd_offset<std::size_t>(g2, ngroups, g1);
                cur_scale[pair_offset1] = block_scaling;
                work.full_weight[pair_offset1] += block_weight;

                auto pair_offset2 = sanisizer::nd_offset<std::size_t>(g1, ngroups, g2);
                cur_scale[pair_offset2] = block_scaling;
                work.full_weight[pair_offset2] += block_weight;
            }
        }
    }
}

template<typename Value_, typename Group_, typename Index_, typename Stat_, typename Threshold_>
void process_auc_for_rows(
    AucScanWorkspace<Value_, Group_, Index_, Stat_>& work,
    std::size_t ngroups,
    std::size_t nblocks,
    Threshold_ threshold,
    Stat_* output) 
{
    auto& auc_buffer = work.common_buffer;
    auto ngroups2 = auc_buffer.size();
    std::fill_n(output, ngroups2, 0);

    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        auto& wrk = work.block_workspaces[b];
        auto& nz = work.block_num_zeros[b];
        const auto& tt = work.block_totals[b];

        if (threshold) {
            compute_pairwise_auc(wrk, nz, tt, threshold, false);
        } else {
            compute_pairwise_auc(wrk, nz, tt, false);
        }

        // Adding to the blocks.
        const auto& block_scale = work.block_scale[b];
        for (decltype(ngroups2) g = 0; g < ngroups2; ++g) {
            if (block_scale[g]) {
                output[g] += auc_buffer[g] / block_scale[g];
            }
        }
    }

    for (decltype(ngroups) g1 = 0; g1 < ngroups; ++g1) {
        for (decltype(ngroups) g2 = 0; g2 < ngroups; ++g2) {
            auto offset = sanisizer::nd_offset<std::size_t>(g2, ngroups, g1);
            auto& current = output[offset];
            if (work.full_weight[offset]) {
                current /= work.full_weight[offset];
            } else if (g1 != g2) {
                current = std::numeric_limits<Stat_>::quiet_NaN();
            } // g1 == g2 gets a current = 0, which is technically wrong, but no one should be using the self-comparison effect size anyway.
        }
    }
}

template<bool single_block_, typename Value_, typename Index_, typename Group_, typename Block_, typename Stat_, typename Weight_, typename Threshold_>
void scan_matrix_by_row(
    const tatami::Matrix<Value_, Index_>& matrix, 
    std::size_t ngroups,
    const Group_* group,
    std::size_t nblocks, // should be equal to 1 if single_block_ = 1.
    const Block_* block, // ignored if single_block_ = true.
    std::size_t ncombos, // should be equal to ngroups if single_block_ = true. 
    const std::size_t* combinations, // ignored if single_block_ = true.
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    Stat_* auc,
    const std::vector<Index_>& combo_size,
    const std::vector<Weight_>& combo_weights,
    Threshold_ threshold,
    int num_threads)
{
    Index_ NC = matrix.ncol();
    const auto* grouping = [&]{
        if constexpr(single_block_) {
            return group;
        } else {
            return combinations;
        }
    }();

    if constexpr(single_block_) {
        assert(ngroups == ncombos);
        assert(nblocks == 1);
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        // A vast array of AUC-related bits and pieces.
        AucScanWorkspace<Value_, Group_, Index_, Stat_> auc_work;
        if (auc) {
            initialize_auc_workspace(auc_work, ngroups, nblocks, combo_size, combo_weights);
        }

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
            auto ext = tatami::consecutive_extractor<true>(matrix, true, start, length);
            auto tmp_index = sanisizer::create<std::vector<Index_> >(ncombos);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto offset = sanisizer::product_unsafe<std::size_t>(r, ncombos);
                auto mean_ptr = combo_means.data() + offset;
                auto var_ptr = combo_vars.data() + offset;
                auto det_ptr = combo_detected.data() + offset;

                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                tatami_stats::grouped_variances::direct(
                    range.value,
                    range.index,
                    range.number,
                    grouping,
                    ncombos,
                    combo_size.data(),
                    mean_ptr,
                    var_ptr,
                    tmp_index.data(),
                    /* skip_nan = */ false,
                    /* invalid_count = */ static_cast<Index_*>(NULL)
                );

                for (Index_ i = 0; i < range.number; ++i) {
                    det_ptr[grouping[range.index[i]]] += (range.value[i] != 0);
                }
                for (decltype(ncombos) co = 0; co < ncombos; ++co) {
                    det_ptr[co] /= combo_size[co];
                }

                if (auc) {
                    auto nzIt = auc_work.block_num_zeros.begin();
                    for (const auto& t : auc_work.block_totals) {
                        std::copy(t.begin(), t.end(), nzIt->begin());
                        ++nzIt;
                    }
                    for (auto& p : auc_work.block_workspaces) {
                        p.paired.clear();
                    }

                    for (Index_ j = 0; j < range.number; ++j) {
                        if (range.value[j]) {
                            auto c = range.index[j];
                            auto b = [&]{
                                if constexpr(single_block_) {
                                    return 0;
                                } else {
                                    return block[c];
                                }
                            }();
                            auto g = group[c];
                            auc_work.block_workspaces[b].paired.emplace_back(range.value[j], g);
                            --(auc_work.block_num_zeros[b][g]);
                        }
                    }

                    auto auc_ptr = auc + sanisizer::product_unsafe<std::size_t>(r, ngroups, ngroups);
                    process_auc_for_rows(auc_work, ngroups, nblocks, threshold, auc_ptr);
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, true, start, length);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                auto offset = sanisizer::product_unsafe<std::size_t>(r, ncombos);
                auto mean_ptr = combo_means.data() + offset;
                auto var_ptr = combo_vars.data() + offset;
                auto det_ptr = combo_detected.data() + offset;

                auto ptr = ext->fetch(vbuffer.data());
                tatami_stats::grouped_variances::direct(
                    ptr,
                    NC,
                    grouping,
                    ncombos,
                    combo_size.data(),
                    mean_ptr,
                    var_ptr,
                    /* skip_nan = */ false,
                    /* invalid_count = */ static_cast<Index_*>(NULL)
                );
                mean_ptr += ncombos;
                var_ptr += ncombos;

                for (Index_ c = 0; c < NC; ++c) {
                    det_ptr[grouping[c]] += (ptr[c] != 0);
                }
                for (decltype(ncombos) co = 0; co < ncombos; ++co) {
                    det_ptr[co] /= combo_size[co];
                }
                det_ptr += ncombos;

                if (auc) {
                    for (auto& z : auc_work.block_num_zeros) {
                        std::fill(z.begin(), z.end(), 0);
                    }
                    for (auto& p : auc_work.block_workspaces) {
                        p.paired.clear();
                    }

                    for (Index_ c = 0; c < NC; ++c) {
                        auto b = [&]{
                            if constexpr(single_block_) {
                                return 0;
                            } else {
                                return block[c];
                            }
                        }();
                        auto g = group[c];
                        if (ptr[c]) {
                            auc_work.block_workspaces[b].paired.emplace_back(ptr[c], g);
                        } else {
                            ++(auc_work.block_num_zeros[b][g]);
                        }
                    }

                    auto auc_ptr = auc + sanisizer::product_unsafe<std::size_t>(r, ngroups, ngroups);
                    process_auc_for_rows(auc_work, ngroups, nblocks, threshold, auc_ptr);
                }
            }
        }
    }, matrix.nrow(), num_threads);
}

template<typename Value_, typename Index_, typename Combination_, typename Stat_>
void scan_matrix_by_column(
    const tatami::Matrix<Value_, Index_>& matrix, 
    std::size_t ncombos,
    const Combination_* combinations,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    const std::vector<Index_>& combo_size,
    int num_threads)
{
    Index_ NC = matrix.ncol();
    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);

        // Using local buffers to avoid problems with false sharing.
        std::vector<std::vector<Stat_> > tmp_means, tmp_vars, tmp_dets;
        {
            tmp_means.reserve(ncombos);
            tmp_vars.reserve(ncombos);
            tmp_dets.reserve(ncombos);
            auto len = tatami::cast_Index_to_container_size<std::vector<Stat_> >(length);
            for (decltype(ncombos) co = 0; co < ncombos; ++co) {
                tmp_means.emplace_back(len);
                tmp_vars.emplace_back(len);
                tmp_dets.emplace_back(len);
            }
        }

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
            auto ext = tatami::consecutive_extractor<true>(matrix, false, static_cast<Index_>(0), NC, start, length);
            std::vector<tatami_stats::variances::RunningSparse<Stat_, Value_, Index_> > runners;
            runners.reserve(ncombos);
            for (decltype(ncombos) co = 0; co < ncombos; ++co) {
                runners.emplace_back(length, tmp_means[co].data(), tmp_vars[co].data(), /* skip_nan = */ false, start);
            }

            for (Index_ c = 0; c < NC; ++c) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                auto co = combinations[c];
                runners[co].add(range.value, range.index, range.number);

                auto& curdet = tmp_dets[co];
                for (Index_ i = 0; i < range.number; ++i) {
                    curdet[range.index[i] - start] += (range.value[i] != 0);
                }
            }

            for (auto& run : runners) {
                run.finish();
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, false, static_cast<Index_>(0), NC, start, length);
            std::vector<tatami_stats::variances::RunningDense<Stat_, Value_, Index_> > runners;
            runners.reserve(ncombos);
            for (decltype(ncombos) co = 0; co < ncombos; ++co) {
                runners.emplace_back(length, tmp_means[co].data(), tmp_vars[co].data(), /* skip_nan = */ false);
            }

            for (Index_ c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(vbuffer.data());
                auto co = combinations[c];
                runners[co].add(ptr);

                auto& curdet = tmp_dets[co];
                for (Index_ r = 0; r < length; ++r) {
                    curdet[r] += (ptr[r] != 0);
                }
            }

            for (auto& run : runners) {
                run.finish();
            }
        }

        // Moving it all into the output buffers at the end.
        for (Index_ r = 0; r < length; ++r) {
            auto offset = sanisizer::product_unsafe<std::size_t>(start + r, ncombos);
            auto mean_ptr = combo_means.data() + offset;
            auto var_ptr = combo_vars.data() + offset;
            auto det_ptr = combo_detected.data() + offset;

            for (decltype(ncombos) co = 0; co < ncombos; ++co) {
                mean_ptr[co] = tmp_means[co][r];
                var_ptr[co] = tmp_vars[co][r];
                det_ptr[co] = tmp_dets[co][r] / combo_size[co];
            }
        }
    }, matrix.nrow(), num_threads);
}

}

}

#endif
