#ifndef SCRAN_MARKERS_SCAN_MATRIX_HPP 
#define SCRAN_MARKERS_SCAN_MATRIX_HPP

#include <vector>
#include <cassert>
#include <algorithm>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

#include "cohens_d.hpp"
#include "auc.hpp"
#include "simple_diff.hpp"
#include "utils.hpp"

namespace scran_markers {

namespace internal {

template<typename Value_, typename Group_, typename Stat_, typename Index_>
struct AucScanWorkspace {
    std::vector<AucWorkspace<Value_, Group_, Stat_> > block_workspaces;
    std::vector<std::vector<Index_> > block_num_zeros;
    std::vector<std::vector<Index_> > block_totals;
    std::vector<Stat_> common_buffer;

    bool use_mean;

    // Only when use_mean = true.
    std::optional<std::vector<std::vector<Stat_> > > block_scale;
    std::optional<std::vector<Stat_> > full_weight;

    // Only when use_mean = false.
    std::optional<std::vector<std::vector<std::vector<Stat_> > > > pairwise_buffers;
    std::optional<scran_blocks::SingleQuantileVariable<Stat_, typename std::vector<Stat_>::iterator> > calculator;
};

template<typename Value_, typename Group_, typename Stat_, typename Index_>
AucScanWorkspace<Value_, Group_, Stat_, Index_> initialize_workspace_for_auc(
    const std::size_t ngroups,
    const std::size_t nblocks,
    const std::vector<Index_>& combo_size,
    const BlockAverageInfo<Stat_>& average_info
) {
    AucScanWorkspace<Value_, Group_, Stat_, Index_> work;

    const auto ngroups2 = sanisizer::product<typename std::vector<Stat_>::size_type>(ngroups, ngroups);
    work.common_buffer.resize(ngroups2
#ifdef SCRAN_MARKERS_TEST_INIT
        , SCRAN_MARKERS_TEST_INIT
#endif
    );

    work.block_workspaces.reserve(nblocks);
    work.block_num_zeros.reserve(nblocks);
    work.block_totals.reserve(nblocks);

    sanisizer::cast<typename std::vector<Index_>::size_type>(ngroups);
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        // All workspaces just re-use the same buffer for the AUCs, so make sure to run compute_pairwise_auc() for only one block at a time.
        work.block_workspaces.emplace_back(ngroups, work.common_buffer.data()); 
        work.block_num_zeros.emplace_back(
            ngroups
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        work.block_totals.emplace_back(
            ngroups
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
    }

    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
            work.block_totals[b][g] = combo_size[sanisizer::nd_offset<std::size_t>(g, ngroups, b)]; // remember that the groups are the fastest changing dimension in this array.
        }
    }

    if (average_info.use_mean()) {
        const auto& combo_weights = average_info.combo_weights();
        work.block_scale.emplace();
        work.block_scale->reserve(nblocks);
        work.full_weight.emplace();
        work.full_weight->resize(ngroups2);
        work.use_mean = true;

        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            work.block_scale->emplace_back(ngroups2);
            auto& cur_scale = (*work.block_scale)[b];
            const auto& cur_totals = work.block_totals[b];

            for (I<decltype(ngroups)> g1 = 1; g1 < ngroups; ++g1) {
                const auto w1 = combo_weights[sanisizer::nd_offset<std::size_t>(g1, ngroups, b)];
                const Stat_ denom1 = cur_totals[g1];
                if (denom1 == 0) {
                    continue;
                }

                for (I<decltype(g1)> g2 = 0; g2 < g1; ++g2) {
                    const Stat_ denom2 = cur_totals[g2];
                    if (denom2 == 0) {
                        continue;
                    }

                    const Stat_ block_denom = denom1 * denom2;
                    const Stat_ block_weight = w1 * combo_weights[sanisizer::nd_offset<std::size_t>(g2, ngroups, b)];
                    const Stat_ block_scaling = block_denom / block_weight;

                    const auto pair_offset1 = sanisizer::nd_offset<std::size_t>(g2, ngroups, g1);
                    cur_scale[pair_offset1] = block_scaling;
                    (*work.full_weight)[pair_offset1] += block_weight;

                    const auto pair_offset2 = sanisizer::nd_offset<std::size_t>(g1, ngroups, g2);
                    cur_scale[pair_offset2] = block_scaling;
                    (*work.full_weight)[pair_offset2] += block_weight;
                }
            }
        }

    } else {
        work.pairwise_buffers.emplace();
        work.pairwise_buffers->reserve(ngroups);
        sanisizer::cast<I<decltype(work.pairwise_buffers->front().size())> >(ngroups);
        for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
            work.pairwise_buffers->emplace_back(ngroups);
        }
        work.calculator.emplace(sanisizer::cast<std::size_t>(nblocks), average_info.quantile());
        work.use_mean = false;
    }

    return work;
}

template<typename Value_, typename Group_, typename Stat_, typename Index_, typename Threshold_>
void process_auc_for_rows(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const Threshold_ threshold,
    Stat_* const output
) {
    auto& auc_buffer = work.common_buffer;
    const auto ngroups2 = auc_buffer.size();
    if (work.use_mean) {
        std::fill_n(output, ngroups2, 0);
    } else {
        for (auto& buffers : *work.pairwise_buffers){
            for (auto& individual : buffers) {
                individual.clear();
            }
        }
    }

    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        auto& wrk = work.block_workspaces[b];
        auto& nz = work.block_num_zeros[b];
        const auto& tt = work.block_totals[b];

        const bool normalize = !work.use_mean;
        if (threshold) {
            compute_pairwise_auc(wrk, nz, tt, threshold, normalize);
        } else {
            compute_pairwise_auc(wrk, nz, tt, normalize);
        }

        if (work.use_mean) {
            const auto& block_scale = (*work.block_scale)[b];
            for (I<decltype(ngroups2)> g = 0; g < ngroups2; ++g) {
                const auto scale = block_scale[g];
                if (scale) {
                    output[g] += auc_buffer[g] / scale;
                }
            }

        } else {
            for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
                auto& curbuffers = (*work.pairwise_buffers)[g1];
                for (I<decltype(ngroups)> g2 = 0; g2 < ngroups; ++g2) {
                    if (g1 != g2) {
                        const auto val = auc_buffer[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)];
                        if (!std::isnan(val)) {
                            curbuffers[g2].push_back(val);
                        }
                    }
                }
            }
        }
    }

    for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
        for (I<decltype(ngroups)> g2 = 0; g2 < ngroups; ++g2) {
            const auto offset = sanisizer::nd_offset<std::size_t>(g2, ngroups, g1);
            auto& current = output[offset];

            if (work.use_mean) {
                if (g1 != g2) {
                    const auto full = (*work.full_weight)[offset];
                    if (full) {
                        current /= full;
                    } else {
                        current = std::numeric_limits<Stat_>::quiet_NaN();
                    }
                } else {
                    // We do nothing for g1 == g2, so current defaults to 0 from the initial fill.
                    // This is technically wrong, but no one should be using the self-comparison effect size anyway.
                }

            } else {
                if (g1 != g2) {
                    auto& curbuffer = (*work.pairwise_buffers)[g1][g2];
                    current = (*work.calculator)(curbuffer.size(), curbuffer.begin(), curbuffer.end());
                } else {
                    // Explicitly set this to zero because we didn't do an initial fill in quantile mode.
                    current = 0;
                }
            }
        }
    }
}

template<
    bool single_block_,
    typename Value_,
    typename Index_,
    typename Group_,
    typename Block_,
    typename Combo_,
    typename Stat_,
    class AucResultInitialize_,
    class AucResultProcess_
>
void scan_matrix_by_row_custom_auc(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const std::size_t ngroups,
    const Group_* const group,
    const std::size_t nblocks, // should be equal to 1 if single_block_ = 1.
    const Block_* const block, // ignored if single_block_ = true.
    const std::size_t ncombos, // should be equal to ngroups if single_block_ = true. 
    const Combo_* const combo, // ignored if single_block_ = true.
    const std::vector<Index_>& combo_size,
    const BlockAverageInfo<Stat_>& average_info,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    const bool do_auc,
    AucResultInitialize_ auc_result_init, // generate workspace for processing the final AUC results.
    AucResultProcess_ auc_result_process, // process the pairwise AUC comparisons into the final AUC results.
    const int num_threads
) {
    const Index_ NC = matrix.ncol();
    const auto grouping = [&]{
        if constexpr(single_block_) {
            return group;
        } else {
            return combo;
        }
    }();

    if constexpr(single_block_) {
        assert(ngroups == ncombos);
        assert(nblocks == 1);
    }

    tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        const bool do_means = !combo_means.empty();
        const bool do_detected = !combo_detected.empty();
        const bool do_vars = !combo_vars.empty();

        // Note: do_vars = true implies do_means = true,
        // as there is no situation where we need the variances but not the means.
        if (do_vars) {
            assert(do_means);
        }

        // A vast array of AUC-related bits and pieces.
        std::optional<AucScanWorkspace<Value_, Group_, Stat_, Index_> > auc_work;
        std::optional<I<decltype(auc_result_init(0))> > auc_res_work;
        if (do_auc) {
            auc_work = initialize_workspace_for_auc<Value_, Group_, Stat_, Index_>(ngroups, nblocks, combo_size, average_info);
            auc_res_work = auc_result_init(t);
        }

        const auto divide = [&](Stat_* ptr) -> void {
            for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                ptr[co] /= combo_size[co];
            }
        };

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
            auto ext = tatami::consecutive_extractor<true>(matrix, true, start, length);
            auto tmp_index = sanisizer::create<std::vector<Index_> >(ncombos);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                const auto offset = sanisizer::product_unsafe<std::size_t>(r, ncombos);
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());

                if (do_vars) {
                    const auto var_ptr = combo_vars.data() + offset;
                    const auto mean_ptr = combo_means.data() + offset;
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
                } else if (do_means) {
                    const auto mean_ptr = combo_means.data() + offset;
                    for (Index_ i = 0; i < range.number; ++i) {
                        mean_ptr[grouping[range.index[i]]] += range.value[i];
                    }
                    divide(mean_ptr);
                }

                if (do_detected) {
                    const auto det_ptr = combo_detected.data() + offset;
                    for (Index_ i = 0; i < range.number; ++i) {
                        det_ptr[grouping[range.index[i]]] += (range.value[i] != 0);
                    }
                    divide(det_ptr);
                }

                if (do_auc) {
                    auto nzIt = auc_work->block_num_zeros.begin();
                    for (const auto& t : auc_work->block_totals) {
                        std::copy(t.begin(), t.end(), nzIt->begin());
                        ++nzIt;
                    }
                    for (auto& p : auc_work->block_workspaces) {
                        p.paired.clear();
                    }

                    for (Index_ j = 0; j < range.number; ++j) {
                        if (range.value[j]) {
                            const auto c = range.index[j];
                            const auto b = [&]{
                                if constexpr(single_block_) {
                                    return 0;
                                } else {
                                    return block[c];
                                }
                            }();
                            const auto g = group[c];
                            auc_work->block_workspaces[b].paired.emplace_back(range.value[j], g);
                            --(auc_work->block_num_zeros[b][g]);
                        }
                    }

                    auc_result_process(r, *auc_work, *auc_res_work);
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, true, start, length);

            for (Index_ r = start, end = start + length; r < end; ++r) {
                const auto ptr = ext->fetch(vbuffer.data());
                const auto offset = sanisizer::product_unsafe<std::size_t>(r, ncombos);

                if (do_vars) {
                    const auto mean_ptr = combo_means.data() + offset;
                    const auto var_ptr = combo_vars.data() + offset;
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
                } else if (do_means) {
                    const auto mean_ptr = combo_means.data() + offset;
                    for (Index_ c = 0; c < NC; ++c) {
                        mean_ptr[grouping[c]] += ptr[c];
                    }
                    divide(mean_ptr);
                }

                if (do_detected) {
                    const auto det_ptr = combo_detected.data() + offset;
                    for (Index_ c = 0; c < NC; ++c) {
                        det_ptr[grouping[c]] += (ptr[c] != 0);
                    }
                    divide(det_ptr);
                }

                if (do_auc) {
                    for (auto& z : auc_work->block_num_zeros) {
                        std::fill(z.begin(), z.end(), 0);
                    }
                    for (auto& p : auc_work->block_workspaces) {
                        p.paired.clear();
                    }

                    for (Index_ c = 0; c < NC; ++c) {
                        const auto b = [&]{
                            if constexpr(single_block_) {
                                return 0;
                            } else {
                                return block[c];
                            }
                        }();
                        const auto g = group[c];
                        if (ptr[c]) {
                            auc_work->block_workspaces[b].paired.emplace_back(ptr[c], g);
                        } else {
                            ++(auc_work->block_num_zeros[b][g]);
                        }
                    }

                    auc_result_process(r, *auc_work, *auc_res_work);
                }
            }
        }
    }, matrix.nrow(), num_threads);
}

template<
    bool single_block_,
    typename Value_,
    typename Index_,
    typename Group_,
    typename Block_,
    typename Combo_,
    typename Stat_, 
    typename Threshold_
>
void scan_matrix_by_row_full_auc(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const std::size_t ngroups,
    const Group_* const group,
    const std::size_t nblocks,
    const Block_* const block,
    const std::size_t ncombos,
    const Combo_* const combo,
    const std::vector<Index_>& combo_size,
    const BlockAverageInfo<Stat_>& average_info,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    Stat_* const auc,
    const Threshold_ threshold,
    const int num_threads
) {
    scan_matrix_by_row_custom_auc<single_block_>(
        matrix, 
        ngroups,
        group,
        nblocks,
        block,
        ncombos,
        combo,
        combo_size,
        average_info,
        combo_means,
        combo_vars,
        combo_detected,
        /* do_auc = */ auc != NULL,
        /* auc_result_initialize = */ [&](int) -> bool {
            return false;
        },
        /* auc_result_process = */ [&](const Index_ gene, AucScanWorkspace<Value_, Group_, Stat_, Index_>& auc_work, bool) -> void {
            const auto auc_ptr = auc + sanisizer::product_unsafe<std::size_t>(gene, ngroups, ngroups);
            process_auc_for_rows(auc_work, ngroups, nblocks, threshold, auc_ptr);
        },
        num_threads
    );
}

template<typename Value_, typename Index_, typename Combo_, typename Stat_>
void scan_matrix_by_column(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const std::size_t ncombos,
    const Combo_* const combo,
    const std::vector<Index_>& combo_size,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    const int num_threads
) {
    const Index_ NC = matrix.ncol();
    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        const bool do_means = !combo_means.empty();
        const bool do_detected = !combo_detected.empty();
        const bool do_vars = !combo_vars.empty();

        // Using local buffers to avoid problems with false sharing.
        const auto len = tatami::cast_Index_to_container_size<std::vector<Stat_> >(length);
        const auto allocate_tmp = [&](std::vector<std::vector<Stat_> >& tmp) -> void {
            tmp.reserve(ncombos);
            for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                tmp.emplace_back(len);
            }
        };

        std::vector<std::vector<Stat_> > tmp_means, tmp_vars, tmp_dets;
        if (do_vars) {
            allocate_tmp(tmp_means);
            allocate_tmp(tmp_vars);
        } else if (do_means) {
            allocate_tmp(tmp_means);
        }

        if (do_detected) {
            allocate_tmp(tmp_dets);
        }

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
            auto ext = tatami::consecutive_extractor<true>(matrix, false, static_cast<Index_>(0), NC, start, length);

            std::vector<tatami_stats::variances::RunningSparse<Stat_, Value_, Index_> > runners;
            if (do_vars) {
                runners.reserve(ncombos);
                for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                    runners.emplace_back(length, tmp_means[co].data(), tmp_vars[co].data(), /* skip_nan = */ false, start);
                }
            }

            for (Index_ c = 0; c < NC; ++c) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                const auto co = combo[c];

                if (do_vars) {
                    runners[co].add(range.value, range.index, range.number);
                } else if (do_means) {
                    auto& curmean = tmp_means[co];
                    for (Index_ i = 0; i < range.number; ++i) {
                        curmean[range.index[i] - start] += range.value[i];
                    }
                }

                if (do_detected) {
                    auto& curdet = tmp_dets[co];
                    for (Index_ i = 0; i < range.number; ++i) {
                        curdet[range.index[i] - start] += (range.value[i] != 0);
                    }
                }
            }

            if (do_vars) {
                for (auto& run : runners) {
                    run.finish();
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, false, static_cast<Index_>(0), NC, start, length);

            std::vector<tatami_stats::variances::RunningDense<Stat_, Value_, Index_> > runners;
            if (do_vars) {
                runners.reserve(ncombos);
                for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                    runners.emplace_back(length, tmp_means[co].data(), tmp_vars[co].data(), /* skip_nan = */ false);
                }
            }

            for (Index_ c = 0; c < NC; ++c) {
                const auto ptr = ext->fetch(vbuffer.data());
                const auto co = combo[c];

                if (do_vars) {
                    runners[co].add(ptr);
                } else if (do_means) {
                    auto& curmean = tmp_means[co];
                    for (Index_ r = 0; r < length; ++r) {
                        curmean[r] += ptr[r];
                    }
                }

                if (do_detected) {
                    auto& curdet = tmp_dets[co];
                    for (Index_ r = 0; r < length; ++r) {
                        curdet[r] += (ptr[r] != 0);
                    }
                }
            }

            if (do_vars) {
                for (auto& run : runners) {
                    run.finish();
                }
            }
        }

        // Moving it all into the output buffers at the end.
        for (Index_ r = 0; r < length; ++r) {
            const auto offset = sanisizer::product_unsafe<std::size_t>(start + r, ncombos);

            if (do_vars) {
                const auto mean_ptr = combo_means.data() + offset;
                const auto var_ptr = combo_vars.data() + offset;
                for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                    mean_ptr[co] = tmp_means[co][r];
                    var_ptr[co] = tmp_vars[co][r];
                }
            } else if (do_means) {
                const auto mean_ptr = combo_means.data() + offset;
                for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                    mean_ptr[co] = tmp_means[co][r] / combo_size[co];
                }
            }

            if (do_detected) {
                const auto det_ptr = combo_detected.data() + offset;
                for (I<decltype(ncombos)> co = 0; co < ncombos; ++co) {
                    det_ptr[co] = tmp_dets[co][r] / combo_size[co];
                }
            }
        }
    }, matrix.nrow(), num_threads);
}

}

}

#endif
