#ifndef SCRAN_MARKERS_SCAN_MATRIX_HPP 
#define SCRAN_MARKERS_SCAN_MATRIX_HPP

#include <vector>
#include <cassert>
#include <algorithm>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "quickstats/quickstats.hpp"
#include "sanisizer/sanisizer.hpp"

#include "cohens_d.hpp"
#include "auc.hpp"
#include "simple_diff.hpp"
#include "utils.hpp"

namespace scran_markers {

namespace internal {

/*****************************************************/

template<typename Value_, typename Group_, typename Stat_, typename Index_>
struct AucScanWorkspace {
    std::vector<AucWorkspace<Value_, Group_, Stat_> > block_workspaces;
    std::vector<std::vector<Index_> > block_num_zeros;
    std::vector<std::vector<Index_> > block_totals;

    // All workspaces just re-use the same buffer for the AUCs,
    // so make sure to run compute_pairwise_auc() and extract the results for one block before moving onto the next block.
    std::vector<Stat_> common_buffer;

    bool use_mean;

    // Only when use_mean = true.
    std::optional<std::vector<std::vector<Stat_> > > block_scale;
    std::optional<std::vector<Stat_> > full_weight;

    // Only when use_mean = false.
    std::optional<std::vector<std::vector<std::vector<Stat_> > > > pairwise_buffers;
    std::optional<quickstats::SingleQuantileVariableNumber<Stat_> > calculator;
};

template<typename Value_, typename Group_, typename Stat_, typename Index_>
void initialize_workspace_for_auc_common(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const std::vector<Index_>& combo_size
) {
    const auto num_groups2 = sanisizer::product<typename std::vector<Stat_>::size_type>(num_groups, num_groups);
    work.common_buffer.resize(num_groups2
#ifdef SCRAN_MARKERS_TEST_INIT
        , SCRAN_MARKERS_TEST_INIT
#endif
    );

    work.block_workspaces.reserve(num_blocks);
    work.block_num_zeros.reserve(num_blocks);
    work.block_totals.reserve(num_blocks);

    sanisizer::cast<typename std::vector<Index_>::size_type>(num_groups);
    for (I<decltype(num_blocks)> b = 0; b < num_blocks; ++b) {
        work.block_workspaces.emplace_back(num_groups, work.common_buffer.data()); 
        work.block_num_zeros.emplace_back(
            num_groups
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        work.block_totals.emplace_back(
            num_groups
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
    }

    for (I<decltype(num_blocks)> b = 0; b < num_blocks; ++b) {
        for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
            // Remember that the groups are the fastest changing dimension in the 'combo_size' array.
            work.block_totals[b][g] = combo_size[sanisizer::nd_offset<std::size_t>(g, num_groups, b)];
        }
    }
}

template<typename Value_, typename Group_, typename Stat_, typename Index_>
void initialize_workspace_for_auc_with_block_mean(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const BlockAverageInfo<Stat_>& average_info
) {
    const auto& combo_weights = average_info.combo_weights();
    work.block_scale.emplace();
    work.block_scale->reserve(num_blocks);
    work.full_weight.emplace();

    const auto num_groups2 = work.common_buffer.size();
    work.full_weight->resize(num_groups2);
    work.use_mean = true;

    for (I<decltype(num_blocks)> b = 0; b < num_blocks; ++b) {
        work.block_scale->emplace_back(num_groups2);
        auto& cur_scale = (*work.block_scale)[b];
        const auto& cur_totals = work.block_totals[b];

        for (I<decltype(num_groups)> g1 = 1; g1 < num_groups; ++g1) {
            const auto w1 = combo_weights[sanisizer::nd_offset<std::size_t>(g1, num_groups, b)];
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
                const Stat_ block_weight = w1 * combo_weights[sanisizer::nd_offset<std::size_t>(g2, num_groups, b)];
                const Stat_ block_scaling = block_denom / block_weight;

                const auto pair_offset1 = sanisizer::nd_offset<std::size_t>(g2, num_groups, g1);
                cur_scale[pair_offset1] = block_scaling;
                (*work.full_weight)[pair_offset1] += block_weight;

                const auto pair_offset2 = sanisizer::nd_offset<std::size_t>(g1, num_groups, g2);
                cur_scale[pair_offset2] = block_scaling;
                (*work.full_weight)[pair_offset2] += block_weight;
            }
        }
    }
}

template<typename Value_, typename Group_, typename Stat_, typename Index_>
void initialize_workspace_for_auc_with_block_quantile(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const BlockAverageInfo<Stat_>& average_info
) {
    work.pairwise_buffers.emplace();
    work.pairwise_buffers->reserve(num_groups);
    sanisizer::cast<I<decltype(work.pairwise_buffers->front().size())> >(num_groups);
    for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
        work.pairwise_buffers->emplace_back(num_groups);
    }
    work.calculator.emplace(sanisizer::cast<std::size_t>(num_blocks), average_info.quantile());
    work.use_mean = false;
}

template<typename Value_, typename Group_, typename Stat_, typename Index_>
AucScanWorkspace<Value_, Group_, Stat_, Index_> initialize_workspace_for_auc(
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const std::vector<Index_>& combo_size,
    const BlockAverageInfo<Stat_>& average_info
) {
    AucScanWorkspace<Value_, Group_, Stat_, Index_> output;
    initialize_workspace_for_auc_common(output, num_groups, num_blocks, combo_size);
    if (average_info.use_mean()) {
        initialize_workspace_for_auc_with_block_mean(output, num_groups, num_blocks, average_info);
    } else {
        initialize_workspace_for_auc_with_block_quantile(output, num_groups, num_blocks, average_info);
    }
    return output;
}

/*****************************************************/

template<typename Value_, typename Group_, typename Stat_, typename Index_, typename Block_, typename Threshold_>
void compute_pairwise_auc(AucScanWorkspace<Value_, Group_, Stat_, Index_>& work, const Block_ b, const Threshold_ threshold, const bool normalize) {
    auto& wrk = work.block_workspaces[b];
    auto& nz = work.block_num_zeros[b];
    const auto& tt = work.block_totals[b];
    if (threshold) {
        compute_pairwise_auc(wrk, nz, tt, threshold, normalize);
    } else {
        compute_pairwise_auc(wrk, nz, tt, normalize);
    }
}

template<typename Value_, typename Group_, typename Stat_, typename Index_, typename Threshold_>
void process_auc_for_rows_block_mean(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const Threshold_ threshold,
    Stat_* const output
) {
    auto& auc_buffer = work.common_buffer;
    const auto num_groups2 = auc_buffer.size();
    std::fill_n(output, num_groups2, 0);

    for (I<decltype(num_blocks)> b = 0; b < num_blocks; ++b) {
        compute_pairwise_auc(work, b, threshold, false);
        const auto& block_scale = (*work.block_scale)[b];
        for (I<decltype(num_groups2)> g = 0; g < num_groups2; ++g) {
            const auto scale = block_scale[g];
            if (scale) {
                output[g] += auc_buffer[g] / scale;
            }
        }
    }

    for (I<decltype(num_groups)> g1 = 0; g1 < num_groups; ++g1) {
        for (I<decltype(num_groups)> g2 = 0; g2 < num_groups; ++g2) {
            const auto offset = sanisizer::nd_offset<std::size_t>(g2, num_groups, g1);
            auto& current = output[offset];
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
        }
    }
}

template<typename Value_, typename Group_, typename Stat_, typename Index_, typename Threshold_>
void process_auc_for_rows_block_quantile(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const Threshold_ threshold,
    Stat_* const output
) {
    auto& auc_buffer = work.common_buffer;
    for (auto& buffers : *work.pairwise_buffers) {
        for (auto& individual : buffers) {
            individual.clear();
        }
    }

    for (I<decltype(num_blocks)> b = 0; b < num_blocks; ++b) {
        compute_pairwise_auc(work, b, threshold, true);
        for (I<decltype(num_groups)> g1 = 0; g1 < num_groups; ++g1) {
            auto& curbuffers = (*work.pairwise_buffers)[g1];
            for (I<decltype(num_groups)> g2 = 0; g2 < num_groups; ++g2) {
                if (g1 != g2) {
                    const auto val = auc_buffer[sanisizer::nd_offset<std::size_t>(g2, num_groups, g1)];
                    if (!std::isnan(val)) {
                        curbuffers[g2].push_back(val);
                    }
                }
            }
        }
    }

    for (I<decltype(num_groups)> g1 = 0; g1 < num_groups; ++g1) {
        for (I<decltype(num_groups)> g2 = 0; g2 < num_groups; ++g2) {
            const auto offset = sanisizer::nd_offset<std::size_t>(g2, num_groups, g1);
            auto& current = output[offset];
            if (g1 != g2) {
                auto& curbuffer = (*work.pairwise_buffers)[g1][g2];
                current = (*work.calculator)(curbuffer.size(), curbuffer.data());
            } else {
                // Explicitly set this to zero because we didn't do an initial fill in quantile mode.
                current = 0;
            }
        }
    }
}

template<typename Value_, typename Group_, typename Stat_, typename Index_, typename Threshold_>
void process_auc_for_rows(
    AucScanWorkspace<Value_, Group_, Stat_, Index_>& work,
    const std::size_t num_groups,
    const std::size_t num_blocks,
    const Threshold_ threshold,
    Stat_* const output
) {
    if (work.use_mean) {
        process_auc_for_rows_block_mean(work, num_groups, num_blocks, threshold, output);
    } else {
        process_auc_for_rows_block_quantile(work, num_groups, num_blocks, threshold, output);
    }
}

/*****************************************************/

template<
    bool single_block_,
    typename Value_,
    typename Index_,
    typename Group_,
    typename Block_,
    typename Combo_,
    typename Stat_,
    class AucResultInitialize_,
    class AucResultProcess_,
    class AucResultFinalize_
>
int scan_matrix_by_row_custom_auc(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Group_* const group,
    const std::size_t num_groups,
    const Block_* const block, // ignored if single_block_ = true.
    const std::size_t num_blocks, // should be equal to 1 if single_block_ = 1.
    const Combo_* const combo, // ignored if single_block_ = true.
    const std::size_t num_combos, // should be equal to num_groups if single_block_ = true. 
    const std::vector<Index_>& combo_size,
    const BlockAverageInfo<Stat_>& average_info,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    const bool do_auc,
    AucResultInitialize_ auc_result_init, // generate workspace for processing the AUC results.
    AucResultProcess_ auc_result_process, // process the pairwise AUC comparisons into AUC results for each gene.
    AucResultFinalize_ auc_result_finalize, // finalize any AUC workspace handlinmg after AUC results are generated for all genes.
    const int num_threads
) {
    const Index_ NC = matrix.ncol();
    const auto active_factor = [&]{
        if constexpr(single_block_) {
            return group;
        } else {
            return combo;
        }
    }();

    if constexpr(single_block_) {
        assert(num_groups == num_combos);
        assert(num_blocks == 1);
    }

    const bool do_means = !combo_means.empty();
    const bool do_detected = !combo_detected.empty();
    const bool do_vars = !combo_vars.empty();

    // Note: do_vars = true implies do_means = true,
    // as there is no situation where we need the variances but not the means.
    if (do_vars) {
        assert(do_means);
    }

    auto num_used = tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);

        // Creating buffers to store the intermediate statistics to avoid false sharing.
        std::optional<std::vector<Stat_> > cur_means, cur_rss;
        if (do_means) {
            cur_means.emplace(tatami::cast_Index_to_container_size<std::vector<Stat_> >(num_combos));
        }
        if (do_vars) {
            cur_rss.emplace(tatami::cast_Index_to_container_size<std::vector<Stat_> >(num_combos));
        }
        std::optional<std::vector<Index_> > cur_detected;
        if (do_detected) {
            cur_detected.emplace(tatami::cast_Index_to_container_size<std::vector<Index_> >(num_combos));
        }

        // A vast array of AUC-related bits and pieces.
        std::optional<AucScanWorkspace<Value_, Group_, Stat_, Index_> > auc_work;
        std::optional<I<decltype(auc_result_init(0))> > auc_res_work;
        if (do_auc) {
            auc_work = initialize_workspace_for_auc<Value_, Group_, Stat_, Index_>(num_groups, num_blocks, combo_size, average_info);
            auc_res_work = auc_result_init(t);
        }

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
            auto ext = tatami::consecutive_extractor<true>(matrix, true, start, length);

            std::optional<std::vector<Index_> > cur_non_zeros;
            if (do_vars) {
                cur_non_zeros.emplace(tatami::cast_Index_to_container_size<std::vector<Index_> >(num_combos));
            }

            for (Index_ r = start, end = start + length; r < end; ++r) {
                const auto offset = sanisizer::product_unsafe<std::size_t>(r, num_combos);
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());

                if (do_means) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        const auto g = active_factor[range.index[i]];
                        (*cur_means)[g] += range.value[i];
                    }
                    for (std::size_t g = 0; g < num_combos; ++g) {
                        if (combo_size[g]) {
                            (*cur_means)[g] /= combo_size[g];
                        } else {
                            (*cur_means)[g] = std::numeric_limits<Stat_>::quiet_NaN();
                        }
                    }

                    if (do_vars) {
                        for (Index_ i = 0; i < range.number; ++i) {
                            const auto g = active_factor[range.index[i]];
                            const auto delta = range.value[i] - (*cur_means)[g];
                            (*cur_rss)[g] += delta * delta;
                            ++(*cur_non_zeros)[g];
                        }

                        const auto var_ptr = combo_vars.data() + offset;
                        for (std::size_t g = 0; g < num_combos; ++g) {
                            if (combo_size[g] >= 2) {
                                const Stat_ my_rss = (*cur_rss)[g] + (*cur_means)[g] * (*cur_means)[g] * (combo_size[g] - (*cur_non_zeros)[g]);
                                var_ptr[g] = my_rss / (combo_size[g] - 1);
                            } else {
                                var_ptr[g] = std::numeric_limits<Stat_>::quiet_NaN();
                            }
                        }

                        std::fill(cur_rss->begin(), cur_rss->end(), 0);
                        std::fill(cur_non_zeros->begin(), cur_non_zeros->end(), 0);
                    }

                    std::copy(cur_means->begin(), cur_means->end(), combo_means.data() + offset);
                    std::fill(cur_means->begin(), cur_means->end(), 0);
                }

                if (do_detected) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        (*cur_detected)[active_factor[range.index[i]]] += (range.value[i] != 0);
                    }
                    const auto det_ptr = combo_detected.data() + offset;
                    for (std::size_t g = 0; g < num_combos; ++g) {
                        if (combo_size[g]) {
                            det_ptr[g] = static_cast<Stat_>((*cur_detected)[g]) / combo_size[g];
                        } else {
                            det_ptr[g] = std::numeric_limits<Stat_>::quiet_NaN();
                        }
                    }
                    std::fill(cur_detected->begin(), cur_detected->end(), 0);
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
                const auto offset = sanisizer::product_unsafe<std::size_t>(r, num_combos);

                if (do_means) {
                    for (Index_ c = 0; c < NC ; ++c) {
                        (*cur_means)[active_factor[c]] += ptr[c];
                    }
                    for (std::size_t g = 0; g < num_combos; ++g) {
                        if (combo_size[g]) {
                            (*cur_means)[g] /= combo_size[g];
                        } else {
                            (*cur_means)[g] = std::numeric_limits<Stat_>::quiet_NaN();
                        }
                    }

                    if (do_vars) {
                        for (Index_ c = 0; c < NC ; ++c) {
                            const auto g = active_factor[c];
                            const auto delta = ptr[c] - (*cur_means)[g];
                            (*cur_rss)[g] += delta * delta;
                        }

                        const auto var_ptr = combo_vars.data() + offset;
                        for (std::size_t g = 0; g < num_combos; ++g) {
                            if (combo_size[g] >= 2) {
                                var_ptr[g] = (*cur_rss)[g] / (combo_size[g] - 1);
                            } else {
                                var_ptr[g] = std::numeric_limits<Stat_>::quiet_NaN();
                            }
                        }

                        std::fill(cur_rss->begin(), cur_rss->end(), 0);
                    }

                    std::copy(cur_means->begin(), cur_means->end(), combo_means.data() + offset);
                    std::fill(cur_means->begin(), cur_means->end(), 0);
                }

                if (do_detected) {
                    for (Index_ c = 0; c < NC; ++c) {
                        (*cur_detected)[active_factor[c]] += (ptr[c] != 0);
                    }
                    const auto det_ptr = combo_detected.data() + offset;
                    for (std::size_t g = 0; g < num_combos; ++g) {
                        if (combo_size[g]) {
                            det_ptr[g] = static_cast<Stat_>((*cur_detected)[g]) / combo_size[g];
                        } else {
                            det_ptr[g] = std::numeric_limits<Stat_>::quiet_NaN();
                        }
                    }
                    std::fill(cur_detected->begin(), cur_detected->end(), 0);
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

        if (do_auc) {
            auc_result_finalize(t, *auc_res_work);
        }
    }, matrix.nrow(), num_threads);

    return num_used;
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
    const Group_* const group,
    const std::size_t num_groups,
    const Block_* const block,
    const std::size_t num_blocks,
    const Combo_* const combo,
    const std::size_t num_combos,
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
        group,
        num_groups,
        block,
        num_blocks,
        combo,
        num_combos,
        combo_size,
        average_info,
        combo_means,
        combo_vars,
        combo_detected,
        /* do_auc = */ auc != NULL,
        /* auc_result_initialize = */ [](int) -> bool {
            return false;
        },
        /* auc_result_process = */ [&](const Index_ gene, AucScanWorkspace<Value_, Group_, Stat_, Index_>& auc_work, bool) -> void {
            const auto auc_ptr = auc + sanisizer::product_unsafe<std::size_t>(gene, num_groups, num_groups);
            process_auc_for_rows(auc_work, num_groups, num_blocks, threshold, auc_ptr);
        },
        /* auc_result_finalize = */ [](int, bool) -> void {
        },
        num_threads
    );
}

/*****************************************************/

template<typename Value_, typename Index_, typename Combo_, typename Stat_>
void scan_matrix_by_column(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Combo_* const combo,
    const std::size_t num_combos,
    const std::vector<Index_>& combo_size,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    const int num_threads
) {
    const bool do_means = !combo_means.empty();
    const bool do_detected = !combo_detected.empty();
    const bool do_vars = !combo_vars.empty();

    // Note: do_vars = true implies do_means = true,
    // as there is no situation where we need the variances but not the means.
    if (do_vars) {
        assert(do_means);
    }

    std::optional<std::vector<std::optional<std::vector<Stat_> > > > collected_means;
    if (do_means) {
        collected_means.emplace(sanisizer::cast<I<decltype(collected_means->size())> >(num_threads));
    }

    std::optional<std::vector<std::optional<std::vector<Stat_> > > > collected_rss;
    std::optional<std::vector<std::optional<std::vector<Index_> > > > collected_counts;
    if (do_vars) {
        collected_rss.emplace(sanisizer::cast<I<decltype(collected_rss->size())> >(num_threads));
        collected_counts.emplace(sanisizer::cast<I<decltype(collected_counts->size())> >(num_threads));
    }

    // Using Stat_ to hold the number of detected cells in each group, instead of Index_.
    // This avoids another allocation to hold the proportions before transposition.
    std::optional<std::vector<std::optional<std::vector<Stat_> > > > collected_detected;
    if (do_detected) {
        collected_detected.emplace(sanisizer::cast<I<decltype(collected_detected->size())> >(num_threads));
    }

    const Index_ NR = matrix.nrow();
    const auto full_size = sanisizer::product_unsafe<std::size_t>(NR, num_combos);
    const auto nused = tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);

        std::optional<std::vector<Stat_> > tmp_means;
        if (do_means) {
            tmp_means.emplace(sanisizer::cast<I<decltype(tmp_means->size())> >(full_size));
        }

        std::optional<std::vector<Stat_> > tmp_rss;
        std::optional<std::vector<Index_> > tmp_counts;
        if (do_vars) {
            tmp_rss.emplace(sanisizer::cast<I<decltype(tmp_rss->size())> >(full_size));
            tmp_counts.emplace(sanisizer::cast<I<decltype(tmp_counts->size())> >(num_combos));
        }

        std::optional<std::vector<Stat_> > tmp_detected;
        if (do_detected) {
            tmp_detected.emplace(sanisizer::cast<I<decltype(tmp_detected->size())> >(full_size));
        }

        if (matrix.is_sparse()) {
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);
            auto ext = tatami::consecutive_extractor<true>(matrix, false, start, length);

            std::optional<std::vector<Index_> > cur_non_zeros;
            if (do_vars) {
                cur_non_zeros.emplace(sanisizer::cast<I<decltype(cur_non_zeros->size())> >(full_size));
            }

            for (Index_ c = 0; c < length; ++c) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                const auto co = combo[start + c];
                const auto offset = sanisizer::product_unsafe<std::size_t>(co, NR);

                if (do_vars) {
                    ++(*tmp_counts)[co];
                    for (Index_ i = 0; i < range.number; ++i) {
                        const auto r = range.index[i];
                        quickstats::update_rss((*tmp_means)[offset + r], (*tmp_rss)[offset + r], range.value[i], ++(*cur_non_zeros)[offset + r]);
                    }
                } else if (do_means) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        (*tmp_means)[offset + range.index[i]] += range.value[i];
                    }
                }

                if (do_detected) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        (*tmp_detected)[offset + range.index[i]] += (range.value[i] != 0);
                    }
                }
            }

            if (do_vars) {
                for (std::size_t g = 0; g < num_combos; ++g) {
                    const auto cursize = (*tmp_counts)[g];
                    if (cursize == 0) {
                        continue;
                    }
                    const auto offset = sanisizer::product_unsafe<std::size_t>(g, NR);
                    for (Index_ r = 0; r < NR; ++r) {
                        quickstats::update_rss_with_zeros_unsafe((*tmp_means)[offset + r], (*tmp_rss)[offset + r], cursize - (*cur_non_zeros)[offset + r], cursize);
                    }
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, false, start, length);

            for (Index_ c = 0; c < length; ++c) {
                const auto ptr = ext->fetch(vbuffer.data());
                const auto co = combo[start + c];
                const auto offset = sanisizer::product_unsafe<std::size_t>(co, NR);

                if (do_vars) {
                    ++(*tmp_counts)[co];
                    for (Index_ r = 0; r < NR; ++r) {
                        quickstats::update_rss((*tmp_means)[offset + r], (*tmp_rss)[offset + r], ptr[r], (*tmp_counts)[co]);
                    }
                } else if (do_means) {
                    for (Index_ r = 0; r < NR; ++r) {
                        (*tmp_means)[offset + r] += ptr[r];
                    }
                }

                if (do_detected) {
                    for (Index_ r = 0; r < NR; ++r) {
                        (*tmp_detected)[offset + r] += (ptr[r] != 0);
                    }
                }
            }
        }

        if (do_vars) {
            (*collected_rss)[t] = std::move(tmp_rss);
            (*collected_counts)[t] = std::move(tmp_counts);
        }
        if (do_means) {
            (*collected_means)[t] = std::move(tmp_means);
        }
        if (do_detected) {
            (*collected_detected)[t] = std::move(tmp_detected);
        }
    }, matrix.ncol(), num_threads);

    // Reducing the statistics from all threads.
    if (do_vars) {
        auto& first_mean = *(collected_means->front());
        auto& first_rss = *(collected_rss->front());

        if (nused > 1) {
            // We need to allocate a separate vector for the global mean for each gene in each group,
            // as we need to compare the global and per-thread means to recenter the RSS.
            auto global_means = sanisizer::create<std::vector<Stat_> >(NR);
            for (std::size_t g = 0; g < num_combos; ++g) {
                const auto offset = sanisizer::product_unsafe<std::size_t>(g, NR);
                if (combo_size[g] == 0) {
                    std::fill_n(first_mean.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                    std::fill_n(first_rss.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                    continue;
                }

                bool mean_initialized = false;
                for (int u = 0; u < nused; ++u) {
                    const auto cur_count = (*((*collected_counts)[u]))[g];
                    if (cur_count == 0) {
                        continue;
                    }
                    const auto& src = *((*collected_means)[u]);
                    const Stat_ ratio = static_cast<Stat_>(cur_count) / static_cast<Stat_>(combo_size[g]);
                    if (!mean_initialized) { // Don't rely on u == 0 as this group might be empty in the first thread.
                        for (Index_ r = 0; r < NR; ++r) {
                            global_means[r] = ratio * src[offset + r];
                        }
                        mean_initialized = true;
                    } else {
                        for (Index_ r = 0; r < NR; ++r) {
                            global_means[r] += ratio * src[offset + r];
                        }
                    }
                }
                assert(mean_initialized);

                if (combo_size[g] == 1) {
                    std::fill_n(first_rss.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                    std::copy(global_means.begin(), global_means.end(), first_mean.begin() + offset);
                    continue;
                } 

                bool var_initialized = false;
                for (int u = 0; u < nused; ++u) {
                    const auto cur_count = (*((*collected_counts)[u]))[g];
                    if (cur_count == 0) {
                        continue;
                    }
                    const auto& src_means = *((*collected_means)[u]);
                    if (u == 0) { // Special case so that we can optimize for the source being the same as the destination.
                        for (Index_ r = 0; r < NR; ++r) {
                            first_rss[offset + r] = quickstats::recenter_rss_unsafe(cur_count, first_rss[offset + r], src_means[offset + r], global_means[r]);
                        }
                        var_initialized = true;
                    } else {
                        const auto& src_rss = *((*collected_rss)[u]);
                        if (!var_initialized) { // Don't rely on u == 0 as this group might be empty in the first thread.
                            for (Index_ r = 0; r < NR; ++r) {
                                first_rss[offset + r] = quickstats::recenter_rss_unsafe(cur_count, src_rss[offset + r], src_means[offset + r], global_means[r]);
                            }
                            var_initialized = true;
                        } else {
                            for (Index_ r = 0; r < NR; ++r) {
                                first_rss[offset + r] += quickstats::recenter_rss_unsafe(cur_count, src_rss[offset + r], src_means[offset + r], global_means[r]);
                            }
                        }
                    }
                }
                assert(var_initialized);

                for (Index_ r = 0; r < NR; ++r) {
                    // We know that combo_size[g] > 1 at this point, so no need to add protection.
                    first_rss[offset + r] /= combo_size[g] - 1;
                }
                std::copy(global_means.begin(), global_means.end(), first_mean.begin() + offset);
            }

        } else {
            for (std::size_t g = 0; g < num_combos; ++g) {
                const auto offset = sanisizer::product_unsafe<std::size_t>(g, NR);
                if (combo_size[g] == 0) {
                    std::fill_n(first_mean.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                    std::fill_n(first_rss.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                    continue;
                } else if (combo_size[g] == 1) {
                    std::fill_n(first_rss.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                    continue;
                } else {
                    for (Index_ r = 0; r < NR; ++r) {
                        first_rss[offset + r] /= combo_size[g] - 1;
                    }
                }
            }
        }

        tatami::transpose(first_mean.data(), num_combos, NR, combo_means.data());
        tatami::transpose(first_rss.data(), num_combos, NR, combo_vars.data());

    } else if (do_means) {
        auto& first = *(collected_means->front());
        for (int u = 1; u < nused; ++u) {
            const auto& src = *((*collected_means)[u]);
            for (std::size_t f = 0; f < full_size; ++f) {
                first[f] += src[f];
            }
        }
        for (std::size_t g = 0; g < num_combos; ++g) {
            const auto offset = sanisizer::product_unsafe<std::size_t>(g, NR);
            if (combo_size[g] == 0) {
                std::fill_n(first.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            for (Index_ r = 0; r < NR; ++r) {
                first[offset + r] /= combo_size[g];
            }
        }
        tatami::transpose(first.data(), num_combos, NR, combo_means.data());
    }

    if (do_detected) {
        auto& first = *(collected_detected->front());
        for (int u = 1; u < nused; ++u) {
            const auto& src = *((*collected_detected)[u]);
            for (std::size_t f = 0; f < full_size; ++f) {
                first[f] += src[f];
            }
        }
        for (std::size_t g = 0; g < num_combos; ++g) {
            const auto offset = sanisizer::product_unsafe<std::size_t>(g, NR);
            if (combo_size[g] == 0) {
                std::fill_n(first.begin() + offset, NR, std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            for (Index_ r = 0; r < NR; ++r) {
                first[offset + r] /= combo_size[g];
            }
        }
        tatami::transpose(first.data(), num_combos, NR, combo_detected.data());
    }
}

}

}

#endif
