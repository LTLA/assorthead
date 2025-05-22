#ifndef SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP
#define SINGLEPP_ANNOTATE_CELLS_INTEGRATED_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "scaled_ranks.hpp"
#include "SubsetRemapper.hpp"
#include "train_integrated.hpp"
#include "find_best_and_delta.hpp"
#include "fill_labels_in_use.hpp"
#include "correlations_to_score.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace singlepp {

namespace internal {

// All data structures in the Workspace can contain anything on input, as they
// are cleared and filled by compute_single_reference_score_integrated(). They
// are only persisted to reduce the number of new allocations when looping
// across cells and references.
template<typename Index_, typename Value_, typename Float_>
struct PerReferenceIntegratedWorkspace {
    SubsetRemapper<Index_> intersect_mapping;
    bool direct_mapping_filled;
    SubsetRemapper<Index_> direct_mapping;

    RankedVector<Value_, Index_> test_ranked;
    RankedVector<Index_, Index_> ref_ranked;

    std::vector<Float_> test_scaled;
    std::vector<Float_> ref_scaled;
    std::vector<Float_> all_correlations;
};

template<typename RefLabel_, typename Label_, typename Index_, typename Value_, typename Float_>
Float_ compute_single_reference_score_integrated(
    RefLabel_ ref_i,
    Label_ best,
    RankedVector<Value_, Index_> test_ranked_full,
    const TrainedIntegrated<Index_>& trained,
    const std::vector<Index_>& miniverse,
    PerReferenceIntegratedWorkspace<Index_, Value_, Float_>& workspace,
    Float_ quantile)
{
    // Further subsetting to the intersection of markers that are
    // actual present in this particular reference.
    const SubsetRemapper<Index_>* mapping;
    if (trained.check_availability[ref_i]) {
        const auto& cur_available = trained.available[ref_i];
        workspace.intersect_mapping.clear();
        workspace.intersect_mapping.reserve(miniverse.size());
        for (auto c : miniverse) {
            if (cur_available.find(c) != cur_available.end()) {
                workspace.intersect_mapping.add(c);
            }
        }
        mapping = &(workspace.intersect_mapping);

    } else {
        // If we don't need to check availability of genes, we only need to
        // populate the direct mapping once per reference, because it will be
        // the same for all references that don't need an availability check.
        if (!workspace.direct_mapping_filled) {
            workspace.direct_mapping.clear();
            workspace.direct_mapping.reserve(miniverse.size());
            for (auto c : miniverse) {
                workspace.direct_mapping.add(c);
            }
            workspace.direct_mapping_filled = true;
        }
        mapping = &(workspace.direct_mapping);
    } 

    mapping->remap(test_ranked_full, workspace.test_ranked);
    workspace.test_scaled.resize(workspace.test_ranked.size());
    scaled_ranks(workspace.test_ranked, workspace.test_scaled.data());

    // Now actually calculating the score for the best group for
    // this cell in this reference. This assumes that
    // 'ranked' already contains sorted pairs where the
    // indices refer to the rows of the original data matrix.
    const auto& best_ranked = trained.ranked[ref_i][best];
    auto nranked = best_ranked.size();
    workspace.all_correlations.clear();
    workspace.ref_scaled.resize(workspace.test_scaled.size());

    for (decltype(nranked) s = 0; s < nranked; ++s) {
        workspace.ref_ranked.clear();
        mapping->remap(best_ranked[s], workspace.ref_ranked);
        scaled_ranks(workspace.ref_ranked, workspace.ref_scaled.data());
        Float_ cor = distance_to_correlation<Float_>(workspace.test_scaled, workspace.ref_scaled);
        workspace.all_correlations.push_back(cor);
    }

    return correlations_to_score(workspace.all_correlations, quantile);
}

template<typename Index_, typename Label_, typename Float_, typename RefLabel_, typename Value_>
std::pair<RefLabel_, Float_> fine_tune_integrated(
    Index_ i,
    RankedVector<Value_, Index_> test_ranked_full,
    std::vector<Float_>& all_scores,
    const TrainedIntegrated<Index_>& trained,
    const std::vector<const Label_*>& assigned,
    std::vector<RefLabel_>& reflabels_in_use, // workspace data structure: input value is ignored, output value should not be used.
    std::unordered_set<Index_>& miniverse_tmp, // workspace data structure: input value is ignored, output value should not be used.
    std::vector<Index_>& miniverse, // workspace data structure: input value is ignored, output value should not be used.
    PerReferenceIntegratedWorkspace<Index_, Value_, Float_>& workspace, // collection of workspace data structures, obviously.
    Float_ quantile,
    Float_ threshold)
{
    auto candidate = fill_labels_in_use(all_scores, threshold, reflabels_in_use);

    // We skip fine-tuning if all or only one labels are selected. If all
    // labels are chosen, there is no contraction of the marker space, and the
    // scores will not change; if only one label is chosen, then that's that.
    while (reflabels_in_use.size() > 1 && reflabels_in_use.size() != all_scores.size()) {
        miniverse_tmp.clear();
        for (auto r : reflabels_in_use) {
            auto curassigned = assigned[r][i];
            const auto& curmarkers = trained.markers[r][curassigned];
            miniverse_tmp.insert(curmarkers.begin(), curmarkers.end());
        }

        miniverse.clear();
        miniverse.insert(miniverse.end(), miniverse_tmp.begin(), miniverse_tmp.end());
        std::sort(miniverse.begin(), miniverse.end()); // sorting for consistency in floating-point summation within scaled_ranks().

        all_scores.clear();
        workspace.direct_mapping_filled = false;
        for (auto r : reflabels_in_use) {
            auto score = compute_single_reference_score_integrated(r, assigned[r][i], test_ranked_full, trained, miniverse, workspace, quantile);
            all_scores.push_back(score);
        }

        candidate = update_labels_in_use(all_scores, threshold, reflabels_in_use); 
    }

    return candidate;
}

template<typename Value_, typename Index_, typename Label_, typename Float_, typename RefLabel_>
void annotate_cells_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedIntegrated<Index_>& trained,
    const std::vector<const Label_*>& assigned,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    RefLabel_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads)
{
    auto NR = test.nrow();
    auto nref = trained.markers.size();
    tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(trained.universe));

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        std::unordered_set<Index_> miniverse_tmp;
        std::vector<Index_> miniverse;

        RankedVector<Value_, Index_> test_ranked_full;
        test_ranked_full.reserve(NR);
        std::vector<Value_> buffer(trained.universe.size());

        PerReferenceIntegratedWorkspace<Index_, Value_, Float_> workspace;
        workspace.test_ranked.reserve(NR);
        workspace.ref_ranked.reserve(NR);

        std::vector<Float_> all_scores;
        std::vector<RefLabel_> reflabels_in_use;

        // We perform an indexed extraction, so all subsequent indices
        // will refer to indices into this subset (i.e., 'universe').
        auto mat_work = tatami::consecutive_extractor<false>(&test, false, start, len, universe_ptr);

        for (Index_ i = start, end = start + len; i < end; ++i) {
            // Extracting only the markers of the best labels for this cell.
            miniverse_tmp.clear();
            for (decltype(nref) r = 0; r < nref; ++r) {
                auto curassigned = assigned[r][i];
                const auto& curmarkers = trained.markers[r][curassigned];
                miniverse_tmp.insert(curmarkers.begin(), curmarkers.end());
            }

            miniverse.clear();
            miniverse.insert(miniverse.end(), miniverse_tmp.begin(), miniverse_tmp.end());
            std::sort(miniverse.begin(), miniverse.end()); // sorting for consistency in floating-point summation within scaled_ranks().

            test_ranked_full.clear();
            auto ptr = mat_work->fetch(buffer.data());
            for (auto u : miniverse) {
                test_ranked_full.emplace_back(ptr[u], u);
            }
            std::sort(test_ranked_full.begin(), test_ranked_full.end());

            // Scanning through each reference and computing the score for the best group.
            all_scores.clear();
            workspace.direct_mapping_filled = false;
            for (decltype(nref) r = 0; r < nref; ++r) {
                auto score = compute_single_reference_score_integrated(
                    r,
                    assigned[r][i],
                    test_ranked_full,
                    trained,
                    miniverse,
                    workspace,
                    quantile
                );
                all_scores.push_back(score);
                if (scores[r]) {
                    scores[r][i] = score;
                }
            }

            std::pair<Label_, Float_> candidate;
            if (!fine_tune) {
                candidate = find_best_and_delta<Label_>(all_scores);
            } else {
                candidate = fine_tune_integrated(
                    i,
                    test_ranked_full,
                    all_scores,
                    trained,
                    assigned,
                    reflabels_in_use,
                    miniverse_tmp,
                    miniverse,
                    workspace,
                    quantile,
                    threshold
                );
            }

            best[i] = candidate.first;
            if (delta) {
                delta[i] = candidate.second;
            }
        }
    }, test.ncol(), num_threads);
}

}

}

#endif
