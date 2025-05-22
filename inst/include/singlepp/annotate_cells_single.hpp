#ifndef SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP
#define SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "knncolle/knncolle.hpp"

#include "Markers.hpp"
#include "build_indices.hpp"
#include "SubsetSanitizer.hpp"
#include "SubsetRemapper.hpp"
#include "find_best_and_delta.hpp"
#include "scaled_ranks.hpp"
#include "correlations_to_score.hpp"
#include "fill_labels_in_use.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

namespace singlepp {

namespace internal {

template<typename Label_, typename Index_, typename Float_, typename Value_>
class FineTuneSingle {
private:
    std::vector<Label_> my_labels_in_use;

    SubsetRemapper<Index_> my_gene_subset;

    std::vector<Float_> my_scaled_left, my_scaled_right;

    std::vector<Float_> my_all_correlations;

    RankedVector<Value_, Index_> my_input_sub;

    RankedVector<Index_, Index_> my_ref_sub;

public:
    std::pair<Label_, Float_> run(
        const RankedVector<Value_, Index_>& input, 
        const std::vector<PerLabelReference<Index_, Float_> >& ref,
        const Markers<Index_>& markers,
        std::vector<Float_>& scores,
        Float_ quantile,
        Float_ threshold)
    {
        auto candidate = fill_labels_in_use(scores, threshold, my_labels_in_use);

        // Use the input_size as a hint for the number of addressable genes.
        // This should be exact if subset_to_markers() was used on the input,
        // but the rest of the code is safe even if the hint isn't perfect.
        my_gene_subset.reserve(input.size());

        // If there's only one top label, we don't need to do anything else.
        // We also give up if every label is in range, because any subsequent
        // calculations would use all markers and just give the same result.
        while (my_labels_in_use.size() > 1 && my_labels_in_use.size() < scores.size()) {
            my_gene_subset.clear();
            for (auto l : my_labels_in_use) {
                for (auto l2 : my_labels_in_use){ 
                    for (auto c : markers[l][l2]) {
                        my_gene_subset.add(c);
                    }
                }
            }

            my_gene_subset.remap(input, my_input_sub);
            my_scaled_left.resize(my_input_sub.size());
            my_scaled_right.resize(my_input_sub.size());
            scaled_ranks(my_input_sub, my_scaled_left.data());
            scores.clear();

            auto nlabels_used = my_labels_in_use.size();
            for (decltype(nlabels_used) i = 0; i < nlabels_used; ++i) {
                auto curlab = my_labels_in_use[i];

                my_all_correlations.clear();
                const auto& curref = ref[curlab];
                auto NC = curref.index->num_observations();

                for (decltype(NC) c = 0; c < NC; ++c) {
                    // Technically we could be faster if we remembered the
                    // subset from the previous fine-tuning iteration, but this
                    // requires us to (possibly) make a copy of the entire
                    // reference set; we can't afford to do this in each thread.
                    my_gene_subset.remap(curref.ranked[c], my_ref_sub);
                    scaled_ranks(my_ref_sub, my_scaled_right.data());

                    Float_ cor = distance_to_correlation<Float_>(my_scaled_left, my_scaled_right);
                    my_all_correlations.push_back(cor);
                }

                Float_ score = correlations_to_score(my_all_correlations, quantile);
                scores.push_back(score);
            }

            candidate = update_labels_in_use(scores, threshold, my_labels_in_use); 
        }

        return candidate;
    }
};

template<typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_single(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<Index_> subset,
    const std::vector<PerLabelReference<Index_, Float_> >& ref,
    const Markers<Index_>& markers,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads)
{
    // Figuring out how many neighbors to keep and how to compute the quantiles.
    auto num_labels = ref.size();
    std::vector<Index_> search_k(num_labels);
    std::vector<std::pair<Float_, Float_> > coeffs(num_labels);

    for (decltype(num_labels) r = 0; r < num_labels; ++r) {
        Float_ denom = static_cast<Float_>(ref[r].index->num_observations()) - 1;
        Float_  prod = denom * (1 - quantile);
        auto k = std::ceil(prod) + 1;
        search_k[r] = k;

        // `(1 - quantile) - (k - 2) / denom` represents the gap to the smaller quantile,
        // while `(k - 1) / denom - (1 - quantile)` represents the gap from the larger quantile.
        // The size of the gap is used as the weight for the _other_ quantile, i.e., 
        // the closer you are to a quantile, the higher the weight.
        // We convert these into proportions by dividing by their sum, i.e., `1/denom`.
        coeffs[r].first = static_cast<Float_>(k - 1) - prod;
        coeffs[r].second = prod - static_cast<Float_>(k - 2);
    }

    SubsetSanitizer<Index_> subsorted(subset);
    tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &(subsorted.extraction_subset()));

    tatami::parallelize([&](int, Index_ start, Index_ length) {
        auto ext = tatami::consecutive_extractor<false>(&test, false, start, length, subset_ptr);

        auto num_subset = subset.size();
        std::vector<Value_> buffer(num_subset);
        RankedVector<Value_, Index_> vec;
        vec.reserve(num_subset);

        std::vector<std::unique_ptr<knncolle::Searcher<Index_, Float_, Float_> > > searchers;
        searchers.reserve(num_labels);
        for (decltype(num_labels) r = 0; r < num_labels; ++r) {
            searchers.emplace_back(ref[r].index->initialize());
        }
        std::vector<Float_> distances;

        FineTuneSingle<Label_, Index_, Float_, Value_> ft;
        std::vector<Float_> curscores(num_labels);

        for (Index_ c = start, end = start + length; c < end; ++c) {
            auto ptr = ext->fetch(buffer.data());
            subsorted.fill_ranks(ptr, vec);
            scaled_ranks(vec, buffer.data()); // 'buffer' can be re-used for output here, as all data is already extracted to 'vec'.

            curscores.resize(num_labels);
            for (decltype(num_labels) r = 0; r < num_labels; ++r) {
                // No need to use knncolle::cap_k_query(), as our quantile
                // calculations guarantee that 'k' is less than or equal to the
                // number of observations in the reference.
                auto k = search_k[r];
                searchers[r]->search(buffer.data(), k, NULL, &(distances));

                Float_ last = distances[k - 1];
                last = 1 - 2 * last * last;
                if (k == 1) {
                    curscores[r] = last;
                } else {
                    Float_ next = distances[k - 2];
                    next = 1 - 2 * next * next;
                    curscores[r] = coeffs[r].first * next + coeffs[r].second * last;
                }

                if (scores[r]) {
                    scores[r][c] = curscores[r];
                }
            }

            std::pair<Label_, Float_> chosen;
            if (!fine_tune) {
                chosen = find_best_and_delta<Label_>(curscores);
            } else {
                chosen = ft.run(vec, ref, markers, curscores, quantile, threshold);
            }
            best[c] = chosen.first;
            if (delta) {
                delta[c] = chosen.second;
            }
        }
    }, test.ncol(), num_threads);

    return;
}

}

}

#endif
