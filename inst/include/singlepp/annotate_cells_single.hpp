#ifndef SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP
#define SINGLEPP_ANNOTATE_CELLS_SINGLE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "train_single.hpp"
#include "SubsetSanitizer.hpp"
#include "SubsetRemapper.hpp"
#include "find_best_and_delta.hpp"
#include "scaled_ranks.hpp"
#include "l2.hpp"
#include "correlations_to_score.hpp"
#include "fill_labels_in_use.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <type_traits>
#include <optional>

namespace singlepp {

template<bool ref_sparse_, typename Index_, typename Float_>
const auto& get_per_label_references(const BuiltReference<Index_, Float_>& built) {
    if constexpr(ref_sparse_) {
        assert(built.sparse.has_value());
        assert(!built.dense.has_value());
        return *(built.sparse);
    } else {
        assert(!built.sparse.has_value());
        assert(built.dense.has_value());
        return *(built.dense);
    }
}

template<bool query_sparse_, bool ref_sparse_, typename Label_, typename Index_, typename Float_, typename Value_>
class FineTuneSingle {
private:
    std::vector<Label_> my_labels_in_use;

    SubsetRemapper<Index_> my_gene_subset;
    RankedVector<Value_, Index_> my_subset_query;
    RankedVector<Index_, Index_> my_subset_ref;
    typename std::conditional<ref_sparse_, RankedVector<Index_, Index_>, bool>::type my_subset_ref_alt;

    std::vector<Float_> my_scaled_query;
    typename std::conditional<ref_sparse_, std::vector<std::pair<Index_, Float_> >, std::vector<Float_> >::type my_scaled_ref;
    typename std::conditional<query_sparse_, std::vector<std::pair<Index_, Float_> >, bool>::type my_scaled_query_sparse_buffer;

    std::vector<Float_> my_all_l2;

public:
    typedef typename std::conditional<ref_sparse_, SparsePerLabel<Index_, Float_>, DensePerLabel<Index_, Float_> >::type PerLabel;

    FineTuneSingle(const Index_ full_num_markers, const std::vector<PerLabel>& ref) : my_gene_subset(full_num_markers) {
        sanisizer::reserve(my_labels_in_use, ref.size());

        sanisizer::reserve(my_subset_query, full_num_markers); 
        sanisizer::reserve(my_subset_ref, full_num_markers); 
        if constexpr(ref_sparse_) {
            sanisizer::reserve(my_subset_ref_alt, full_num_markers); 
        }

        sanisizer::resize(my_scaled_query, full_num_markers);
        sanisizer::reserve(my_scaled_ref, full_num_markers);
        if constexpr(query_sparse_) {
            sanisizer::reserve(my_scaled_query_sparse_buffer, full_num_markers);
        }

        I<decltype(get_num_samples(ref.front()))> max_labels = 0;
        for (const auto& curref : ref) {
            max_labels = std::max(max_labels, get_num_samples(curref));
        }
        sanisizer::reserve(my_all_l2, max_labels);
    }

    // For testing only.
    FineTuneSingle(const TrainedSingle<Index_, Float_>& trained) : 
        FineTuneSingle(trained.subset().size(), get_per_label_references<ref_sparse_>(trained.built()))
    {}

public:
    std::pair<Label_, Float_> run(
        const RankedVector<Value_, Index_>& input, 
        const TrainedSingle<Index_, Float_>& trained,
        const std::vector<PrecomputedQuantileDetails<Index_, Float_> >& quantile_details,
        const Float_ threshold,
        std::vector<Float_>& scores
    ) {
        auto candidate = fill_labels_in_use(scores, threshold, my_labels_in_use);
        const auto& ref = get_per_label_references<ref_sparse_>(trained.built());
        const auto& markers = trained.markers();

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
            my_gene_subset.remap(input, my_subset_query);
            const auto current_num_markers = my_gene_subset.size();

            bool query_has_nonzero = false;
            my_scaled_query.resize(current_num_markers);
            if constexpr(query_sparse_) {
                const auto substart = my_subset_query.begin();
                const auto subend = my_subset_query.end();
                auto zero_ranges = find_zero_ranges<Value_, Index_>(substart, subend);
                query_has_nonzero = scaled_ranks_sparse<Index_, Value_, Float_>(
                    current_num_markers,
                    substart,
                    zero_ranges.first,
                    zero_ranges.second,
                    subend,
                    my_scaled_query_sparse_buffer,
                    my_scaled_query.data()
                );
            } else {
                query_has_nonzero = scaled_ranks_dense(
                    current_num_markers,
                    my_subset_query,
                    my_scaled_query.data()
                );
            }

            if constexpr(!ref_sparse_) {
                my_scaled_ref.resize(current_num_markers);
            }

            scores.clear();
            auto nlabels_used = my_labels_in_use.size();
            for (I<decltype(nlabels_used)> i = 0; i < nlabels_used; ++i) {
                auto curlab = my_labels_in_use[i];

                my_all_l2.clear();
                const auto& curref = ref[curlab];
                const auto NC = get_num_samples(curref);

                for (I<decltype(NC)> c = 0; c < NC; ++c) {
                    // Technically we could be faster if we remembered the
                    // subset from the previous fine-tuning iteration, but this
                    // requires us to (possibly) make a copy of the entire
                    // reference set; we can't afford to do this in each thread.

                    Float_ l2 = 0;
                    if constexpr(ref_sparse_) {
                        const auto nStart = curref.negative_ranked.begin();
                        my_gene_subset.remap(nStart + curref.negative_indptrs[c], nStart + curref.negative_indptrs[c + 1], my_subset_ref);
                        const auto pStart = curref.positive_ranked.begin();
                        my_gene_subset.remap(pStart + curref.positive_indptrs[c], pStart + curref.positive_indptrs[c + 1], my_subset_ref_alt);

                        l2 = scaled_ranks_sparse_l2(
                            current_num_markers,
                            my_scaled_query.data(),
                            query_has_nonzero,
                            my_subset_ref,
                            my_subset_ref_alt,
                            my_scaled_ref
                        );

                    } else {
                        const auto full_num_markers = my_gene_subset.capacity();
                        const auto refstart = curref.all_ranked.begin() + sanisizer::product_unsafe<std::size_t>(full_num_markers, c);
                        const auto refend = refstart + full_num_markers;
                        my_gene_subset.remap(refstart, refend, my_subset_ref);

                        l2 = scaled_ranks_dense_l2(
                            current_num_markers,
                            my_scaled_query.data(),
                            my_subset_ref,
                            my_scaled_ref.data()
                        );
                    }

                    my_all_l2.push_back(l2);
                }

                const Float_ score = l2_to_score(my_all_l2, quantile_details[curlab]);
                scores.push_back(score);
            }

            candidate = update_labels_in_use(scores, threshold, my_labels_in_use); 
        }

        return candidate;
    }
};

template<bool query_sparse_, bool ref_sparse_, typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_single_raw(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingle<Index_, Float_>& trained,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto& subset = trained.subset();
    const Index_ num_markers = subset.size(); // cast is safe as 'subset' is a unique subset of the rows of the reference matrix.
    SubsetNoop<query_sparse_, Index_> subsorted(subset);

    const auto& built = trained.built();
    const auto& ref = get_per_label_references<ref_sparse_>(built);
    const auto num_labels = ref.size();

    auto quantile_details = std::vector<PrecomputedQuantileDetails<Index_, Float_> >(num_labels);
    Index_ max_num_samples = 0;
    for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
        const auto num_samples = get_num_samples(ref[r]);
        quantile_details[r] = precompute_quantile_details(num_samples, quantile);
        max_num_samples = std::max(max_num_samples, num_samples);
    }

    tatami::parallelize([&](int, Index_ start, Index_ length) {
        tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, &subset);
        auto ext = tatami::consecutive_extractor<query_sparse_>(&test, false, start, length, std::move(subset_ptr));

        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_markers);
        auto ibuffer = [&](){
            if constexpr(query_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_markers);
            } else {
                return false;
            }
        }();

        RankedVector<Value_, Index_> vec;
        vec.reserve(num_markers);
        FindClosestNeighborsWorkspace<Index_, Float_> find_work(max_num_samples);
        
        auto query_scaled = sanisizer::create<std::vector<Float_> >(num_markers);
        typename std::conditional<query_sparse_, std::vector<std::pair<Index_, Float_> >, bool>::type query_sparse_buffer;
        if constexpr(query_sparse_) {
            sanisizer::reserve(query_sparse_buffer, num_markers);
        }

        std::optional<FineTuneSingle<query_sparse_, ref_sparse_, Label_, Index_, Float_, Value_> > ft;
        if (fine_tune) {
            ft.emplace(num_markers, ref);
        }
        std::vector<Float_> curscores(num_labels);

        for (Index_ c = start, end = start + length; c < end; ++c) {
            bool query_has_nonzero = false;
            if constexpr(query_sparse_) {
                const auto info = ext->fetch(vbuffer.data(), ibuffer.data());
                subsorted.fill_ranks(info, vec);
                const auto vStart = vec.begin(), vEnd = vec.end();
                const auto zero_ranges = find_zero_ranges<Value_, Index_>(vStart, vEnd);
                query_has_nonzero = scaled_ranks_sparse<Index_, Value_, Float_>(
                    num_markers,
                    zero_ranges.first - vStart,
                    vStart,
                    zero_ranges.first,
                    vEnd - zero_ranges.second,
                    zero_ranges.second,
                    vEnd,
                    query_sparse_buffer,
                    /* zero processing */ [&](const Float_ x) -> void {
                        std::fill(query_scaled.begin(), query_scaled.end(), x);
                    },
                    /* non-zero processing */ [&](std::pair<Index_, Float_>& pair, const Float_ x) -> void {
                        query_scaled[pair.first] = x;
                    }
                );

            } else {
                auto info = ext->fetch(vbuffer.data());
                subsorted.fill_ranks(info, vec);
                query_has_nonzero = scaled_ranks_dense(num_markers, vec, query_scaled.data());
            }

            curscores.resize(num_labels);
            for (I<decltype(num_labels)> r = 0; r < num_labels; ++r) {
                const auto& qdeets = quantile_details[r];
                const Index_ k = qdeets.right_index + 1; // cast is safe as k <= num_samples.
                find_closest_neighbors<ref_sparse_>(num_markers, query_scaled, query_has_nonzero, k, ref[r], find_work);

                const Float_ right_l2 = get_furthest_neighbor(find_work).first;
                const Float_ right_cor = l2_to_correlation(right_l2);
                if (!qdeets.find_left) {
                    curscores[r] = right_cor;
                } else {
                    pop_furthest_neighbor(find_work);
                    const Float_ left_l2 = get_furthest_neighbor(find_work).first;
                    const Float_ left_cor = l2_to_correlation(left_l2);
                    curscores[r] = left_cor + (right_cor - left_cor) * qdeets.right_prop; // see l2_to_score() for more details.
                }

                if (scores[r]) {
                    scores[r][c] = curscores[r];
                }
            }

            std::pair<Label_, Float_> chosen;
            if (!fine_tune) {
                chosen = find_best_and_delta<Label_>(curscores);
            } else {
                chosen = ft->run(vec, trained, quantile_details, threshold, curscores);
            }

            best[c] = chosen.first;
            if (delta) {
                delta[c] = chosen.second;
            }
        }
    }, test.ncol(), num_threads);

    return;
}

template<typename Value_, typename Index_, typename Float_, typename Label_>
void annotate_cells_single(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingle<Index_, Float_>& trained,
    Float_ quantile,
    bool fine_tune,
    Float_ threshold,
    Label_* best, 
    const std::vector<Float_*>& scores,
    Float_* delta,
    int num_threads
) {
    const auto ref_sparse = trained.built().sparse.has_value();
    if (test.is_sparse()) {
        if (ref_sparse) {
            annotate_cells_single_raw<true, true>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        } else {
            annotate_cells_single_raw<true, false>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        }
    } else {
        if (ref_sparse) {
            annotate_cells_single_raw<false, true>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        } else {
            annotate_cells_single_raw<false, false>(test, trained, quantile, fine_tune, threshold, best, scores, delta, num_threads);
        }
    }
}

}

#endif
