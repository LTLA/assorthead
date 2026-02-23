#ifndef SINGLEPP_BUILD_REFERENCE_HPP
#define SINGLEPP_BUILD_REFERENCE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"
#include "aarand/aarand.hpp"

#include "utils.hpp"
#include "scaled_ranks.hpp"
#include "SubsetSanitizer.hpp"
#include "l2.hpp"

#include <vector>
#include <memory>
#include <algorithm>
#include <cstddef> 
#include <random>
#include <optional>
#include <cassert>

namespace singlepp {

template<typename Index_, typename Float_>
struct DensePerLabel {
    std::vector<Float_> data;

    // Structures for the KMKNN search.
    std::vector<Float_> distances;
    std::vector<std::pair<Index_, Index_> > seed_ranges;

    // Concatenation of RankedVectors for all samples.
    RankedVector<Index_, Index_> all_ranked;
};

template<typename Index_, typename Float_>
const Float_* retrieve_vector(const Index_ num_markers, const DensePerLabel<Index_, Float_>& ref, const Index_ col) {
    return ref.data.data() + sanisizer::product_unsafe<std::size_t>(num_markers, col);
}

/*** Sparse matrix constructs ***/

template<typename Index_, typename Float_>
struct SparsePerLabel {
    std::vector<Float_> value;
    std::vector<Index_> index;
    std::vector<std::size_t> indptrs;
    std::vector<Float_> zeros;

    // Structures for the KMKNN search.
    std::vector<Float_> distances;
    std::vector<std::pair<Index_, Index_> > seed_ranges;

    // Concatenation of RankedVectors for all samples.
    RankedVector<Index_, Index_> negative_ranked, positive_ranked;
    std::vector<std::size_t> negative_indptrs, positive_indptrs;
};

template<typename Index_, typename Float_>
struct CompressedSparseVector {
    Index_ number;
    const Float_* value;
    const Index_* index;
    Float_ zero;
    Float_* remapping = NULL;
};

template<typename Index_, typename Float_>
CompressedSparseVector<Index_, Float_> retrieve_vector([[maybe_unused]] const Index_ num_markers, const SparsePerLabel<Index_, Float_>& mat, const Index_ col) {
    CompressedSparseVector<Index_, Float_> output;
    const auto seed_start = mat.indptrs[col], seed_len = mat.indptrs[col + 1] - seed_start;
    output.number = seed_len;
    output.index = mat.index.data() + seed_start;
    output.value = mat.value.data() + seed_start;
    output.zero = mat.zeros[col];
    return output;
}

template<typename Index_, typename Float_>
Index_ get_sparse_num(const CompressedSparseVector<Index_, Float_>& x) { return x.number; }

template<typename Index_, typename Float_>
Index_ get_sparse_index(const CompressedSparseVector<Index_, Float_>& x, const Index_ i) { return x.index[i]; }

template<typename Index_, typename Float_>
Float_ get_sparse_value(const CompressedSparseVector<Index_, Float_>& x, const Index_ i) { return x.value[i]; }

template<typename Index_, typename Float_>
Float_ get_sparse_zero(const CompressedSparseVector<Index_, Float_>& x) { return x.zero; }

template<typename Index_, typename Float_>
void check_sparse_index_sorted_and_unique(const CompressedSparseVector<Index_, Float_>& x) { assert(is_sorted_unique(x.number, x.index)); }

/*** KMKNN building ***/ 

template<typename Index_, typename Float_>
Index_ get_num_samples(const DensePerLabel<Index_, Float_>& ref) {
    return ref.distances.size();
}

template<typename Index_, typename Float_>
Index_ get_num_samples(const SparsePerLabel<Index_, Float_>& ref) {
    return ref.distances.size();
}

template<bool ref_sparse_, typename Index_, typename Float_>
std::vector<Index_> select_seeds(
    const Index_ num_markers,
    const Index_ num_samples,
    typename std::conditional<ref_sparse_, SparsePerLabel<Index_, Float_>, DensePerLabel<Index_, Float_> >::type& ref
) {
    // No need to check for overlow, num_samples >= num_seeds here.
    Index_ num_seeds = std::round(std::sqrt(num_samples));

    // Implementing a variant of kmeans++ initialization to select representative ("seed") points.
    // We also record the minimum distance between each sample and its assigned seed.
    auto assignment = sanisizer::create<std::vector<Index_> >(num_samples);
    auto mindist = sanisizer::create<std::vector<Float_> >(num_samples, 1);
    std::vector<Index_> identities;
    {
        sanisizer::reserve(identities, num_samples);
        auto cumulative = sanisizer::create<std::vector<Float_> >(num_samples);
        std::mt19937_64 eng(/* seed = */ 6237u + num_markers * static_cast<std::size_t>(num_samples)); // making a semi-deterministic seed that depends on the input data. 

        auto densified_seed = [&](){
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Float_> >(num_markers);
            } else {
                return false;
            }
        }();

        for (Index_ se = 0; se < num_seeds; ++se) {
            cumulative[0] = mindist[0];
            for (Index_ sam = 1; sam < num_samples; ++sam) {
                cumulative[sam] = cumulative[sam - 1] + mindist[sam];
            }

            const auto total = cumulative.back();
            if (total == 0) { // a.k.a. only duplicates left.
                break;
            }

            Index_ chosen_id = 0;
            do {
                const Float_ sampled_weight = total * aarand::standard_uniform<Float_>(eng);
                // Subtraction is safe as we already checked for valid ptrdiff.
                chosen_id = std::lower_bound(cumulative.begin(), cumulative.end(), sampled_weight) - cumulative.begin();

                // We wrap this in a do/while to defend against edge cases where ties are chosen.
                // The most obvious of these is when you get a `sampled_weight` of zero _and_ there exists a bunch of zeros at the start of `cumulative`.
                // One could also get unexpected ties from limited precision in floating point comparisons, so we'll just be safe and implement a loop here.
            } while (chosen_id == num_samples || mindist[chosen_id] == 0);

            mindist[chosen_id] = 0;
            assignment[chosen_id] = se;
            identities.push_back(chosen_id);

            // Now updating the distances and assignments of all observations based on the new seed.
            const auto seed_info = retrieve_vector(num_markers, ref, chosen_id);
            bool seed_has_nonzero = false;
            if constexpr(ref_sparse_) {
                seed_has_nonzero = densify_sparse_vector(num_markers, seed_info, densified_seed);
            }

            for (Index_ sam = 0; sam < num_samples; ++sam) {
                auto& mdist = mindist[sam];
                if (mdist == 0) {
                    continue;
                }

                const auto sam_info = retrieve_vector(num_markers, ref, sam);
                const auto l2 = [&](){
                    if constexpr(ref_sparse_) {
                        return sparse_l2(num_markers, densified_seed.data(), seed_has_nonzero, sam_info);
                    } else {
                        return dense_l2(num_markers, seed_info, sam_info);
                    }
                }();

                if (se == 0) {
                    mdist = l2;
                } else if (l2 < mdist) {
                    mdist = l2;
                    assignment[sam] = se;
                }
            }
        }

        num_seeds = identities.size(); // updating for the actual number of seeds, if there were duplicates.
    }

    // Now populating the rest of the seed information.
    {
        auto grouping = sanisizer::create<std::vector<std::vector<std::pair<Float_, Index_> > > >(num_seeds);
        for (Index_ sam = 0; sam < num_samples; ++sam) {
            grouping[assignment[sam]].emplace_back(mindist[sam], sam);
        }

        ref.distances.reserve(num_samples);
        sanisizer::resize(ref.distances, num_seeds);
        ref.seed_ranges.reserve(num_seeds);

        for (Index_ se = 0; se < num_seeds; ++se) {
            auto& curgroup = grouping[se];
            std::sort(curgroup.begin(), curgroup.end());

            // -1 to remove the seed point itself.
            const auto range_start = ref.distances.size();
            ref.seed_ranges.emplace_back(range_start, curgroup.size() - 1);

            const auto seed = identities[se];
            for (const auto& x : curgroup) {
                const auto curid = x.second;
                if (curid != seed) {
                    ref.distances.push_back(std::sqrt(x.first));
                    identities.push_back(curid);
                }
            }
        }
    }

    if constexpr(ref_sparse_) {
        // Resorting the data to be more cache-friendly.
        // This is too hard to do in-place with sparse data, so we just run over it and do it manually.
        std::vector<Float_> value;
        value.reserve(ref.value.size());
        std::vector<Index_> index;
        index.reserve(ref.index.size());
        std::vector<std::size_t> indptrs;
        indptrs.reserve(ref.indptrs.size());
        indptrs.push_back(0);
        std::vector<Float_> zeros;
        zeros.reserve(ref.zeros.size());

        for (auto sam : identities) {
            const auto sam_start = ref.indptrs[sam], sam_end = ref.indptrs[sam + 1];
            value.insert(value.end(), ref.value.begin() + sam_start, ref.value.begin() + sam_end);
            index.insert(index.end(), ref.index.begin() + sam_start, ref.index.begin() + sam_end);
            indptrs.push_back(value.size());
            zeros.push_back(ref.zeros[sam]);
        }

        ref.value.swap(value);
        ref.index.swap(index);
        ref.indptrs.swap(indptrs);
        ref.zeros.swap(zeros);

    } else {
        // Reordering the data in-place to be more cache-friendly.
        auto used = sanisizer::create<std::vector<char> >(num_samples);
        auto data_buffer = sanisizer::create<std::vector<Float_> >(num_markers);

        for (Index_ x = 0; x < num_samples; ++x) {
            if (used[x]) {
                continue;
            }

            Index_ replacement = identities[x];
            if (replacement == x) {
                continue;
            }

            auto previous_ptr = ref.data.data() + sanisizer::product_unsafe<std::size_t>(x, num_markers);
            std::copy_n(previous_ptr, num_markers, data_buffer.data());

            do {
                auto next_ptr = ref.data.data() + sanisizer::product_unsafe<std::size_t>(replacement, num_markers);
                std::copy_n(next_ptr, num_markers, previous_ptr);
                previous_ptr = next_ptr;

                used[replacement] = true;
                replacement = identities[replacement];
            } while (replacement != x);

            std::copy_n(data_buffer.data(), num_markers, previous_ptr);
        }
    }

    return identities;
}

/*** KMKNN search ***/ 

template<typename Index_, typename Float_>
struct FindClosestNeighborsWorkspace {
    FindClosestNeighborsWorkspace(Index_ num_samples) {
        sanisizer::reserve(seed_distances, num_samples);
        sanisizer::reserve(closest_neighbors, num_samples);
    }
    std::vector<std::pair<Float_, Index_> > seed_distances;
    std::vector<std::pair<Float_, Index_> > closest_neighbors;
};

template<bool ref_sparse_, typename Index_, typename Float_>
void find_closest_neighbors(
    const Index_ num_markers,
    const std::vector<Float_>& query,
    const bool query_has_nonzero,
    const Index_ k,
    const typename std::conditional<ref_sparse_, SparsePerLabel<Index_, Float_>, DensePerLabel<Index_, Float_> >::type& ref,
    FindClosestNeighborsWorkspace<Index_, Float_>& work
) {
    const auto num_seeds = ref.seed_ranges.size();
    const auto num_neighbors = sanisizer::cast<I<decltype(work.closest_neighbors.size())> >(k);

    auto compute_distance = [&](Index_ se) -> Float_ {
        const auto refinfo = retrieve_vector(num_markers, ref, se);
        if constexpr(ref_sparse_) {
            return sparse_l2(num_markers, query.data(), query_has_nonzero, refinfo);
        } else {
            return dense_l2(num_markers, query.data(), refinfo);
        }
    };

    // First compute the distance from the query to each seed and sort in increasing order.
    work.seed_distances.clear();
    for (I<decltype(num_seeds)> se = 0; se < num_seeds; ++se) {
        const auto dist_raw = compute_distance(se);
        work.seed_distances.emplace_back(dist_raw, se);
    }
    std::sort(work.seed_distances.begin(), work.seed_distances.end());

    work.closest_neighbors.clear();
    const auto to_add = sanisizer::min(num_neighbors, work.seed_distances.size()); // adding the smallest distances preferentially.
    work.closest_neighbors.insert(work.closest_neighbors.end(), work.seed_distances.begin(), work.seed_distances.begin() + to_add);
    std::make_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());
    Float_ threshold_raw = (work.closest_neighbors.size() < num_neighbors ? std::numeric_limits<Float_>::infinity() : work.closest_neighbors.front().first);

    // Now we traverse each seed.
    for (const auto& curseed : work.seed_distances) {
        const Index_ seed = curseed.second;
        const auto& ranges = ref.seed_ranges[seed];
        if (ranges.second == 0) { // i.e., the seed was the only point in its own cluster.
            continue;
        }

        Index_ firstsubj = ranges.first, lastsubj = ranges.first + ranges.second;
        if (!std::isinf(threshold_raw)) {
            const Float_ threshold = std::sqrt(threshold_raw);
            const Float_ query2seed = std::sqrt(curseed.first);
            const Float_ max_subj2seed = ref.distances[lastsubj - 1];

            /* This exploits the triangle inequality to ignore points where:
             *     threshold + subject-to-seed < query-to-seed 
             * All points (if any) within this cluster with distances at or above 'lower_bd' are potentially countable.
             *
             * If the maximum distance between a subject and the seed is less than 'lower_bd', there's no point proceeding,
             * as we know that all other subjects will have smaller distances and are thus uncountable.
             */
            const Float_ lower_bd = query2seed - threshold;
            if (max_subj2seed < lower_bd) {
                continue;
            }
            firstsubj = std::lower_bound(ref.distances.data() + firstsubj, ref.distances.data() + lastsubj, lower_bd) - ref.distances.data();

            /* This exploits the triangle inequality in an opposite manner to that described above, to ignore points where:
             *     threshold + query-to-seed < subject-to-seed
             * All points (if any) within this cluster with distances at or below 'upper_bd' are potentially countable.
             * 
             * If query-to-seed distance is greater than or equal to the largest possible subject-to-seed for a seed,
             * there's no point exploiting this inequality as we must examine all subjects... so we don't bother with it.
             *
             * We could also skip this seed altogether if the minimum subject-to-seed distance is greater than 'upper_bd'.
             * However, this seems too unlikely to warrant a special clause.
             */
            const Float_ upper_bd = query2seed + threshold;
            if (max_subj2seed > upper_bd) {
                lastsubj = std::upper_bound(ref.distances.data() + firstsubj, ref.distances.data() + lastsubj, upper_bd) - ref.distances.data();
            }
        }

        for (auto s = firstsubj; s < lastsubj; ++s) {
            const auto dist2subj_raw = compute_distance(s);
            if (dist2subj_raw <= threshold_raw) {
                work.closest_neighbors.emplace_back(dist2subj_raw, s);
                std::push_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());

                if (work.closest_neighbors.size() >= num_neighbors) {
                    if (work.closest_neighbors.size() > num_neighbors) {
                        std::pop_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());
                        work.closest_neighbors.pop_back();
                    }
                    threshold_raw = work.closest_neighbors.front().first;

                    /* P.S. We could also consider increasing 'firstsubj' as 'threshold_raw' decreases. 
                     * The idea would be to exploit the triangle inequality to quickly skip over more points. 
                     * However, this is pointless because 'lower_bd' will never increase enough to skip subsequent observations.
                     * We wouldn't have been able to skip the observation that we just added,
                     * so there's no way we could skip observations with larger subject-to-seed distances.
                     *
                     * P.P.S. We could also consider decreasing 'lastsubj' as 'threshold_raw' decreases.
                     * The idea would be to exploit the triangle inequality to terminate sooner. 
                     * However, this doesn't seem to provide a lot of benefit in practice. 
                     * In theory, we can only trim the search space if the query already lies in a seed's hypersphere (as 'upper_bd' cannot decrease below 'query2seed').
                     * Even then, 'upper_bd' is usually too large; testing indicates that a reduced 'upper_bd' only trims away a single observation at a time.
                     * There are also practical challenges as changes to 'lastsubj' within the loop might prevent out-of-order CPU execution;
                     * we need to do more memory accesses to 'dist2seeds' to check if 'lastsubj' can be decreased;
                     * and we need to run an extra 'normalize()' to recompute 'upper_bd' inside the loop.
                     * All in all, I don't think it's worth it.
                     */
                }
            }
        }
    }
}

template<typename Index_, typename Float_>
const std::pair<Float_, Index_>& get_furthest_neighbor(const FindClosestNeighborsWorkspace<Index_, Float_>& work) {
    return work.closest_neighbors.front();
}

template<typename Index_, typename Float_>
void pop_furthest_neighbor(FindClosestNeighborsWorkspace<Index_, Float_>& work) {
    std::pop_heap(work.closest_neighbors.begin(), work.closest_neighbors.end());
}

/*** Overlord function ***/ 

template<typename Index_, typename Float_>
struct BuiltReference {
    std::optional<std::vector<DensePerLabel<Index_, Float_> > > dense;
    std::optional<std::vector<SparsePerLabel<Index_, Float_> > > sparse;
};

template<bool ref_sparse_, typename Index_, typename Float_>
auto& allocate_references(BuiltReference<Index_, Float_>& output, const std::size_t num_labels) {
    if constexpr(ref_sparse_) {
        output.sparse.emplace(sanisizer::cast<I<decltype(output.sparse->size())> >(num_labels));
        return *(output.sparse);
    } else {
        output.dense.emplace(sanisizer::cast<I<decltype(output.dense->size())> >(num_labels));
        return *(output.dense);
    }
}

template<typename Stat_, typename Index_>
std::pair<typename RankedVector<Stat_, Index_>::const_iterator, typename RankedVector<Stat_, Index_>::const_iterator> find_zero_ranges(
    const typename RankedVector<Stat_, Index_>::const_iterator& begin,
    const typename RankedVector<Stat_, Index_>::const_iterator& end
) {
    auto nonneg = std::lower_bound(begin, end, std::make_pair<Stat_, Index_>(0, 0));
    auto pos = nonneg;
    while (pos != end && pos->first == 0) { // Skipping over any explicit zeros, we only want to store positive values here.
        ++pos;
    }
    return std::make_pair(nonneg, pos);
}

template<bool ref_sparse_, typename Float_, typename Value_, typename Index_, typename Label_>
BuiltReference<Index_, Float_> build_reference_raw(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    int num_threads
) {
    const auto num_markers = sanisizer::cast<Index_>(subset.size());
    const auto num_samples = ref.ncol();
    if (num_samples == 0) {
        throw std::runtime_error("reference dataset must have at least one column");
    }

    const auto num_labels = sanisizer::sum<std::size_t>(*std::max_element(labels, labels + num_samples), 1);
    auto label_count = sanisizer::create<std::vector<Index_> >(num_labels);
    auto label_offsets = sanisizer::create<std::vector<Index_> >(num_samples);
    for (I<decltype(num_samples)> i = 0; i < num_samples; ++i) {
        auto& lcount = label_count[labels[i]];
        label_offsets[i] = lcount;
        ++lcount;
    }

    for (I<decltype(num_labels)> l = 0; l < num_labels; ++l) {
        const auto labcount = label_count[l];
        if (labcount == 0) {
            throw std::runtime_error(std::string("no entries for label ") + std::to_string(l));
        }
    }

    BuiltReference<Index_, Float_> output;
    auto& nnrefs = allocate_references<ref_sparse_>(output, num_labels);

    typename std::conditional<ref_sparse_, bool, std::vector<std::vector<RankedVector<Index_, Index_> > > >::type tmp_ref_ranked;
    typename std::conditional<ref_sparse_, std::vector<std::vector<RankedVector<Index_, Index_> > >, bool>::type positive_ref_ranked, negative_ref_ranked;
    sanisizer::cast<typename RankedVector<Index_, Index_>::size_type>(num_markers); // check that we can allocate these inside the loop.
    if constexpr(ref_sparse_) {
        sanisizer::resize(positive_ref_ranked, num_labels);
        sanisizer::resize(negative_ref_ranked, num_labels);
        for (I<decltype(num_labels)> l = 0; l < num_labels; ++l) {
            const auto labcount = label_count[l];
            sanisizer::resize(positive_ref_ranked[l], labcount);
            sanisizer::resize(negative_ref_ranked[l], labcount);
        }
    } else {
        sanisizer::resize(tmp_ref_ranked, num_labels);
        for (I<decltype(num_labels)> l = 0; l < num_labels; ++l) {
            const auto labcount = label_count[l];
            auto& curlab = nnrefs[l];
            curlab.data.resize(sanisizer::product<I<decltype(curlab.data.size())> >(labcount, num_markers));
            sanisizer::resize(tmp_ref_ranked[l], labcount);
        }
    }

    std::optional<SubsetNoop<ref_sparse_, Index_> > subnoop;
    std::optional<SubsetSanitizer<ref_sparse_, Index_> > subsorted;
    const std::vector<Index_>* subptr;
    const bool subset_noop = is_sorted_unique(subset.size(), subset.data());
    if (subset_noop) {
        subnoop.emplace(subset);
        subptr = &(subnoop->extraction_subset());
    } else {
        subsorted.emplace(subset);
        subptr = &(subsorted->extraction_subset());
    }

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        tatami::VectorPtr<Index_> subset_ptr(tatami::VectorPtr<Index_>{}, subptr);
        auto ext = tatami::consecutive_extractor<ref_sparse_>(ref, false, start, len, std::move(subset_ptr));
        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_markers);
        auto ibuffer = [&](){
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_markers);
            } else {
                return false;
            }
        }();

        RankedVector<Value_, Index_> query_ranked;
        sanisizer::reserve(query_ranked, num_markers);

        for (Index_ c = start, end = start + len; c < end; ++c) {
            const auto col = [&](){
                if constexpr(ref_sparse_) {
                    return ext->fetch(vbuffer.data(), ibuffer.data());
                } else {
                    return ext->fetch(vbuffer.data());
                }
            }();

            if (subset_noop) {
                subnoop->fill_ranks(col, query_ranked); 
            } else {
                subsorted->fill_ranks(col, query_ranked); 
            }

            const auto curlab = labels[c];
            const auto curoff = label_offsets[c];

            if constexpr(ref_sparse_) {
                const auto qStart = query_ranked.begin();
                const auto qEnd = query_ranked.end();
                auto zero_ranges = find_zero_ranges<Value_, Index_>(qStart, qEnd);
                simplify_ranks<Value_, Index_>(qStart, zero_ranges.first, negative_ref_ranked[curlab][curoff]);
                simplify_ranks<Value_, Index_>(zero_ranges.second, qEnd, positive_ref_ranked[curlab][curoff]);
            } else {
                simplify_ranks(query_ranked, tmp_ref_ranked[curlab][curoff]);
                const auto scaled = nnrefs[curlab].data.data() + sanisizer::product_unsafe<std::size_t>(curoff, num_markers);
                scaled_ranks_dense(num_markers, query_ranked, scaled); 
            }
        }
    }, num_samples, num_threads);

    tatami::parallelize([&](int, std::size_t start, std::size_t len) {
        for (std::size_t l = start, end = start + len; l < end; ++l) {
            auto& curlab = nnrefs[l]; 
            const auto labcount = label_count[l];

            if constexpr(ref_sparse_) {
                const auto& neg_ranked = negative_ref_ranked[l];
                const auto& pos_ranked = positive_ref_ranked[l];

                std::size_t positive_nzeros = 0, negative_nzeros = 0;
                for (const auto& x : neg_ranked) {
                    negative_nzeros = sanisizer::sum<std::size_t>(negative_nzeros, x.size());
                }
                for (const auto& x : pos_ranked) {
                    positive_nzeros = sanisizer::sum<std::size_t>(positive_nzeros, x.size());
                }

                const std::size_t total_nzeros = sanisizer::sum<std::size_t>(negative_nzeros, positive_nzeros);
                sanisizer::reserve(curlab.value, total_nzeros);
                sanisizer::reserve(curlab.index, total_nzeros);
                curlab.indptrs.reserve(sanisizer::sum<I<decltype(curlab.indptrs.size())> >(labcount, 1));
                curlab.indptrs.push_back(0);
                sanisizer::reserve(curlab.zeros, labcount);

                SparseScaled<Index_, Float_> scaled; 
                sanisizer::reserve(scaled.nonzero, labcount);
                for (Index_ c = 0; c < labcount; ++c) {
                    scaled_ranks_sparse(num_markers, neg_ranked[c], pos_ranked[c], scaled);
                    sort_by_first(scaled.nonzero); 
                    for (const auto& y : scaled.nonzero) {
                        curlab.index.push_back(y.first);
                        curlab.value.push_back(y.second);
                    }
                    curlab.indptrs.push_back(curlab.value.size());
                    curlab.zeros.push_back(scaled.zero);
                }

                auto identities = select_seeds<ref_sparse_, Index_, Float_>(num_markers, labcount, curlab);

                sanisizer::reserve(curlab.negative_ranked, negative_nzeros);
                curlab.negative_indptrs.reserve(sanisizer::sum<I<decltype(curlab.positive_indptrs.size())> >(labcount, 1));
                curlab.negative_indptrs.push_back(0);
                sanisizer::reserve(curlab.positive_ranked, positive_nzeros);
                curlab.positive_indptrs.reserve(sanisizer::sum<I<decltype(curlab.positive_indptrs.size())> >(labcount, 1));
                curlab.positive_indptrs.push_back(0);

                for (auto sam : identities) {
                    curlab.negative_ranked.insert(curlab.negative_ranked.end(), neg_ranked[sam].begin(), neg_ranked[sam].end());
                    curlab.negative_indptrs.push_back(curlab.negative_ranked.size());
                    curlab.positive_ranked.insert(curlab.positive_ranked.end(), pos_ranked[sam].begin(), pos_ranked[sam].end());
                    curlab.positive_indptrs.push_back(curlab.positive_ranked.size());
                }

            } else {
                auto identities = select_seeds<ref_sparse_, Index_, Float_>(num_markers, labcount, curlab);
                sanisizer::reserve(curlab.all_ranked, curlab.data.size());
                const auto& ref_ranked = tmp_ref_ranked[l];
                for (auto sam : identities) {
                    curlab.all_ranked.insert(curlab.all_ranked.end(), ref_ranked[sam].begin(), ref_ranked[sam].end()); 
                }
            }
        }
    }, num_labels, num_threads);

    return output;
}

template<typename Float_, typename Value_, typename Index_, typename Label_>
BuiltReference<Index_, Float_> build_reference(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    int num_threads
) {
    if (ref.is_sparse()) {
        return build_reference_raw<true, Float_>(ref, labels, subset, num_threads); 
    } else {
        return build_reference_raw<false, Float_>(ref, labels, subset, num_threads); 
    }
}

}

#endif
