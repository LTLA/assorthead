#ifndef SINGLEPP_TRAIN_INTEGRATED_HPP
#define SINGLEPP_TRAIN_INTEGRATED_HPP

#include "defs.hpp"

#include "train_single.hpp"
#include "Intersection.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <memory>

/**
 * @file train_integrated.hpp
 * @brief Prepare for integrated classification across references.
 */

namespace singlepp {

/**
 * @brief Input to `train_integrated()`.
 *
 * Instances of this class should not be manually created but instead returned by `prepare_integrated_input()`. 
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 */
template<typename Value_, typename Index_, typename Label_>
struct TrainIntegratedInput {
    /**
     * @cond
     */
    const tatami::Matrix<Value_, Index_>* ref;
    const Label_* labels;
    const Markers<Index_>* ref_markers;
    const std::vector<Index_>* test_subset;
    Index_ test_nrow;
    std::shared_ptr<const Intersection<Index_> > intersection;
    /**
     * @endcond
     */
};

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload assumes that the reference and test datasets have the same genes.
 * All inputs are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param ref Matrix containing the reference expression values, where rows are genes and columns are reference profiles.
 * The number and identity of genes should be identical to the test dataset to be classified in `classify_integrated()`.
 * @param[in] labels Pointer to an array of label assignments.
 * Values should be integers in \f$[0, L)\f$ where \f$L\f$ is the number of unique labels.
 * @param trained Classifier created by calling `train_single()` on `ref` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Value_, typename Index_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels, 
    const TrainedSingle<Index_, Float_>& trained
) {
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.ref = &ref;
    output.labels = labels;

    output.ref_markers = &(trained.markers());
    output.test_subset = &(trained.subset());

    output.test_nrow = ref.nrow(); // remember, test and ref are assumed to have the same features.
    return output;
}

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload requires an existing intersection between the test and reference datasets. 
 * All inputs are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param test_nrow Number of features in the test dataset.
 * @param intersection Vector defining the intersection of genes between the test and reference datasets. 
 * Each pair corresponds to a gene where the first and second elements represent the row indices of that gene in the test and reference matrices, respectively.
 * The first element of each pair should be non-negative and less than `test_nrow`, while the second element should be non-negative and less than `ref->nrow()`.
 * See `intersect_genes()` for more details.
 * @param ref Matrix containing the reference expression values, where rows are genes and columns are reference profiles.
 * The number and identity of genes should be consistent with `intersection`.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Values should be integers in \f$[0, L)\f$ where \f$L\f$ is the number of unique labels.
 * @param trained Classifier created by calling `train_single()` on `test_nrow`, `intersection`, `ref` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Index_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input(
    Index_ test_nrow,
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels, 
    const TrainedSingle<Index_, Float_>& trained
) {
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.ref = &ref;
    output.labels = labels;

    output.ref_markers = &(trained.markers());
    output.test_subset = &(trained.subset());

    output.test_nrow = test_nrow;
    output.intersection = std::shared_ptr<const Intersection<Index_> >(std::shared_ptr<Intersection<Index_> >{}, &intersection);
    return output;
}

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload automatically identifies the intersection of genes between the test and reference datasets.
 * All inputs are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Id_ Type of the gene identifier. 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param test_nrow Number of rows (i.e., genes) in the test dataset.
 * @param[in] test_id Pointer to an array of length equal to `test_nrow`.
 * This should contain a unique identifier for each row of `mat`, typically a gene name or index.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] ref_id Pointer to an array equal to the number of rows of `ref`.
 * This should contain a unique identifier for each row in `ref`, comparable to those in `test_id`.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each sample.
 * Values should be integers in \f$[0, L)\f$ where \f$L\f$ is the number of unique labels.
 * @param trained Classifier created by calling `train_single()` on `test_nrow`, `test_id`, `ref`, `ref_id` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Index_, typename Id_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const Label_* labels,
    const TrainedSingle<Index_, Float_>& trained
) {
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.ref = &ref;
    output.labels = labels;

    output.ref_markers = &(trained.markers());
    output.test_subset = &(trained.subset());

    output.test_nrow = test_nrow;
    auto intersection = intersect_genes(test_nrow, test_id, ref.nrow(), ref_id);
    output.intersection = std::shared_ptr<const Intersection<Index_> >(new Intersection<Index_>(std::move(intersection)));
    return output;
}

/**
 * @cond
 */
template<typename Index_>
struct IntegratedReference {
    struct DensePerLabel {
        Index_ num_samples;
        std::vector<Index_> markers; // indices to 'universe'
        RankedVector<Index_, Index_> all_ranked; // .second contains indices to 'universe'
    };

    struct SparsePerLabel {
        Index_ num_samples;
        std::vector<Index_> markers; // indices to 'universe'
        RankedVector<Index_, Index_> negative_ranked, positive_ranked; // .second contains indices to 'universe'
        std::vector<std::size_t> negative_indptrs, positive_indptrs; 
    };

    std::optional<std::vector<DensePerLabel> > dense;
    std::optional<std::vector<SparsePerLabel> > sparse;
};
/**
 * @endcond
 */

/**
 * @brief Classifier that integrates multiple reference datasets.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 */
template<typename Index_>
class TrainedIntegrated {
public:
    /**
     * @cond
     */
    TrainedIntegrated(Index_ test_nrow, std::vector<Index_> universe, std::vector<IntegratedReference<Index_> > references) :
        my_test_nrow(test_nrow),
        my_universe(std::move(universe)),
        my_references(std::move(references))
    {}

    const auto& references() const {
        return my_references;
    }
    /**
     * @endcond
     */

private:
    Index_ my_test_nrow;
    std::vector<Index_> my_universe;
    std::vector<IntegratedReference<Index_> > my_references;

public:
    /**
     * @return Number of reference datasets.
     */
    std::size_t num_references() const {
        return my_references.size();
    }

    /**
     * @return Number of rows that should be present in the test dataset.
     */
    Index_ test_nrow() const {
        return my_test_nrow;
    }

    /**
     * @return The subset of genes in the test dataset to be used for classification.
     * Each value is a row index into the test matrix.
     * Values are sorted and unique.
     */
    const std::vector<Index_>& subset() const {
        return my_universe;
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of labels in this reference.
     */
    std::size_t num_labels(std::size_t r) const {
        const auto& ref = my_references[r];
        if (ref.dense.has_value()) {
            return ref.dense->size();
        } else {
            return ref.sparse->size();
        }
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of profiles in this reference.
     */
    std::size_t num_profiles(std::size_t r) const {
        std::size_t num_prof = 0;
        const auto& ref = my_references[r];
        if (ref.dense.has_value()) {
            for (const auto& lab : *(ref.dense)) {
                num_prof += sanisizer::sum<std::size_t>(num_prof, lab.num_samples);
            }
        } else {
            for (const auto& lab : *(ref.sparse)) {
                num_prof += sanisizer::sum<std::size_t>(num_prof, lab.num_samples);
            }
        }
        return num_prof;
    }
};

/**
 * @brief Options for `train_integrated()`. 
 */
struct TrainIntegratedOptions {
    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<bool ref_sparse_, typename Value_, typename Index_, typename Label_>
void train_integrated_per_reference_simple(
    const TrainIntegratedInput<Value_, Label_, Index_>& input,
    const std::vector<Index_>& universe,
    const std::vector<Index_>& remap_test_to_universe,
    const TrainIntegratedOptions& options,
    const std::vector<Index_>& positions,
    std::vector<std::vector<RankedVector<Index_, Index_> > >& out_ranked,
    typename std::conditional<ref_sparse_, std::vector<std::vector<RankedVector<Index_, Index_> > >&, bool>::type other_ranked 
) {
    const auto& ref = *(input.ref);
    const auto NC = ref.ncol();
    const auto num_universe = universe.size();

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        auto vbuffer = sanisizer::create<std::vector<Value_> >(num_universe);
        auto ibuffer = [&](){
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Index_> >(num_universe);
            } else {
                return false;
            }
        }();

        RankedVector<Value_, Index_> tmp_ranked;
        tmp_ranked.reserve(num_universe);

        // 'universe' technically refers to the row indices of the test matrix,
        // but in simple mode, the rows of the test and reference are the same, so we can use it directly here.
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &universe);
        auto ext = tatami::consecutive_extractor<ref_sparse_>(ref, false, start, len, std::move(universe_ptr)); 

        for (Index_ c = start, end = start + len; c < end; ++c) {
            tmp_ranked.clear();

            if constexpr(ref_sparse_) {
                auto info = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                    const auto remapped = remap_test_to_universe[info.index[i]];
                    assert(sanisizer::is_less_than(remapped, num_universe));
                    tmp_ranked.emplace_back(info.value[i], remapped);
                }
            } else {
                auto ptr = ext->fetch(vbuffer.data());
                for (I<decltype(num_universe)> i = 0; i < num_universe; ++i) {
                    tmp_ranked.emplace_back(ptr[i], i); // a.k.a. remap_test_to_universe[universe[i]]. 
                }
            }

            std::sort(tmp_ranked.begin(), tmp_ranked.end());

            if constexpr(ref_sparse_) {
                const auto tStart = tmp_ranked.begin(), tEnd = tmp_ranked.end();
                auto zero_ranges = find_zero_ranges<Value_, Index_>(tStart, tEnd);
                simplify_ranks<Value_, Index_>(tStart, zero_ranges.first, out_ranked[input.labels[c]][positions[c]]);
                simplify_ranks<Value_, Index_>(zero_ranges.second, tEnd, other_ranked[input.labels[c]][positions[c]]);
            } else {
                simplify_ranks(tmp_ranked, out_ranked[input.labels[c]][positions[c]]);
            }
        }
    }, NC, options.num_threads);
}

template<bool ref_sparse_, typename Value_, typename Index_, typename Label_>
void train_integrated_per_reference_intersect(
    const TrainIntegratedInput<Value_, Label_, Index_>& input,
    const std::vector<Index_>& remap_test_to_universe,
    const Index_ test_nrow,
    const TrainIntegratedOptions& options,
    const std::vector<Index_>& positions,
    std::vector<std::vector<RankedVector<Index_, Index_> > >& out_ranked,
    typename std::conditional<ref_sparse_, std::vector<std::vector<RankedVector<Index_, Index_> > >&, bool>::type other_ranked 
) {
    const auto& ref = *(input.ref);
    const auto NC = ref.ncol();

    std::vector<Index_> ref_subset;
    sanisizer::reserve(ref_subset, input.intersection->size());
    auto remap_ref_subset_to_universe = sanisizer::create<std::vector<Index_> >(ref.nrow(), test_nrow); // all entries of remap_test_to_universe are less than test_nrow. 
    for (const auto& pair : *(input.intersection)) {
        const auto rdex = remap_test_to_universe[pair.first];
        if (rdex != test_nrow) {
            ref_subset.push_back(pair.second);
            remap_ref_subset_to_universe[pair.second] = rdex;
        }
    }
    std::sort(ref_subset.begin(), ref_subset.end());

    typename std::conditional<ref_sparse_, bool, std::vector<Index_> >::type remap_dense_to_universe;
    if constexpr(!ref_sparse_) {
        remap_dense_to_universe.reserve(ref_subset.size());
        for (auto r : ref_subset) {
            remap_dense_to_universe.push_back(remap_ref_subset_to_universe[r]);
        }
    }

    tatami::parallelize([&](int, Index_ start, Index_ len) {
        const auto ref_subset_size = ref_subset.size();
        auto vbuffer = sanisizer::create<std::vector<Value_> >(ref_subset_size);
        auto ibuffer = [&]() {
            if constexpr(ref_sparse_) {
                return sanisizer::create<std::vector<Index_> >(ref_subset_size);
            } else {
                return false;
            }
        }();

        RankedVector<Value_, Index_> tmp_ranked;
        tmp_ranked.reserve(ref_subset_size);
        tatami::VectorPtr<Index_> to_extract_ptr(tatami::VectorPtr<Index_>{}, &ref_subset);
        auto ext = tatami::consecutive_extractor<ref_sparse_>(ref, false, start, len, std::move(to_extract_ptr));

        for (Index_ c = start, end = start + len; c < end; ++c) {
            tmp_ranked.clear();

            if constexpr(ref_sparse_) {
                auto info = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                    tmp_ranked.emplace_back(info.value[i], remap_ref_subset_to_universe[info.index[i]]);
                }
            } else {
                auto ptr = ext->fetch(vbuffer.data());
                for (I<decltype(ref_subset_size)> i = 0; i < ref_subset_size; ++i) {
                    tmp_ranked.emplace_back(ptr[i], remap_dense_to_universe[i]);
                }
            }

            std::sort(tmp_ranked.begin(), tmp_ranked.end());

            if constexpr(ref_sparse_) {
                const auto tStart = tmp_ranked.begin(), tEnd = tmp_ranked.end();
                auto zero_ranges = find_zero_ranges<Value_, Index_>(tStart, tEnd);
                simplify_ranks<Value_, Index_>(tStart, zero_ranges.first, out_ranked[input.labels[c]][positions[c]]);
                simplify_ranks<Value_, Index_>(zero_ranges.second, tEnd, other_ranked[input.labels[c]][positions[c]]);
            } else {
                simplify_ranks(tmp_ranked, out_ranked[input.labels[c]][positions[c]]);
            }
        }
    }, NC, options.num_threads);
}
/**
 * @endcond
 */

/**
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the labels.
 *
 * @param inputs Vector of references, each of which is typically constructed with `prepare_integrated_input()`.
 * @param options Further options.
 *
 * @return A pre-built classifier that integrates multiple references, for use in `classify_integrated()`.
 */
template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(const std::vector<TrainIntegratedInput<Value_, Index_, Label_> >& inputs, const TrainIntegratedOptions& options) {
    std::vector<Index_> universe;
    const auto nrefs = inputs.size();
    auto references = sanisizer::create<std::vector<IntegratedReference<Index_> > >(nrefs);

    // Checking that the number of genes in the test dataset are consistent.
    Index_ test_nrow = 0;
    if (inputs.size()) {
        test_nrow = inputs.front().test_nrow;
        for (const auto& in : inputs) {
            if (!sanisizer::is_equal(in.test_nrow, test_nrow)) {
                throw std::runtime_error("inconsistent number of rows in the test dataset across entries of 'inputs'");
            }
        }
    }

    // Identify the union of all marker genes as the universe, but excluding those markers that are not present in intersections.
    // Note that 'universe' contains sorted and unique row indices for the test matrix, where 'remap_test_to_universe[universe[i]] == i'.
    auto remap_test_to_universe = sanisizer::create<std::vector<Index_> >(test_nrow, test_nrow);
    {
        auto present = sanisizer::create<std::vector<char> >(test_nrow);
        auto count_refs = sanisizer::create<std::vector<I<decltype(nrefs)> > >(test_nrow);
        universe.reserve(test_nrow);

        for (const auto& in : inputs) {
            const auto& markers = *(in.ref_markers);
            const auto& test_subset = *(in.test_subset);

            for (const auto& labmrk : markers) {
                for (const auto& mrk : labmrk) {
                    for (const auto y : mrk) {
                        const auto ty = test_subset[y];
                        if (!present[ty]) {
                            present[ty] = true;
                            universe.push_back(ty);
                        }
                    }
                }
            }

            if (in.intersection) {
                for (const auto& pp : *(in.intersection)) {
                    count_refs[pp.first] += 1;
                }
            } else {
                for (auto& x : count_refs) {
                    x += 1;
                }
            }
        }

        std::sort(universe.begin(), universe.end());
        const auto num_universe = universe.size();
        I<decltype(num_universe)> keep = 0;
        for (I<decltype(num_universe)> u = 0; u < num_universe; ++u) {
            const auto marker = universe[u];
            if (count_refs[marker] == nrefs) {
                universe[keep] = marker;
                remap_test_to_universe[marker] = keep;
                ++keep;
            }
        }
        universe.resize(keep);
        universe.shrink_to_fit();
    }

    auto is_active = sanisizer::create<std::vector<char> >(test_nrow);
    std::vector<Index_> active_genes;
    active_genes.reserve(test_nrow);

    for (I<decltype(nrefs)> r = 0; r < nrefs; ++r) {
        const auto& curinput = inputs[r];
        const auto& currefmarkers = *(curinput.ref_markers);
        const auto& test_subset = *(curinput.test_subset);
        const auto nlabels = currefmarkers.size();
        auto& currefout = references[r];

        const Index_ NC = curinput.ref->ncol();
        const bool is_sparse = curinput.ref->is_sparse();
        if (is_sparse) {
            currefout.sparse.emplace(sanisizer::as_size_type<I<decltype(*(currefout.sparse))> >(nlabels));
        } else {
            currefout.dense.emplace(sanisizer::as_size_type<I<decltype(*(currefout.dense))> >(nlabels));
        }

        // Assembling the per-label markers for this reference.
        for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
            active_genes.clear();
            for (const auto& labmark : currefmarkers[l]) {
                for (const auto y : labmark) {
                    const auto ty = test_subset[y];
                    if (!is_active[ty]) {
                        is_active[ty] = true;
                        active_genes.push_back(ty);
                    }
                }
            }

            std::vector<Index_> markers;
            markers.reserve(active_genes.size());

            for (const auto a : active_genes) {
                const auto universe_index = remap_test_to_universe[a];
                if (universe_index != test_nrow) { // ignoring genes not in the intersection.
                    markers.push_back(universe_index);
                }
                is_active[a] = false;
            }

            if (is_sparse) {
                (*(currefout.sparse))[l].markers.swap(markers);
            } else {
                (*(currefout.dense))[l].markers.swap(markers);
            }
        }

        // Pre-allocating the ranked vectors.
        std::vector<Index_> positions;
        positions.reserve(NC);
        auto samples_per_label = sanisizer::create<std::vector<Index_> >(nlabels);
        for (Index_ c = 0; c < NC; ++c) {
            auto& pos = samples_per_label[curinput.labels[c]];
            positions.push_back(pos);
            ++pos;
        }

        if (curinput.ref->is_sparse()) {
            auto negative_ranked = sanisizer::create<std::vector<std::vector<RankedVector<Index_, Index_> > > >(nlabels);
            auto positive_ranked = sanisizer::create<std::vector<std::vector<RankedVector<Index_, Index_> > > >(nlabels);
            for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
                const auto num_samples = samples_per_label[l];
                sanisizer::resize(negative_ranked[l], num_samples);
                sanisizer::resize(positive_ranked[l], num_samples);
            }

            if (curinput.intersection) {
                train_integrated_per_reference_intersect<true>(curinput, remap_test_to_universe, test_nrow, options, positions, negative_ranked, positive_ranked);
            } else {
                train_integrated_per_reference_simple<true, Value_>(curinput, universe, remap_test_to_universe, options, positions, negative_ranked, positive_ranked);
            }

            for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
                auto& curlabout = (*(currefout.sparse))[l];
                const auto num_samples = samples_per_label[l];
                curlabout.num_samples = num_samples;

                I<decltype(curlabout.negative_ranked.size())> num_neg = 0;
                for (const auto& x : negative_ranked[l]) {
                    num_neg = sanisizer::sum<I<decltype(num_neg)> >(num_neg, x.size());
                }

                I<decltype(curlabout.positive_ranked.size())> num_pos = 0;
                for (const auto& x : positive_ranked[l]) {
                    num_pos = sanisizer::sum<I<decltype(num_pos)> >(num_pos, x.size());
                }

                curlabout.negative_ranked.reserve(num_neg);
                curlabout.negative_indptrs.reserve(sanisizer::sum<I<decltype(curlabout.negative_indptrs.size())> >(num_samples, 1));
                curlabout.negative_indptrs.push_back(0);
                for (const auto& x : negative_ranked[l]) {
                    curlabout.negative_ranked.insert(curlabout.negative_ranked.end(), x.begin(), x.end());
                    curlabout.negative_indptrs.push_back(curlabout.negative_ranked.size());
                }

                curlabout.positive_ranked.reserve(num_pos);
                curlabout.positive_indptrs.reserve(sanisizer::sum<I<decltype(curlabout.positive_indptrs.size())> >(num_samples, 1));
                curlabout.positive_indptrs.push_back(0);
                for (const auto& x : positive_ranked[l]) {
                    curlabout.positive_ranked.insert(curlabout.positive_ranked.end(), x.begin(), x.end());
                    curlabout.positive_indptrs.push_back(curlabout.positive_ranked.size());
                }
            }

        } else {
            auto out_ranked = sanisizer::create<std::vector<std::vector<RankedVector<Index_, Index_> > > >(nlabels);
            for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
                const auto num_samples = samples_per_label[l];
                sanisizer::resize(out_ranked[l], num_samples);
            }

            if (curinput.intersection) {
                train_integrated_per_reference_intersect<false>(curinput, remap_test_to_universe, test_nrow, options, positions, out_ranked, true);
            } else {
                train_integrated_per_reference_simple<false, Value_>(curinput, universe, remap_test_to_universe, options, positions, out_ranked, true);
            }

            for (I<decltype(nlabels)> l = 0; l < nlabels; ++l) {
                auto& curlabout = (*(currefout.dense))[l];
                curlabout.num_samples = samples_per_label[l];
                curlabout.all_ranked.reserve(sanisizer::product<I<decltype(curlabout.all_ranked.size())> >(universe.size(), curlabout.num_samples));
                for (const auto& x : out_ranked[l]) {
                    curlabout.all_ranked.insert(curlabout.all_ranked.end(), x.begin(), x.end());
                }
            }
        }
    }

    return TrainedIntegrated<Index_>(test_nrow, std::move(universe), std::move(references));
}

}

#endif
