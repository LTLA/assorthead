#ifndef SINGLEPP_TRAIN_INTEGRATED_HPP
#define SINGLEPP_TRAIN_INTEGRATED_HPP

#include "defs.hpp"

#include "scaled_ranks.hpp"
#include "train_single.hpp"
#include "Intersection.hpp"

#include <vector>
#include <unordered_set>
#include <unordered_map>
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
 * Instances of this class should not be manually created, but instead returned by `prepare_integrated_input()` and `prepare_integrated_input_intersect()`.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 */
template<typename Value_ = DefaultValue, typename Index_ = DefaultIndex, typename Label_ = DefaultLabel>
struct TrainIntegratedInput {
    /**
     * @cond
     */
    Index_ test_nrow;

    const tatami::Matrix<Value_, Index_>* ref;

    const Label_* labels;

    std::vector<std::vector<Index_> > markers;

    bool with_intersection = false;

    const Intersection<Index_>* user_intersection = NULL;

    Intersection<Index_> auto_intersection;
    /**
     * @endcond
     */
};

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload assumes that the reference and test datasets have the same genes.
 * `ref` and `labels` are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
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
    const TrainedSingle<Index_, Float_>& trained)
{
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.test_nrow = ref.nrow(); // remember, test and ref are assumed to have the same features.
    output.ref = &ref;
    output.labels = labels;

    const auto& subset = trained.get_subset();
    const auto& old_markers = trained.get_markers();
    auto nlabels = old_markers.size();

    // Adding the markers for each label, indexed according to their
    // position in the test matrix. This assumes that 'mat_subset' is
    // appropriately specified to contain the test's row indices. 
    auto& new_markers = output.markers;
    new_markers.reserve(nlabels);
    std::unordered_set<Index_> unified;

    for (decltype(nlabels) i = 0; i < nlabels; ++i) {
        unified.clear();
        for (const auto& x : old_markers[i]) {
            unified.insert(x.begin(), x.end());
        }
        new_markers.emplace_back(unified.begin(), unified.end());
        auto& cur_new_markers = new_markers.back();
        for (auto& y : cur_new_markers) {
            y = subset[y];
        }
    }

    return output;
}

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload requires an existing intersection between the test and reference datasets. 
 * `intersection`, `ref` and `labels` are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
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
 * @param trained Classifier created by calling `train_single_intersect()` on `test_nrow`, `intersection`, `ref` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Index_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input_intersect(
    Index_ test_nrow,
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels, 
    const TrainedSingleIntersect<Index_, Float_>& trained) 
{
    TrainIntegratedInput<Value_, Index_, Label_> output;
    output.test_nrow = test_nrow;
    output.ref = &ref;
    output.labels = labels;

    // Updating the markers so that they point to rows of the test matrix.
    const auto& old_markers = trained.get_markers();
    auto nlabels = old_markers.size();
    auto& new_markers = output.markers;
    new_markers.resize(nlabels);

    const auto& test_subset = trained.get_test_subset();
    std::unordered_set<Index_> unified;

    for (decltype(nlabels) i = 0; i < nlabels; ++i) {
        const auto& cur_old_markers = old_markers[i];

        unified.clear();
        for (const auto& x : cur_old_markers) {
            unified.insert(x.begin(), x.end());
        }

        auto& cur_new_markers = new_markers[i];
        cur_new_markers.reserve(unified.size());
        for (auto y : unified) {
            cur_new_markers.push_back(test_subset[y]);
        }
    }

    output.with_intersection = true;
    output.user_intersection = &intersection;
    return output;
}

/**
 * @cond
 */
// For back-compatibility only.
template<typename Index_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input_intersect(
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels, 
    const TrainedSingleIntersect<Index_, Float_>& trained) 
{
    return prepare_integrated_input_intersect<Index_, Value_, Label_, Float_>(-1, intersection, ref, labels, trained);
}
/**
 * @endcond
 */

/**
 * Prepare a reference dataset for `train_integrated()`.
 * This overload automatically identifies the intersection of genes between the test and reference datasets.
 * `ref` and `labels` are expected to remain valid until `train_integrated()` is called.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Id_ Type of the gene identifier. 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
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
 * @param trained Classifier created by calling `train_single_intersect()` on `test_nrow`, `test_id`, `ref`, `ref_id` and `labels`.
 *
 * @return An opaque input object for `train_integrated()`.
 */
template<typename Index_, typename Id_, typename Value_, typename Label_, typename Float_>
TrainIntegratedInput<Value_, Index_, Label_> prepare_integrated_input_intersect(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const Label_* labels,
    const TrainedSingleIntersect<Index_, Float_>& trained) 
{
    auto intersection = intersect_genes(test_nrow, test_id, ref.nrow(), ref_id);
    auto output = prepare_integrated_input_intersect(test_nrow, intersection, ref, labels, trained);
    output.user_intersection = NULL;
    output.auto_intersection.swap(intersection);
    return output;
}

/**
 * @brief Classifier that integrates multiple reference datasets.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 */
template<typename Index_>
class TrainedIntegrated {
public:
    /**
     * @return Number of reference datasets.
     */
    std::size_t num_references() const {
        return markers.size();
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of labels in this reference.
     */
    std::size_t num_labels(std::size_t r) const {
        return markers[r].size();
    }

    /**
     * @param r Reference dataset of interest.
     * @return Number of profiles in this reference.
     */
    std::size_t num_profiles(std::size_t r) const {
        std::size_t n = 0;
        for (const auto& ref : ranked[r]) {
            n += ref.size();
        }
        return n;
    }

public:
    /**
     * @cond
     */
    // Technically this should be private, but it's a pain to add
    // templated friend functions, so I can't be bothered.
    Index_ test_nrow;
    std::vector<Index_> universe; // To be used by classify_integrated() for indexed extraction.

    std::vector<uint8_t> check_availability;
    std::vector<std::unordered_set<Index_> > available; // indices to 'universe'
    std::vector<std::vector<std::vector<Index_> > > markers; // indices to 'universe'
    std::vector<std::vector<std::vector<internal::RankedVector<Index_, Index_> > > > ranked; // .second contains indices to 'universe'
    /**
     * @endcond
     */
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
namespace internal {

template<typename Value_, typename RefLabel_, typename Input_, typename Index_>
void train_integrated_per_reference(
    RefLabel_ ref_i,
    Input_& curinput,
    TrainedIntegrated<Index_>& output,
    const std::unordered_map<Index_, Index_> remap_to_universe,
    const TrainIntegratedOptions& options)
{
    auto curlab = curinput.labels;
    const auto& ref = *(curinput.ref);

    // Reindexing the markers so that they contain indices into to the universe.
    auto& curmarkers = output.markers[ref_i];
    if constexpr(std::is_const<Input_>::value) {
        curmarkers.swap(curinput.markers);
    } else {
        curmarkers = curinput.markers;
    }
    for (auto& outer : curmarkers) {
        for (auto& x : outer) {
            x = remap_to_universe.find(x)->second;
        }
    }

    // Pre-allocating the vectors of pre-ranked expression.
    auto& cur_ranked = output.ranked[ref_i];
    std::vector<Index_> positions;
    {
        auto nlabels = curmarkers.size();
        Index_ NC = ref.ncol();
        positions.reserve(NC);                

        std::vector<Index_> samples_per_label(nlabels);
        for (Index_ c = 0; c < NC; ++c) {
            auto& pos = samples_per_label[curlab[c]];
            positions.push_back(pos);
            ++pos;
        }

        cur_ranked.resize(nlabels);
        for (decltype(nlabels) l = 0; l < nlabels; ++l) {
            cur_ranked[l].resize(samples_per_label[l]);
        }
    }

    if (!curinput.with_intersection) {
        // The universe is guaranteed to be sorted and unique, see its derivation
        // in internal::train_integrated() below. This means we can directly use it
        // for indexed extraction from a tatami::Matrix.
        tatami::VectorPtr<Index_> universe_ptr(tatami::VectorPtr<Index_>{}, &(output.universe));

        tatami::parallelize([&](int, Index_ start, Index_ len) {
            std::vector<Value_> buffer(output.universe.size());
            internal::RankedVector<Value_, Index_> tmp_ranked;
            tmp_ranked.reserve(output.universe.size());
            auto ext = tatami::consecutive_extractor<false>(&ref, false, start, len, universe_ptr); 

            for (Index_ c = start, end = start + len; c < end; ++c) {
                auto ptr = ext->fetch(buffer.data());

                tmp_ranked.clear();
                for (int i = 0, end = output.universe.size(); i < end; ++i, ++ptr) {
                    tmp_ranked.emplace_back(*ptr, i);
                }
                std::sort(tmp_ranked.begin(), tmp_ranked.end());

                auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
                simplify_ranks(tmp_ranked, final_ranked);
            }
        }, ref.ncol(), options.num_threads);

    } else {
        output.check_availability[ref_i] = 1;

        // Need to remap from indices on the test matrix to those in the current reference matrix
        // so that we can form an appropriate vector for indexed tatami extraction. 
        const auto& intersection = (curinput.user_intersection == NULL ? curinput.auto_intersection : *(curinput.user_intersection));
        std::unordered_map<Index_, Index_> intersection_map;
        intersection_map.reserve(intersection.size());
        for (const auto& in : intersection) {
            intersection_map[in.first] = in.second;
        }

        std::vector<std::pair<Index_, Index_> > intersection_in_universe;
        intersection_in_universe.reserve(output.universe.size());
        auto& cur_available = output.available[ref_i];
        cur_available.reserve(output.universe.size());

        for (Index_ i = 0, end = output.universe.size(); i < end; ++i) {
            auto it = intersection_map.find(output.universe[i]);
            if (it != intersection_map.end()) {
                intersection_in_universe.emplace_back(it->second, i); // using 'i' as we want to work with indices into 'universe', not the indices of the universe itself.
                cur_available.insert(i);
            }
        }
        std::sort(intersection_in_universe.begin(), intersection_in_universe.end());

        std::vector<Index_> to_extract; 
        to_extract.reserve(intersection_in_universe.size());
        for (const auto& p : intersection_in_universe) {
            to_extract.push_back(p.first);
        }
        tatami::VectorPtr<Index_> to_extract_ptr(tatami::VectorPtr<Index_>{}, &to_extract);

        tatami::parallelize([&](int, Index_ start, Index_ len) {
            std::vector<Value_> buffer(to_extract.size());
            internal::RankedVector<Value_, Index_> tmp_ranked;
            tmp_ranked.reserve(to_extract.size());
            auto ext = tatami::consecutive_extractor<false>(&ref, false, start, len, to_extract_ptr);

            for (Index_ c = start, end = start + len; c < end; ++c) {
                auto ptr = ext->fetch(buffer.data());

                tmp_ranked.clear();
                for (const auto& p : intersection_in_universe) {
                    tmp_ranked.emplace_back(*ptr, p.second); // remember, 'p.second' corresponds to indices into the universe.
                    ++ptr;
                }
                std::sort(tmp_ranked.begin(), tmp_ranked.end());

                auto& final_ranked = cur_ranked[curlab[c]][positions[c]];
                simplify_ranks(tmp_ranked, final_ranked);
            }
        }, ref.ncol(), options.num_threads);
    }
}

template<typename Value_, typename Index_, typename Inputs_>
TrainedIntegrated<Index_> train_integrated(Inputs_& inputs, const TrainIntegratedOptions& options) {
    TrainedIntegrated<Index_> output;
    auto nrefs = inputs.size();
    output.check_availability.resize(nrefs);
    output.available.resize(nrefs);
    output.markers.resize(nrefs);
    output.ranked.resize(nrefs);

    // Checking that the number of genes in the test dataset are consistent.
    output.test_nrow = -1;
    for (const auto& in : inputs) {
        if (output.test_nrow == static_cast<Index_>(-1)) {
            output.test_nrow = in.test_nrow;
        } else if (in.test_nrow != static_cast<Index_>(-1) && in.test_nrow != output.test_nrow) {
            throw std::runtime_error("inconsistent number of rows in the test dataset across entries of 'inputs'");
        }
    }

    // Identify the union of all marker genes.
    std::unordered_map<Index_, Index_> remap_to_universe;
    std::unordered_set<Index_> subset_tmp;
    for (const auto& in : inputs) {
        for (const auto& mrk : in.markers) {
            subset_tmp.insert(mrk.begin(), mrk.end());
        }
    }

    output.universe.insert(output.universe.end(), subset_tmp.begin(), subset_tmp.end());
    std::sort(output.universe.begin(), output.universe.end());
    remap_to_universe.reserve(output.universe.size());
    for (Index_ i = 0, end = output.universe.size(); i < end; ++i) {
        remap_to_universe[output.universe[i]] = i;
    }

    for (decltype(nrefs) r = 0; r < nrefs; ++r) {
        train_integrated_per_reference<Value_>(r, inputs[r], output, remap_to_universe, options);
    }

    return output;
}

}
/**
 * @endcond
 */

/**
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 *
 * @param inputs Vector of references, typically constructed with `prepare_integrated_input()` or `prepare_integrated_input_intersect()`.
 * @param options Further options.
 *
 * @return A pre-built classifier that integrates multiple references, for use in `classify_integrated()`.
 */
template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(const std::vector<TrainIntegratedInput<Value_, Index_, Label_> >& inputs, const TrainIntegratedOptions& options) {
    return internal::train_integrated<Value_, Index_>(inputs, options);
}

/**
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 *
 * @param inputs Vector of references, typically constructed with `prepare_integrated_input()` or `prepare_integrated_input_intersect()`.
 * @param options Further options.
 *
 * @return A pre-built classifier that integrates multiple references, for use in `classify_integrated()`.
 */
template<typename Value_, typename Index_, typename Label_>
TrainedIntegrated<Index_> train_integrated(std::vector<TrainIntegratedInput<Value_, Index_, Label_> >&& inputs, const TrainIntegratedOptions& options) {
    return internal::train_integrated<Value_, Index_>(inputs, options);
}

}

#endif
