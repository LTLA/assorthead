#ifndef SINGLEPP_TRAIN_SINGLE_HPP
#define SINGLEPP_TRAIN_SINGLE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "build_reference.hpp"
#include "subset_to_markers.hpp"

#include <vector>
#include <memory>
#include <cstddef>

/**
 * @file train_single.hpp
 * @brief Train a classifier from a single reference.
 */

namespace singlepp {

/**
 * @brief Options for `train_single()` and friends.
 */
struct TrainSingleOptions {
    /**
     * Number of top markers to use from each pairwise comparison between labels.
     * Larger values improve the stability of the correlations at the cost of increasing noise and computational work.
     *
     * When the test and reference datasets do not have the same features, the specified number of top markers is taken from the intersection of features.
     * This avoids that some markers will be selected even if not all genes in `markers` are present in the test dataset.
     *
     * Setting this to a negative value will instruct `train_single()` to use all supplied markers.
     * This is useful in situations where the supplied markers have already been curated.
     */
    int top = -1;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
template<typename Index_, typename Float_>
std::size_t get_num_labels_from_built(const BuiltReference<Index_, Float_>& built) {
    if (built.sparse.has_value()) {
        return built.sparse->size();
    } else {
        return built.dense->size();
    }
}

template<typename Index_, typename Float_>
std::size_t get_num_profiles_from_built(const BuiltReference<Index_, Float_>& built) {
    std::size_t n = 0;
    if (built.sparse.has_value()) {
        for (const auto& ref : *(built.sparse)) {
            n += get_num_samples(ref);
        }
    } else {
        for (const auto& ref : *(built->dense)) {
            n += get_num_samples(ref);
        }
    }
    return n;
}
/**
 * @endcond
 */

/**
 * @brief Classifier trained from a single reference.
 *
 * Instances of this class should not be directly constructed, but instead returned by calling `train_single()` on a reference dataset.
 * Each instance can be used in `classify_single()` with a test dataset that has the same number and order of genes as the reference dataset. 
 * 
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Index_, typename Float_>
class TrainedSingle {
public:
    /**
     * @cond
     */
    TrainedSingle(
        Index_ test_nrow,
        Markers<Index_> markers,
        std::vector<Index_> subset,
        BuiltReference<Index_, Float_> built
    ) : 
        my_test_nrow(test_nrow),
        my_markers(std::move(markers)),
        my_subset(std::move(subset)),
        my_built(std::move(built)) 
    {}
    /**
     * @endcond
     */

private:
    Index_ my_test_nrow;
    Markers<Index_> my_markers;
    std::vector<Index_> my_subset;
    BuiltReference<Index_, Float_> my_built;

public:
    /**
     * @return Number of rows that should be present in the test dataset.
     */
    Index_ test_nrow() const {
        return my_test_nrow;
    }

    /**
     * @return A vector of vectors of vectors of ranked marker genes to be used in the classification.
     * In the innermost vectors, each value is an index into the subset vector (see `subset()`),
     * e.g., `subset()[markers()[2][1].front()]` is the row index of the first marker of the third label over the first label.
     * The set of marker genes is a subset of the input `markers` used in `train_single()`. 
     */
    const Markers<Index_>& markers() const {
        return my_markers;
    }

    /**
     * @return The subset of genes in the test dataset that were used in the classification.
     * Each value is a row index into the test matrix.
     * Values are sorted and unique.
     */
    const std::vector<Index_>& subset() const {
        return my_subset;
    }

    /**
     * @return Number of labels in this reference.
     */
    std::size_t num_labels() const {
        return get_num_labels_from_built(my_built);
    }

    /**
     * @return Number of profiles in this reference.
     */
    std::size_t num_profiles() const {
        return get_num_profiles_from_built(my_built);
    }

    /**
     * @cond
     */
    const auto& built() const {
        return my_built;
    }
    /**
     * @endcond
     */
};

/**
 * Prepare a single labelled reference dataset for use in `classify_single()`.
 * This involves pre-ranking the markers based on their expression in each reference profile,
 * so that the Spearman correlations can be computed without repeated sorting. 
 * We also construct neighbor search indices for rapid calculation of the classification score.
 *
 * The classifier returned by this function should only be used in `classify_single()` with a test dataset that has the same genes as the reference dataset.
 * If the test dataset has different genes, use the `train_single()` overloads that accept the intersection of genes between the test and reference dataset.
 * 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param ref Matrix for the reference expression profiles.
 * Rows are genes while columns are profiles.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each reference profile.
 * Labels should be integers in \f$[0, L)\f$ where \f$L\f$ is the total number of unique labels.
 * @param markers A vector of vectors of ranked marker genes for each pairwise comparison between labels, see `singlepp::Markers` for more details.
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single()` with a test dataset.
 */
template<typename Float_ = double, typename Value_, typename Index_, typename Label_>
TrainedSingle<Index_, Float_> train_single(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    Markers<Index_> markers,
    const TrainSingleOptions& options
) {
    auto subset = internal::subset_to_markers(markers, options.top);
    auto subref = build_reference<Float_>(ref, labels, subset, options.num_threads);
    const Index_ test_nrow = ref.nrow(); // remember, test and ref are assumed to have the same features.
    return TrainedSingle<Index_, Float_>(test_nrow, std::move(markers), std::move(subset), std::move(subref));
}

/**
 * Overload of `train_single()` that uses a pre-computed intersection of genes between the reference dataset and an as-yet-unspecified test dataset.
 * Most users will prefer to use the other `train_single()` overload that accepts `test_id` and `ref_id` and computes the intersection automatically.
 *
 * The classifier returned by this function should only be used in `classify_single()` with a test dataset that is compatible with the mappings in `intersection`.
 * That is, the gene in the `intersection[i].first`-th row of the test dataset should correspond to the `intersection[i].second`-th row of the reference dataset.
 *
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
 *
 * @param test_nrow Number of features in the test dataset.
 * @param intersection Vector defining the intersection of genes between the test and reference datasets.
 * Each pair corresponds to a gene where the first and second elements represent the row indices of that gene in the test and reference matrices, respectively.
 * The first element of each pair should be non-negative and less than `test_nrow`, while the second element should be non-negative and less than `ref->nrow()`.
 * See `intersect_genes()` for more details.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each reference profile.
 * Labels should be integers in \f$[0, L)\f$ where \f$L\f$ is the total number of unique labels.
 * @param markers A vector of vectors of ranked marker genes for each pairwise comparison between labels, see `singlepp::Markers` for more details.
 * @param[out] ref_subset Pointer to a vector in which to store the subset of rows of `ref` that contains the markers to be used for classification.
 * On output, the vector is filled with unique (but not necessarily sorted) row indices of length equal to `TrainedSingle::subset()`,
 * where each value contains the reference row that matches the test row indexed by the corresponding entry of `TrainedSingle::subset()`.
 * If `NULL`, the rows of the reference matrix are not returned.
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single()`. 
 */
template<typename Float_ = double, typename Index_, typename Value_, typename Label_>
TrainedSingle<Index_, Float_> train_single(
    Index_ test_nrow,
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels,
    Markers<Index_> markers,
    std::vector<Index_>* ref_subset,
    const TrainSingleOptions& options
) {
    auto pairs = internal::subset_to_markers(intersection, markers, options.top);
    auto subref = build_reference<Float_>(ref, labels, pairs.second, options.num_threads);
    if (ref_subset) {
        *ref_subset = std::move(pairs.second);
    }
    return TrainedSingle<Index_, Float_>(test_nrow, std::move(markers), std::move(pairs.first), std::move(subref));
}

/**
 * Overload of `train_single()` that uses the intersection of genes between the reference dataset and a (future) test dataset.
 * This is useful when the genes are not in the same order and number across the test and reference datasets.
 *
 * The classifier returned by this function should only be used in `classify_single()` with a test dataset
 * that has `test_nrow` rows with the same order and identity of genes as in `test_id`.
 *
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Id_ Type of the gene identifier for each row, typically integer or string.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
 *
 * @param test_nrow Number of rows (genes) in the test dataset.
 * @param[in] test_id Pointer to an array of length equal to `test_nrow`, containing a gene identifier for each row of the test dataset.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param ref An expression matrix for the reference expression profiles, where rows are genes and columns are cells.
 * This should have non-zero columns.
 * @param[in] ref_id Pointer to an array of length equal to the number of rows of `ref`, containing a gene identifier for each row of the reference dataset.
 * Identifiers should be comparable to those in `test_id`.
 * If any duplicate IDs are present, only the first occurrence is used.
 * @param[in] labels An array of length equal to the number of columns of `ref`, containing the label for each reference profile.
 * Labels should be integers in \f$[0, L)\f$ where \f$L\f$ is the total number of unique labels.
 * @param markers A vector of vectors of ranked marker genes for each pairwise comparison between labels, see `singlepp::Markers` for more details.
 * @param[out] ref_subset Pointer to a vector in which to store the subset of rows of `ref` that contains the markers to be used for classification.
 * On output, the vector is filled with unique (but not necessarily sorted) row indices of length equal to `TrainedSingle::subset()`,
 * where each value contains the reference row that matches the test row indexed by the corresponding entry of `TrainedSingle::subset()`.
 * If `NULL`, the rows of the reference matrix are not returned.
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single()`.
 */
template<typename Float_ = double, typename Index_, typename Id_, typename Value_, typename Label_>
TrainedSingle<Index_, Float_> train_single(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const Label_* labels,
    Markers<Index_> markers,
    std::vector<Index_>* ref_subset,
    const TrainSingleOptions& options
) {
    auto intersection = intersect_genes(test_nrow, test_id, ref.nrow(), ref_id);
    return train_single(test_nrow, intersection, ref, labels, std::move(markers), ref_subset, options);
}

}

#endif
