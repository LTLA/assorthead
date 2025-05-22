#ifndef SINGLEPP_TRAIN_SINGLE_HPP
#define SINGLEPP_TRAIN_SINGLE_HPP

#include "defs.hpp"

#include "knncolle/knncolle.hpp"
#include "tatami/tatami.hpp"

#include "build_indices.hpp"
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
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Matrix_ Class of the input data for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 */
template<typename Index_ = DefaultIndex, typename Float_ = DefaultFloat, class Matrix_ = knncolle::Matrix<Index_, Float_> >
struct TrainSingleOptions {
    /**
     * Number of top markers to use from each pairwise comparison between labels.
     * Larger values improve the stability of the correlations at the cost of increasing noise and computational work.
     *
     * Setting this to a negative value will instruct `train_single()` to use all supplied markers.
     * This is useful in situations where the supplied markers have already been curated.
     */
    int top = -1;

    /**
     * Algorithm for the nearest-neighbor search.
     * This allows us to skip the explicit calculation of correlations between each test cell and every reference sample.
     * If NULL, this defaults to an exact search based on `knncolle::VptreeBuilder`.
     */
    std::shared_ptr<knncolle::Builder<Index_, Float_, Float_, Matrix_> > trainer;

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

template<typename Value_, typename Index_, typename Label_, typename Float_, class Matrix_>
std::vector<PerLabelReference<Index_, Float_> > build_references(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    const std::vector<Index_>& subset,
    const TrainSingleOptions<Index_, Float_, Matrix_>& options) 
{
    auto builder = options.trainer;
    if (!builder) {
        builder.reset(new knncolle::VptreeBuilder<Index_, Float_, Float_, Matrix_>(std::make_shared<knncolle::EuclideanDistance<Float_, Float_> >()));
    }
    return build_indices(ref, labels, subset, *builder, options.num_threads);
}

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
        std::vector<internal::PerLabelReference<Index_, Float_> > references) :
        my_test_nrow(test_nrow),
        my_markers(std::move(markers)),
        my_subset(std::move(subset)),
        my_references(std::move(references)) 
    {}
    /**
     * @endcond
     */

private:
    Index_ my_test_nrow;
    Markers<Index_> my_markers;
    std::vector<Index_> my_subset;
    std::vector<internal::PerLabelReference<Index_, Float_> > my_references;

public:
    /**
     * @return Number of rows that should be present in the test dataset.
     */
    Index_ get_test_nrow() const {
        return my_test_nrow;
    }

    /**
     * @return A vector of vectors of vectors of ranked marker genes to be used in the classification.
     * In the innermost vectors, each value is an index into the subset vector (see `get_subset()`),
     * e.g., `get_subset()[get_markers()[2][1].front()]` is the row index of the first marker of the third label over the first label.
     * The set of marker genes is a subset of the input `markers` used in `train_single()`.
     */
    const Markers<Index_>& get_markers() const {
        return my_markers;
    }

    /**
     * @return The subset of genes in the test/reference datasets that were used in the classification.
     * Each value is a row index into either matrix.
     */
    const std::vector<Index_>& get_subset() const {
        return my_subset;
    }

    /**
     * @return Number of labels in this reference.
     */
    std::size_t num_labels() const {
        return my_references.size();
    }

    /**
     * @return Number of profiles in this reference.
     */
    std::size_t num_profiles() const {
        std::size_t n = 0;
        for (const auto& ref : my_references) {
            n += ref.ranked.size();
        }
        return n;
    }

    /**
     * @cond
     */
    const auto& get_references() const {
        return my_references;
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
 * If the test dataset has different genes, consider using `train_single_intersect()` instead.
 * 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Matrix_ Class of the input data for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
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
template<typename Value_, typename Index_, typename Label_, typename Float_, class Matrix_>
TrainedSingle<Index_, Float_> train_single(
    const tatami::Matrix<Value_, Index_>& ref,
    const Label_* labels,
    Markers<Index_> markers,
    const TrainSingleOptions<Index_, Float_, Matrix_>& options)
{
    auto subset = internal::subset_to_markers(markers, options.top);
    auto subref = internal::build_references(ref, labels, subset, options);
    Index_ test_nrow = ref.nrow(); // remember, test and ref are assumed to have the same features.
    return TrainedSingle<Index_, Float_>(test_nrow, std::move(markers), std::move(subset), std::move(subref));
}

/**
 * @brief Classifier built from an intersection of genes.
 *
 * Instances of this class should not be directly constructed, but instead returned by calling `train_single_intersect()`.
 * This uses the intersection of genes between the test dataset and those of the reference dataset.
 * Each instance can be used in `classify_single_intersect()` with a test dataset that has the specified genes.
 * 
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Index_, typename Float_>
class TrainedSingleIntersect {
public:
    /**
     * @cond
     */
    TrainedSingleIntersect(
        Index_ test_nrow,
        Markers<Index_> markers,
        std::vector<Index_> test_subset,
        std::vector<Index_> ref_subset,
        std::vector<internal::PerLabelReference<Index_, Float_> > references) :
        my_test_nrow(test_nrow),
        my_markers(std::move(markers)),
        my_test_subset(std::move(test_subset)),
        my_ref_subset(std::move(ref_subset)),
        my_references(std::move(references)) 
    {}
    /**
     * @endcond
     */

private:
    Index_ my_test_nrow;
    Markers<Index_> my_markers;
    std::vector<Index_> my_test_subset;
    std::vector<Index_> my_ref_subset;
    std::vector<internal::PerLabelReference<Index_, Float_> > my_references;

public:
    /**
     * @return Number of rows that should be present in the test dataset.
     */
    Index_ get_test_nrow() const {
        return my_test_nrow;
    }

    /**
     * @return A vector of vectors of ranked marker genes to be used in the classification.
     * In the innermost vectors, each value is an index into the subset vectors (see `get_test_subset()` and `get_ref_subset()`).
     * e.g., `get_test_subset()[get_markers()[2][1].front()]` is the test matrix row index of the first marker of the third label over the first label.
     * The set of marker genes is a subset of those in the input `markers` in `train_single_intersect()`.
     */
    const Markers<Index_>& get_markers() const {
        return my_markers;
    }

    /**
     * @return Subset of genes in the intersection for the test dataset.
     * These are unique indices into the `test_id` array supplied to `train_single_intersect()`, and can be assumed to represent row indices into the test matrix.
     * This has the same length as the subset vector returned by `get_ref_subset()`, where corresponding entries refer to the same genes in the respective datasets.
     */
    const std::vector<Index_>& get_test_subset() const {
        return my_test_subset;
    }

    /**
     * @return Subset of genes in the intersection for the test dataset.
     * These are unique indices into the `ref_id` matrix supplied to `train_single_intersect()`, and can be assumed to represent row indices into the reference matrix.
     * This has the same length as the subset vector returned by `get_test_subset()`, where corresponding entries refer to the same genes in the respective datasets.
     */
    const std::vector<Index_>& get_ref_subset() const {
        return my_ref_subset;
    }

    /**
     * @return Number of labels in this reference.
     */
    std::size_t num_labels() const {
        return my_references.size();
    }

    /**
     * @return Number of profiles in this reference.
     */
    std::size_t num_profiles() const {
        std::size_t n = 0;
        for (const auto& ref : my_references) {
            n += ref.ranked.size();
        }
        return n;
    }

    /**
     * @cond
     */
    const auto& get_references() const {
        return my_references;
    }
    /**
     * @endcond
     */
};

/**
 * Variant of `train_single()` that uses a pre-computed intersection of genes between the reference dataset and an as-yet-unspecified test dataset.
 * Most users will prefer to use the other `train_single_intersect()` overload that accepts `test_id` and `ref_id` and computes the intersection automatically.
 *
 * The classifier returned by this function should only be used in `classify_single_intersect()` with a test dataset that is compatible with the mappings in `intersection`.
 * That is, the gene in the `intersection[i].first`-th row of the test dataset should correspond to the `intersection[i].second`-th row of the reference dataset.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Matrix_ Class of the input data for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
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
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single_intersect()`. 
 */
template<typename Index_, typename Value_, typename Label_, typename Float_>
TrainedSingleIntersect<Index_, Float_> train_single_intersect(
    Index_ test_nrow,
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels,
    Markers<Index_> markers,
    const TrainSingleOptions<Index_, Float_>& options)
{
    auto pairs = internal::subset_to_markers(intersection, markers, options.top);
    auto subref = internal::build_references(ref, labels, pairs.second, options);
    return TrainedSingleIntersect<Index_, Float_>(test_nrow, std::move(markers), std::move(pairs.first), std::move(pairs.second), std::move(subref));
}

/**
 * @cond
 */
// For back-compatibility only.
template<typename Index_, typename Value_, typename Label_, typename Float_, class Matrix_>
TrainedSingleIntersect<Index_, Float_> train_single_intersect(
    const Intersection<Index_>& intersection,
    const tatami::Matrix<Value_, Index_>& ref, 
    const Label_* labels,
    Markers<Index_> markers,
    const TrainSingleOptions<Index_, Float_, Matrix_>& options)
{
    return train_single_intersect<Index_, Value_, Label_, Float_>(-1, intersection, ref, labels, std::move(markers), options);
}
/**
 * @endcond
 */

/**
 * Variant of `train_single()` that uses the intersection of genes between the reference dataset and a (future) test dataset.
 * This is useful when the genes are not in the same order and number across the test and reference datasets.
 *
 * The classifier returned by this function should only be used in `classify_single_intersect()` with a test dataset
 * that has `test_nrow` rows with the same order and identity of genes as in `test_id`.
 *
 * @tparam Index_ Integer type for the row/column indices of the matrix.
 * @tparam Id_ Type of the gene identifier for each row, typically integer or string.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Matrix_ Class of the input data for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
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
 * @param options Further options.
 *
 * @return A pre-built classifier that can be used in `classify_single_intersect()`.
 */
template<typename Index_, typename Id_, typename Value_, typename Label_, typename Float_, class Matrix_>
TrainedSingleIntersect<Index_, Float_> train_single_intersect(
    Index_ test_nrow,
    const Id_* test_id, 
    const tatami::Matrix<Value_, Index_>& ref, 
    const Id_* ref_id, 
    const Label_* labels,
    Markers<Index_> markers,
    const TrainSingleOptions<Index_, Float_, Matrix_>& options)
{
    auto intersection = intersect_genes(test_nrow, test_id, ref.nrow(), ref_id);
    return train_single_intersect(test_nrow, intersection, ref, labels, std::move(markers), options);
}

}

#endif
