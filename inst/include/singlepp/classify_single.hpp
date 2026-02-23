#ifndef SINGLEPP_CLASSIFY_SINGLE_HPP
#define SINGLEPP_CLASSIFY_SINGLE_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "annotate_cells_single.hpp"
#include "train_single.hpp"

#include <vector> 
#include <cstddef>
#include <stdexcept>

/**
 * @file classify_single.hpp
 * @brief Classify cells in a test dataset based a single reference.
 */

namespace singlepp {

/**
 * @brief Options for `classify_single()` and friends.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Float_ = DefaultFloat>
struct ClassifySingleOptions {
    /**
     * Quantile to use to define a per-label score for each test cell,
     * by applying it to the distribution of correlations between the test cell and that label's reference profiles.
     * Values closer to 0.5 focus on the behavior of the majority of a label's reference profiles.
     * Smaller values will be more sensitive to the presence of a subset of profiles that are more similar to the test cell,
     * which can be useful when the reference profiles themselves are heterogeneous.
     */
    Float_ quantile = 0.8;

    /**
     * Score threshold to use to select the top-scoring subset of labels during fine-tuning.
     * Larger values increase the chance of recovering the correct label at the cost of computational time.
     *
     * This threshold should not be set to to a value that is too large.
     * Otherwise, the first fine-tuning iteration would just contain all labels and there would be no reduction of the marker space.
     */
    Float_ fine_tune_threshold = 0.05;

    /**
     * Whether to perform fine-tuning.
     * This can be disabled to improve speed at the cost of accuracy.
     */
    bool fine_tune = true;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @brief Output buffers for `classify_single()`.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Label_ = DefaultLabel, typename Float_ = DefaultFloat>
struct ClassifySingleBuffers {
    /** 
     * Pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the index of the assigned label for each cell.
     */
    Label_* best;

    /** 
     * Vector of length equal to the number of labels.
     * Each entry contains a pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the (non-fine-tuned) score for each label for each cell.
     * Any pointer may be `NULL` in which case the scores for that label will not be reported.
     */
    std::vector<Float_*> scores;

    /**
     * Pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the difference between the highest and second-highest scores, possibly after fine-tuning.
     * This may also be `NULL` in which case the deltas are not reported.
     */
    Float_* delta;
};

/**
 * @brief Implements the [**SingleR**](https://bioconductor.org/packages/SingleR) algorithm for automated annotation of single-cell RNA-seq data.
 *
 * For each cell, we compute the Spearman rank correlation between that cell and the reference expression profiles.
 * This is done using only the subset of genes that are label-specific markers,
 * most typically the top genes from pairwise comparisons between each label's expression profiles (see `choose_classic_markers()` for an example).
 * For each label, we take the correlations involving that label's reference profiles and convert it into a score.
 * The label with the highest score is used as an initial label for that cell.
 *
 * Next, we apply fine-tuning iterations to improve the label accuracy for each cell by refining the feature space.
 * We find the subset of labels with scores that are "close enough" to the maximum score according to some threshold.
 * We recompute the scores using only the markers for this subset of labels, and we repeat the process until only one label is left or the subset of labels no longer changes.
 * At the end of the iterations, the label with the highest score (or the only label, if just one is left) is used as the label for the cell.
 * This process aims to remove noise by eliminating irrelevant genes when attempting to distinguish closely related labels.
 * 
 * Each label's score is defined as a user-specified quantile of the distribution of correlations across all reference profiles assigned to that label.
 * Larger quantiles focus on similarity between the test cell and the closest profiles for a label, which is useful for broad labels with heterogeneous profiles.
 * Smaller quantiles require the test cell to be similar to the majority of profiles for a label.
 * The use of a quantile ensures that the score adjusts to the number of reference profiles per label;
 * otherwise, just using the "top X correlations" would implicitly favor labels with more reference profiles.
 *
 * The choice of Spearman's correlation provides some robustness against batch effects when comparing reference and test datasets.
 * Only the relative expression _within_ each cell needs to be comparable, not their relative expression across cells.
 * As a result, it does not matter whether raw counts are supplied or log-transformed expression values, as the latter is a monotonic transformation of the latter (within each cell).
 * The algorithm is also robust to differences in technologies between reference and test profiles, though it is preferable to have like-for-like comparisons. 
 *
 * @see
 * Aran D et al. (2019). 
 * Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage.
 * _Nat. Immunol._ 20, 163-172
 * 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Float_ Floating-point type for the correlations and scores.
 * @tparam Label_ Integer type for the reference labels.
 *
 * @param test Expression matrix of the test dataset, where rows are genes and columns are cells.
 * This should have the same order and identity of genes as the reference matrix used to create `trained`.
 * @param trained Classifier returned by `train_single()`.
 * @param[out] buffers Buffers in which to store the classification output.
 * Each non-`NULL` pointer should refer to an array of length equal to the number of columns in `test`.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Float_, typename Label_>
void classify_single(
    const tatami::Matrix<Value_, Index_>& test, 
    const TrainedSingle<Index_, Float_>& trained,
    const ClassifySingleBuffers<Label_, Float_>& buffers,
    const ClassifySingleOptions<Float_>& options) 
{
    if (trained.test_nrow() != test.nrow()) {
        throw std::runtime_error("number of rows in 'test' is not the same as that used to build 'trained'");
    }
    annotate_cells_single(
        test, 
        trained,
        options.quantile, 
        options.fine_tune, 
        options.fine_tune_threshold, 
        buffers.best, 
        buffers.scores, 
        buffers.delta,
        options.num_threads
    );
}

/**
 * @brief Results of `classify_single()` and `classify_single()`.
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Label_ = DefaultLabel, typename Float_ = DefaultFloat>
struct ClassifySingleResults {
    /**
     * @cond
     */
    ClassifySingleResults(std::size_t num_cells, std::size_t num_labels) : best(num_cells), delta(num_cells) {
        scores.reserve(num_labels);
        for (decltype(num_labels) l = 0; l < num_labels; ++l) {
            scores.emplace_back(num_cells);
        }
    }
    /**
     * @endcond
     */

    /** 
     * Vector of length equal to the number of cells in the test dataset,
     * containing the index of the assigned label for each cell.
     */
    std::vector<Label_> best;

    /**
     * Vector of length equal to the number of labels,
     * containing vectors of length equal to the number of cells in the test dataset.
     * Each vector corresponds to a label and contains the (non-fine-tuned) score for each cell.
     */
    std::vector<std::vector<Float_> > scores;

    /** 
     * Vector of length equal to the number of cells in the test dataset.
     * This contains the difference between the highest and second-highest scores for each cell, possibly after fine-tuning.
     */
    std::vector<Float_> delta;
};

/**
 * Overload of `classify_single()` that allocates space for the output statistics.
 *
 * @tparam Label_ Integer type for the reference labels.
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Float_ Floating-point type for the correlations and scores.
 *
 * @param test Expression matrix of the test dataset, where rows are genes and columns are cells.
 * This should have the same order and identity of genes as the reference matrix used to create `trained`.
 * @param trained Classifier returned by `train_single()`.
 * @param options Further options.
 *
 * @return Results of the classification for each cell in the test dataset.
 */
template<typename Label_ = DefaultLabel, typename Value_, typename Index_, typename Float_>
ClassifySingleResults<Label_, Float_> classify_single(
    const tatami::Matrix<Value_, Index_>& test,
    const TrainedSingle<Index_, Float_>& trained,
    const ClassifySingleOptions<Float_>& options) 
{
    ClassifySingleResults<Label_, Float_> output(test.ncol(), trained.num_labels());

    ClassifySingleBuffers<Label_, Float_> buffers;
    buffers.best = output.best.data();
    buffers.delta = output.delta.data();
    buffers.scores.reserve(output.scores.size());
    for (auto& s : output.scores) {
        buffers.scores.emplace_back(s.data());
    }

    classify_single(test, trained, buffers, options);
    return output;
}

}

#endif
