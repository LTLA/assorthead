#ifndef SINGLEPP_CLASSIFY_INTEGRATED_HPP
#define SINGLEPP_CLASSIFY_INTEGRATED_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "annotate_cells_integrated.hpp"
#include "train_integrated.hpp"

#include <vector>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

/**
 * @file classify_integrated.hpp
 * @brief Integrate classifications from multiple references.
 */

namespace singlepp {

/**
 * @brief Options for `classify_integrated()`.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename Float_ = DefaultFloat>
struct ClassifyIntegratedOptions {
    /**
     * Quantile to use to compute a per-reference score from the correlations.
     * This has the same interpretation as `ClassifySingleOptions::quantile`.
     */
    Float_ quantile = 0.8;

    /**
     * Score threshold to use to select the top-scoring subset of references during fine-tuning,
     * see `ClassifySingleOptions::fine_tune` for more details.
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
 * @tparam RefLabel_ Integer type for the label to represent each reference.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename RefLabel_ = DefaultRefLabel, typename Float_ = DefaultFloat>
struct ClassifyIntegratedBuffers {
    /** 
     * Pointer to an array of length equal to the number of test cells.
     * On output, this is filled with the index of the assigned label for each cell.
     */
    RefLabel_* best;

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
 * @brief Integrate classifications from multiple references.
 *
 * When multiple reference datasets are available, we would like to obtain a single prediction for each cell from all of those references.
 * This is somewhat tricky as different references tend to have inconsistent labels, e.g., different vocabularies and cell subtype resolutions, 
 * making it difficult to define sensible groups in a combined "super-reference".
 * Strong batch effects are also likely to exist between different references, complicating the choice of marker genes when comparing between labels of different references.
 *
 * To avoid these issues, we first perform classification within each individual reference using, e.g., `classify_single()`.
 * For each test cell, we collect all the marker genes for that cell's predicted label in each reference.
 * We pool all of these collected markers to obtain a common set of interesting genes.
 * Using this common set of genes, we compute the usual correlation-based score between the test cell's expression profile and its predicted label from each reference,
 * along with some fine-tuning iterations to improve resolution between similar labels.
 * The label with the highest score is considered the best representative across all references.
 *
 * This method is similar to the algorithm described in `classify_single()`,
 * except that we are choosing between the best labels from all references rather than between all labels from one reference.
 * The creation of a common gene set ensures that the correlations can be reasonably compared across references.
 * (Note that differences in the gene sets across references are tolerated by simply ignoring missing genes when computing the correlations.
 * This reduces the comparability of the scores as the actual genes used for each reference will vary; 
 * nonetheless, it is preferred to taking the intersection, which is liable to leave us with very few genes.)
 *
 * Our approach avoids any direct comparison between the expression profiles of different references,
 * allowing us to side-step the question of how to deal with the batch effects.
 * Similarly, we defer responsibility on solving the issue of label heterogeneity,
 * by just passing along the existing labels and leaving it to the user's interpretation.
 * 
 * @tparam Value_ Numeric type for the matrix values.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Label_ Integer type for the labels within each reference.
 * @tparam RefLabel_ Integer type for the label to represent each reference.
 * @tparam Float_ Floating-point type for the correlations and scores.
 * 
 * @param test Expression matrix of the test dataset, where rows are genes and columns are cells.
 * The identity of the rows should be consistent with the reference datasets used to construct `trained`,
 * see `prepare_integrated_input()` and `prepare_integrated_input_intersect()` for details.
 * @param[in] assigned Vector of pointers of length equal to the number of references.
 * Each pointer should point to an array of length equal to the number of columns in `test`,
 * containing the assigned label for each column in each reference.
 * @param trained The integrated classifier returned by `train_integrated()`.
 * @param[out] buffers Buffers in which to store the classification output.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Label_, typename RefLabel_, typename Float_>
void classify_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<const Label_*>& assigned,
    const TrainedIntegrated<Index_>& trained,
    ClassifyIntegratedBuffers<RefLabel_, Float_>& buffers,
    const ClassifyIntegratedOptions<Float_>& options)
{
    if (trained.test_nrow != static_cast<Index_>(-1) && trained.test_nrow != test.nrow()) {
        throw std::runtime_error("number of rows in 'test' is not the same as that used to build 'trained'");
    }
    internal::annotate_cells_integrated(
        test,
        trained,
        assigned,
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
 * @brief Results of `classify_integrated()`.
 * @tparam RefLabel_ Integer type for the label to represent each reference.
 * @tparam Float_ Floating-point type for the correlations and scores.
 */
template<typename RefLabel_ = DefaultRefLabel, typename Float_ = DefaultFloat>
struct ClassifyIntegratedResults {
    /**
     * @cond
     */
    ClassifyIntegratedResults(std::size_t ncells, std::size_t nrefs) : best(ncells), delta(ncells) {
        scores.reserve(nrefs);
        for (decltype(nrefs) r = 0; r < nrefs; ++r) {
            scores.emplace_back(ncells);
        }
    }
    /**
     * @endcond
     */

    /** 
     * Vector of length equal to the number of cells in the test dataset,
     * containing the index of the reference with the top-scoring label for each cell.
     */
    std::vector<RefLabel_> best;

    /**
     * Vector of length equal to the number of references,
     * containing vectors of length equal to the number of cells in the test dataset.
     * Each vector corresponds to a reference and contains the score for the best label in that reference for each cell.
     */
    std::vector<std::vector<Float_> > scores;

    /** 
     * Vector of length equal to the number of cells in the test dataset.
     * This contains the difference between the highest and second-highest scores for each cell.
     */
    std::vector<Float_> delta;
};

/**
 * Overload of `classify_integrated()` that allocates space for the results.
 *
 * @param test Expression matrix of the test dataset, where rows are genes and columns are cells.
 * The identity of the rows should be consistent with the reference datasets used to construct `trained`,
 * see `prepare_integrated_input()` and `prepare_integrated_input_intersect()` for details.
 * @param[in] assigned Vector of pointers of length equal to the number of references.
 * Each pointer should point to an array of length equal to the number of columns in `mat`,
 * containing the assigned label for each column in each reference.
 * @param trained A pre-built classifier produced by `train_integrated()`.
 * @param options Further options.
 *
 * @return Object containing the best reference and associated scores for each cell in `test`.
 */
template<typename RefLabel_ = DefaultRefLabel, typename Value_, typename Index_, typename Label_, typename Float_>
ClassifyIntegratedResults<RefLabel_, Float_> classify_integrated(
    const tatami::Matrix<Value_, Index_>& test,
    const std::vector<const Label_*>& assigned,
    const TrainedIntegrated<Index_>& trained,
    const ClassifyIntegratedOptions<Float_>& options)
{
    ClassifyIntegratedResults<RefLabel_, Float_> results(test.ncol(), trained.num_references());
    ClassifyIntegratedBuffers<RefLabel_, Float_> buffers;
    buffers.best = results.best.data();
    buffers.delta = results.delta.data();
    buffers.scores.reserve(results.scores.size());
    for (auto& s : results.scores) {
        buffers.scores.emplace_back(s.data());
    }
    classify_integrated(test, assigned, trained, buffers, options);
    return results;
}

}

#endif
