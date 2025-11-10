#ifndef GSDECON_COMPUTE_HPP
#define GSDECON_COMPUTE_HPP

#include <algorithm>
#include <numeric>

#include "tatami/tatami.hpp"
#include "irlba/irlba.hpp"
#include "scran_pca/scran_pca.hpp"
#include "sanisizer/sanisizer.hpp"

#include "Options.hpp"
#include "Results.hpp"
#include "utils.hpp"

/**
 * @file compute.hpp
 * @brief Compute per-cell scores for a gene set.
 */

namespace gsdecon {

/**
 * Given an input matrix containing log-expression values for genes in a set of interest, 
 * per-cell scores are defined as the column means of the low-rank approximation of that matrix.
 * The assumption here is that the primary activity of the gene set can be quantified by the largest component of variance amongst its genes.
 * (If this was not the case, one could argue that this gene set is not well-suited to capture the biology attributed to it.)
 * In effect, the rotation vector defines weights for all genes in the set, focusing on genes that contribute to the primary activity.
 *
 * By default, we use a rank-1 approximation (see `Options::rank`).
 * The reported weight for each gene (in `Results::weights`) is simply the absolute value of the associated rotation vector from the PCA.
 * Increasing the rank of the approximation may capture more biological signal but also increases noise in the per-cell scores.
 * If higher ranks are used, each gene's weight is instead defined as the root mean square of that gene's values across all rotation vectors.
 *
 * @tparam Value_ Floating-point type of the data.
 * @tparam Index_ Integer type of the indices.
 * @tparam Float_ Floating-point type of the output.
 *
 * @param[in] matrix A matrix where columns correspond to cells and rows correspond to genes.
 * Entries are typically log-expression values. 
 * @param options Further options. 
 * @param[out] output Collection of buffers in which to store the scores and weights.
 */
template<typename Value_, typename Index_, typename Float_>
void compute(const tatami::Matrix<Value_, Index_>& matrix, const Options& options, const Buffers<Float_>& output) {
    if (check_edge_cases(matrix, options.rank, output)) {
        return;
    }

    scran_pca::SimplePcaOptions sopt;
    sopt.number = options.rank;
    sopt.scale = options.scale;
    sopt.realize_matrix = options.realize_matrix;
    sopt.num_threads = options.num_threads;
    sopt.irlba_options = options.irlba_options;
    const auto res = scran_pca::simple_pca(matrix, sopt);

    const Float_ shift = std::accumulate(res.center.begin(), res.center.end(), static_cast<Float_>(0)) / matrix.nrow();
    std::fill_n(output.scores, matrix.ncol(), shift);
    process_output(res.rotation, res.components, options.scale, res.scale, output);
}

/**
 * Overload of `compute()` that allocates memory for the results.
 *
 * @tparam Float_ Floating-point type of the output.
 * @tparam Value_ Floating-point type of the data.
 * @tparam Index_ Integer type of the indices.
 *
 * @param[in] matrix A matrix where columns correspond to cells and rows correspond to genes.
 * Entries are typically log-expression values. 
 * @param options Further options. 
 *
 * @return Results of the gene set score calculation.
 */
template<typename Float_ = double, typename Value_, typename Index_>
Results<Float_> compute(const tatami::Matrix<Value_, Index_>& matrix, const Options& options) {
    Results<Float_> output;
    sanisizer::resize(output.weights, matrix.nrow()
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    sanisizer::resize(output.scores, matrix.ncol()
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );

    Buffers<Float_> buffers;
    buffers.weights = output.weights.data();
    buffers.scores = output.scores.data();

    compute(matrix, options, buffers);
    return output;
}

}

#endif
