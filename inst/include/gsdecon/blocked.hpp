#ifndef GSDECON_BLOCKED_HPP
#define GSDECON_BLOCKED_HPP

#include <vector>

#include "Eigen/Dense"
#include "tatami/tatami.hpp"
#include "irlba/irlba.hpp"
#include "scran_pca/scran_pca.hpp"
#include "sanisizer/sanisizer.hpp"

#include "Options.hpp"
#include "Results.hpp"
#include "utils.hpp"

/**
 * @file blocked.hpp
 * @brief Compute per-cell scores with blocking.
 */

namespace gsdecon {

/**
 * Extension of the algorithm described in `compute()` to datasets containing multiple blocks (e.g., batches, samples).
 *
 * In the presence of strong block effects, naively running `compute()` would yield a first PC that is driven by uninteresting inter-block differences.
 * Here, we perform the PCA on the residuals after centering each block, ensuring that the first PC focuses on the interesting variation within each block.
 * Blocks can also be weighted so that they contribute equally to the rotation vector, regardless of the number of cells.
 * The score for each cell is obtained by adding the block-specific centers to the low-rank approximation and computing the column means.
 *
 * Note that the purpose of the blocking is to ensure that inter-block differences do not drive the first few PCs, not to remove the block effects themselves.
 * Using residuals for batch correction requires strong assumptions such as identical block composition and consistent shifts across subpopulations; we do not attempt make that claim.
 * The caller is instead responsible for ensuring that the block structure is still considered in any further analysis of the computed scores.
 *
 * @tparam Value_ Floating-point type of the data.
 * @tparam Index_ Integer type of the indices.
 * @tparam Block_ Integer type of the block assignments.
 * @tparam Float_ Floating-point type of the output.
 *
 * @param[in] matrix A matrix where columns correspond to cells and rows correspond to genes.
 * Entries are typically log-expression values. 
 * @param[in] block Pointer to an array of length equal to the number of columns in `matrix`.
 * This should contain the blocking factor as 0-based block assignments 
 * (i.e., for \f$N\f$ blocks, block identities should run from 0 to \f$N-1\f$ with at least one entry for each block.)
 * @param options Further options. 
 * @param[out] output Collection of buffers in which to store the scores and weights.
 */
template<typename Value_, typename Index_, typename Block_, typename Float_>
void compute_blocked(const tatami::Matrix<Value_, Index_>& matrix, const Block_* const block, const Options& options, const Buffers<Float_>& output) {
    if (check_edge_cases(matrix, options.rank, output)) {
        return;
    }

    scran_pca::BlockedPcaOptions bopt;
    bopt.number = options.rank;
    bopt.scale = options.scale;
    bopt.block_weight_policy = options.block_weight_policy;
    bopt.variable_block_weight_parameters = options.variable_block_weight_parameters;
    bopt.realize_matrix = options.realize_matrix;
    bopt.num_threads = options.num_threads;
    bopt.irlba_options = options.irlba_options;
    const auto res = scran_pca::blocked_pca(matrix, block, bopt);

    // Here, we restore the block-specific centers.
    static_assert(!Eigen::MatrixXd::IsRowMajor); // just double-checking...
    const auto nfeat = res.center.cols();
    const auto nblocks = res.center.rows();
    auto block_means = sanisizer::create<std::vector<Float_> >(nblocks);

    for (I<decltype(nfeat)> f = 0; f < nfeat; ++f) {
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            block_means[b] += res.center.coeff(b, f);
        }
    }
    for (auto& b : block_means) {
        b /= nfeat;
    }

    const auto ncells = res.components.cols();
    for (I<decltype(ncells)> c = 0; c < ncells; ++c) {
        output.scores[c] = block_means[block[c]];
    }
    process_output(res.rotation, res.components, options.scale, res.scale, output);
}

/**
 * Overload of `compute_blocked()` that allocates memory for the results.
 *
 * @tparam Float_ Floating-point type of the output.
 * @tparam Value_ Floating-point type of the data.
 * @tparam Index_ Integer type of the indices.
 * @tparam Block_ Integer type of the block assignments.
 *
 * @param[in] matrix A matrix where columns correspond to cells and rows correspond to genes.
 * Entries are typically log-expression values. 
 * @param[in] block Pointer to an array of length equal to the number of columns in `matrix`.
 * This should contain the blocking factor as 0-based block assignments 
 * (i.e., for \f$N\f$ blocks, block identities should run from 0 to \f$N-1\f$ with at least one entry for each block.)
 * @param options Further options. 
 *
 * @return Results of the gene set score calculation.
 */
template<typename Float_ = double, typename Value_, typename Index_, typename Block_>
Results<Float_> compute_blocked(const tatami::Matrix<Value_, Index_>& matrix, const Block_* const block, const Options& options) {
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

    compute_blocked(matrix, block, options, buffers);
    return output;
}

}

#endif
