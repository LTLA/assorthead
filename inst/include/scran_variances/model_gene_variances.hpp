#ifndef SCRAN_MODEL_GENE_VARIANCES_H
#define SCRAN_MODEL_GENE_VARIANCES_H

#include <algorithm>
#include <vector>
#include <limits>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "scran_blocks/scran_blocks.hpp"
#include "sanisizer/sanisizer.hpp"

#include "fit_variance_trend.hpp"
#include "utils.hpp"

/**
 * @file model_gene_variances.hpp
 * @brief Model the per-gene variances. 
 */

namespace scran_variances {

/**
 * Policy for averaging statistics across blocks.
 *
 * - `MEAN`: weighted mean, where weights are computed using `scran_blocks::compute_weights()`.
 * - `QUANTILE`: quantile, defaulting to 50%, a.k.a., the median.
 * - `NONE`: do not report any inter-block average. 
 */
enum class BlockAveragePolicy : unsigned char { MEAN, QUANTILE, NONE };

/**
 * @brief Options for `model_gene_variances()` and friends.
 */
struct ModelGeneVariancesOptions {
    /**
     * Options for fitting the mean-variance trend.
     */
    FitVarianceTrendOptions fit_variance_trend_options;

    /**
     * Policy to use for averaging statistics across blocks.
     * Only relevant to `model_gene_variances_blocked()`.
     * Ignored for overloads accepting `ModelGeneVariancesBlockedBuffers` where all entries in `average` are `NULL`.
     */
    BlockAveragePolicy block_average_policy = BlockAveragePolicy::MEAN;

    /**
     * Policy for weighting the contribution from each block when computing the mean for each statistic.
     * Only relevant to `model_gene_variances_blocked()` when `ModelGeneVariancesOptions::average_policy = BlockAveragePolicy::MEAN`. 
     *
     * The default of `scran_blocks::WeightPolicy::VARIABLE` is to define equal weights for blocks once they reach a certain size (see `ModelGeneVariancesOptions::variable_block_weight_parameters`).
     * For smaller blocks, the weight is linearly proportional to its size to avoid outsized contributions from very small blocks.
     *
     * Other options include `scran_blocks::WeightPolicy::EQUAL`, where all blocks are equally weighted regardless of size;
     * and `scran_blocks::WeightPolicy::NONE`, where the contribution of each block is proportional to its size.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights, including the threshold at which blocks are considered to be large enough to have equal weight.
     * Only relevant to `model_gene_variances_blocked()` when `ModelGeneVariancesOptions::average_policy = BlockAveragePolicy::MEAN`.
     * and `ModelGeneVariancesOptions::block_weight_policy = scran_blocks::WeightPolicy::VARIABLE`.
     */
    scran_blocks::VariableWeightParameters variable_block_weight_parameters; 

    /**
     * @cond
     */
    // Back-compatibility only.
    bool compute_average = true;
    /**
     * @endcond
     */

    /**
     * Quantile to use as an "average" statistic across blocks.
     * Only relevant to `model_gene_variances_blocked()` when `ModelGeneVariancesOptions::average_policy = BlockAveragePolicy::QUANTILE`. 
     * Defaults to 0.5, a.k.a., the median.
     */
    double block_quantile = 0.5;

    /**
     * Number of threads to use for the variance calculations and trend fitting. 
     * The parallelization scheme is defined by `tatami::parallelize()`. 
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `model_gene_variances()` and friends.
 * @tparam Stat_ Floating-point type of the output statistics.
 *
 * In general, the pointers in this class should _not_ be set to `NULL`.
 * The only exception is for instances of this class that are used as `ModelGeneVariancesBlockedBuffers::average`,
 * where setting the pointer to `NULL` will omit calculation of the corresponding average statistic.
 */
template<typename Stat_>
struct ModelGeneVariancesBuffers {
    /**
     * Pointer to an array of length equal to the number of genes, to be filled with the mean log-expression for each gene.
     */
    Stat_* means;

    /**
     * Pointer to an array of length equal to the number of genes, containing the variance in the log-expression for each gene.
     */
    Stat_* variances;

    /**
     * Pointer to an array of length equal to the number of genes, containing the fitted value of the mean-variance trend for each gene.
     */
    Stat_* fitted;

    /**
     * Vector of length equal to the number of genes, containing the residuals of the mean-variance trend for each gene.
     */
    Stat_* residuals;
};

/**
 * @brief Results of `model_gene_variances()`. 
 * @tparam Stat_ Floating-point type of the output statistics.
 */
template<typename Stat_>
struct ModelGeneVariancesResults {
    /**
     * @cond
     */
    ModelGeneVariancesResults() = default;

    ModelGeneVariancesResults(const std::size_t ngenes) :
        means(sanisizer::cast<I<decltype(means.size())> >(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        variances(sanisizer::cast<I<decltype(variances.size())> >(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        fitted(sanisizer::cast<I<decltype(fitted.size())> >(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        residuals(sanisizer::cast<I<decltype(residuals.size())> >(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        )
    {}
    /**
     * @endcond
     */

    /**
     * Vector of length equal to the number of genes, containing the mean log-expression for each gene.
     */
    std::vector<Stat_> means;

    /**
     * Vector of length equal to the number of genes, containing the variance in the log-expression for each gene.
     */
    std::vector<Stat_> variances;

    /**
     * Vector of length equal to the number of genes, containing the fitted value of the mean-variance trend for each gene.
     */
    std::vector<Stat_> fitted;

    /**
     * Vector of length equal to the number of genes, containing the residuals of the mean-variance trend for each gene.
     */
    std::vector<Stat_> residuals;
};

/**
 * @brief Buffers for `model_gene_variances_blocked()`.
 * @tparam Stat_ Floating-point type of the output statistics.
 */
template<typename Stat_>
struct ModelGeneVariancesBlockedBuffers {
    /**
     * Vector of length equal to the number of blocks,
     * where each entry contains the buffers to store the variance modelling results for a single block.
     */
    std::vector<ModelGeneVariancesBuffers<Stat_> > per_block;

    /**
     * Buffers to store the average across blocks for all statistics in `per_block`.
     * Any of the pointers may be `NULL`, in which case the corresponding average is not computed.
     */
    ModelGeneVariancesBuffers<Stat_> average;
};

/**
 * @brief Results of `model_gene_variances_blocked()`.
 * @tparam Stat_ Floating-point type of the output statistics.
 */
template<typename Stat_>
struct ModelGeneVariancesBlockedResults {
    /**
     * @cond
     */
    ModelGeneVariancesBlockedResults() = default;

    ModelGeneVariancesBlockedResults(const std::size_t ngenes, const std::size_t nblocks, const bool do_average) :
        average(do_average ? ngenes : 0)
    {
        per_block.reserve(nblocks);
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            per_block.emplace_back(ngenes);
        }
    }
    /**
     * @endcond
     */

    /**
     * Vector of length equal to the number of blocks, where each entry contains the variance modelling results for a single block.
     */
    std::vector<ModelGeneVariancesResults<Stat_> > per_block;

    /**
     * Average across blocks for all statistics in `per_block`.
     * This is only populated if `ModelGeneVariancesOptions::average_policy` is not `BlockAveragePolicy::NONE`.
     */
    ModelGeneVariancesResults<Stat_> average;
};

/**
 * @cond
 */
namespace internal {

template<typename Value_, typename Index_, typename Stat_, typename Block_> 
void compute_variances_dense_row(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& buffers,
    const Block_* const block,
    const std::vector<Index_>& block_size,
    const int num_threads)
{
    const bool blocked = (block != NULL);
    const auto nblocks = block_size.size();
    const auto NR = mat.nrow(), NC = mat.ncol();

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto tmp_means = sanisizer::create<std::vector<Stat_> >(blocked ? nblocks : 0);
        auto tmp_vars = sanisizer::create<std::vector<Stat_> >(blocked ? nblocks : 0);

        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ext = tatami::consecutive_extractor<false>(mat, true, start, length);
        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto ptr = ext->fetch(buffer.data());

            if (blocked) {
                tatami_stats::grouped_variances::direct(
                    ptr,
                    NC,
                    block,
                    nblocks,
                    block_size.data(),
                    tmp_means.data(),
                    tmp_vars.data(),
                    false,
                    static_cast<Index_*>(NULL)
                );
                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    buffers[b].means[r] = tmp_means[b];
                    buffers[b].variances[r] = tmp_vars[b];
                }
            } else {
                const auto stat = tatami_stats::variances::direct(ptr, NC, false);
                buffers[0].means[r] = stat.first;
                buffers[0].variances[r] = stat.second;
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Stat_, typename Block_> 
void compute_variances_sparse_row(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& buffers,
    const Block_* const block,
    const std::vector<Index_>& block_size,
    const int num_threads)
{
    const bool blocked = (block != NULL);
    const auto nblocks = block_size.size();
    const auto NR = mat.nrow(), NC = mat.ncol();

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        auto tmp_means = sanisizer::create<std::vector<Stat_> >(nblocks);
        auto tmp_vars = sanisizer::create<std::vector<Stat_> >(nblocks);
        auto tmp_nzero = sanisizer::create<std::vector<Index_> >(nblocks);

        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        auto ext = tatami::consecutive_extractor<true>(mat, true, start, length, [&]{
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            return opt;
        }());

        for (Index_ r = start, end = start + length; r < end; ++r) {
            auto range = ext->fetch(vbuffer.data(), ibuffer.data());

            if (blocked) {
                tatami_stats::grouped_variances::direct(
                    range.value,
                    range.index,
                    range.number,
                    block,
                    nblocks,
                    block_size.data(),
                    tmp_means.data(),
                    tmp_vars.data(),
                    tmp_nzero.data(),
                    false,
                    static_cast<Index_*>(NULL)
                );
                for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
                    buffers[b].means[r] = tmp_means[b];
                    buffers[b].variances[r] = tmp_vars[b];
                }
            } else {
                const auto stat = tatami_stats::variances::direct(range.value, range.number, NC, false);
                buffers[0].means[r] = stat.first;
                buffers[0].variances[r] = stat.second;
            }
        }
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Stat_, typename Block_> 
void compute_variances_dense_column(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& buffers,
    const Block_* const block,
    const std::vector<Index_>& block_size,
    const int num_threads)
{
    const bool blocked = (block != NULL);
    const auto nblocks = block_size.size();
    const auto NR = mat.nrow(), NC = mat.ncol();

    tatami::parallelize([&](const int thread, const Index_ start, const Index_ length) -> void {
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        auto ext = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), NC, start, length);

        auto get_var = [&](Index_ b) -> Stat_* { return buffers[b].variances; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_var)> local_vars(thread, nblocks, start, length, std::move(get_var));
        auto get_mean = [&](Index_ b) -> Stat_* { return buffers[b].means; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_mean)> local_means(thread, nblocks, start, length, std::move(get_mean));

        std::vector<tatami_stats::variances::RunningDense<Stat_, Value_, Index_> > runners;
        runners.reserve(nblocks);
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            runners.emplace_back(length, local_means.data(b), local_vars.data(b), false);
        }

        if (blocked) {
            for (I<decltype(NC)> c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(buffer.data());
                runners[block[c]].add(ptr);
            }
        } else {
            for (I<decltype(NC)> c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(buffer.data());
                runners[0].add(ptr);
            }
        }

        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            runners[b].finish();
        }
        local_vars.transfer();
        local_means.transfer();
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Stat_, typename Block_> 
void compute_variances_sparse_column(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& buffers,
    const Block_* const block,
    const std::vector<Index_>& block_size,
    const int num_threads) 
{
    const bool blocked = (block != NULL);
    const auto nblocks = block_size.size();
    const auto NR = mat.nrow(), NC = mat.ncol();
    auto nonzeros = sanisizer::create<std::vector<std::vector<Index_> > >(
        nblocks,
        tatami::create_container_of_Index_size<std::vector<Index_> >(NR)
    );

    tatami::parallelize([&](const int thread, const Index_ start, const Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
        auto ext = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), NC, start, length, [&]{
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            return opt;
        }());

        auto get_var = [&](Index_ b) -> Stat_* { return buffers[b].variances; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_var)> local_vars(thread, nblocks, start, length, std::move(get_var));
        auto get_mean = [&](Index_ b) -> Stat_* { return buffers[b].means; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_mean)> local_means(thread, nblocks, start, length, std::move(get_mean));

        std::vector<tatami_stats::variances::RunningSparse<Stat_, Value_, Index_> > runners;
        runners.reserve(nblocks);
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            runners.emplace_back(length, local_means.data(b), local_vars.data(b), false, start);
        }

        if (blocked) {
            for (I<decltype(NC)> c = 0; c < NC; ++c) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                runners[block[c]].add(range.value, range.index, range.number);
            }
        } else {
            for (I<decltype(NC)> c = 0; c < NC; ++c) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                runners[0].add(range.value, range.index, range.number);
            }
        }

        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            runners[b].finish();
        }
        local_vars.transfer();
        local_means.transfer();
    }, NR, num_threads);
}

template<typename Value_, typename Index_, typename Stat_, typename Block_> 
void compute_variances(
    const tatami::Matrix<Value_, Index_>& mat,
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& buffers,
    const Block_* const block,
    const std::vector<Index_>& block_size,
    const int num_threads) 
{
    if (mat.prefer_rows()) {
        if (mat.sparse()) {
            compute_variances_sparse_row(mat, buffers, block, block_size, num_threads);
        } else {
            compute_variances_dense_row(mat, buffers, block, block_size, num_threads);
        }
    } else {
        if (mat.sparse()) {
            compute_variances_sparse_column(mat, buffers, block, block_size, num_threads);
        } else {
            compute_variances_dense_column(mat, buffers, block, block_size, num_threads);
        }
    }
}

template<typename Stat_, typename Index_>
void extract_weights(
    const std::vector<Stat_>& block_weights,
    const std::vector<Index_>& block_size,
    const Index_ min_size,
    std::vector<Stat_>& tmp_weights
) {
    const auto nblocks = block_weights.size();
    tmp_weights.clear();
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        if (block_size[b] < min_size) { // skip blocks with insufficient cells.
            continue;
        }
        tmp_weights.push_back(block_weights[b]);
    }
}

template<typename Stat_, typename Index_, class Function_>
void extract_pointers(
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& per_block, 
    const std::vector<Index_>& block_size,
    const Index_ min_size,
    const Function_ fun,
    std::vector<Stat_*>& tmp_pointers
) {
    const auto nblocks = per_block.size();
    tmp_pointers.clear();
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        if (block_size[b] < min_size) { // skip blocks with insufficient cells.
            continue;
        }
        tmp_pointers.push_back(fun(per_block[b]));
    }
}

}
/**
 * @endcond
 */

/** 
 * Model the per-feature variances from a log-expression matrix with blocking.
 * The mean and variance of each gene is computed separately for all cells in each block,
 * and a separate trend is fitted to each block to obtain residuals (see `model_gene_variances()`).
 * This ensures that sample and batch effects do not confound the variance estimates.
 *
 * We also compute the average of each statistic across blocks, using the policy described in `ModelGeneVariancesOptions::average_policy`.
 * This is either a quantile (i.e., median, by default) or weighted mean of values for each gene.
 * Weights are determined by `ModelGeneVariancesOptions::block_weight_policy` and are based on the size of each block.
 * The average residual is particularly useful for feature selection with `choose_highly_variable_genes()`.
 *
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Block_ Integer type of the block IDs.
 * @tparam Stat_ Floating-point type of the output statistics.
 *
 * @param mat Matrix of expression values, typically after normalization and log-transformation.
 * Rows should be genes while columns should be cells.
 * @param[in] block Pointer to an array of length equal to the number of cells.
 * Each entry should be a 0-based block identifier in \f$[0, B)\f$ where \f$B\f$ is the total number of blocks.
 * `block` can also be a `nullptr`, in which case all cells are assumed to belong to the same block.
 * @param[out] buffers Collection of pointers of arrays in which to store the output statistics.
 * The length of `ModelGeneVariancesBlockedResults::per_block` should be equal to the number of blocks.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Block_, typename Stat_>
void model_gene_variances_blocked(
    const tatami::Matrix<Value_, Index_>& mat, 
    const Block_* const block, 
    const ModelGeneVariancesBlockedBuffers<Stat_>& buffers,
    const ModelGeneVariancesOptions& options)
{
    const Index_ NR = mat.nrow(), NC = mat.ncol();
    std::vector<Index_> block_size;

    if (block) {
        block_size = tatami_stats::tabulate_groups(block, NC);
        internal::compute_variances(mat, buffers.per_block, block, block_size, options.num_threads);
    } else {
        block_size.push_back(NC); // everything is one big block.
        internal::compute_variances(mat, buffers.per_block, block, block_size, options.num_threads);
    }
    const auto nblocks = block_size.size();

    FitVarianceTrendWorkspace<Stat_> work;
    auto fopt = options.fit_variance_trend_options;
    fopt.num_threads = options.num_threads;
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        const auto& current = buffers.per_block[b];
        if (block_size[b] >= 2) {
            fit_variance_trend(NR, current.means, current.variances, current.fitted, current.residuals, work, fopt);
        } else {
            std::fill_n(current.fitted, NR, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(current.residuals, NR, std::numeric_limits<double>::quiet_NaN());
        }
    }

    const auto ave_means = buffers.average.means;
    const auto ave_variances = buffers.average.variances;
    const auto ave_fitted = buffers.average.fitted;
    const auto ave_residuals = buffers.average.residuals;

    std::vector<Stat_*> tmp_pointers;
    tmp_pointers.reserve(nblocks);

    if (options.block_average_policy == BlockAveragePolicy::MEAN) {
        const auto block_weight = scran_blocks::compute_weights<Stat_>(block_size, options.block_weight_policy, options.variable_block_weight_parameters);
        std::vector<Stat_> tmp_weights;
        tmp_weights.reserve(nblocks);

        if (ave_means) {
            internal::extract_weights(block_weight, block_size, static_cast<Index_>(1), tmp_weights);
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(1), [](const auto& x) -> Stat_* { return x.means; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_means, /* skip_nan = */ false);
        }

        // Skip blocks without enough cells to compute the variance.
        internal::extract_weights(block_weight, block_size, static_cast<Index_>(2), tmp_weights);

        if (ave_variances) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.variances; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_variances, /* skip_nan = */ false);
        }

        if (ave_fitted) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.fitted; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_fitted, /* skip_nan = */ false);
        }

        if (ave_residuals) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.residuals; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_residuals, /* skip_nan = */ false);
        }

    } else if (options.block_average_policy == BlockAveragePolicy::QUANTILE) {
        if (ave_means) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(1), [](const auto& x) -> Stat_* { return x.means; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_means, /* skip_nan = */ false);
        }

        // Skip blocks without enough cells to compute the variance.

        if (ave_variances) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.variances; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_variances, /* skip_nan = */ false);
        }

        if (ave_fitted) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.fitted; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_fitted, /* skip_nan = */ false);
        }

        if (ave_residuals) {
            internal::extract_pointers(buffers.per_block, block_size, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.residuals; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_residuals, /* skip_nan = */ false);
        }
    }
}

/** 
 * Model the per-gene variances as a function of the mean in single-cell expression data.
 * We compute the mean and variance for each gene and fit a trend to the variances with respect to the means using `fit_variance_trend()`.
 * We assume that most genes at any given abundance are not highly variable, such that the fitted value of the trend is interpreted as the "uninteresting" variance - 
 * this is mostly attributed to technical variation like sequencing noise, but can also represent constitutive biological noise like transcriptional bursting.
 * Under this assumption, the residual can be treated as a measure of biologically interesting variation.
 * Genes with large residuals can then be selected for downstream analyses, e.g., with `choose_highly_variable_genes()`.
 *
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Stat_ Floating-point type of the output statistics.
 *
 * @param mat Matrix of expression values, typically after normalization and log-transformation.
 * Rows should be genes while columns should be cells.
 * @param buffers Collection of buffers in which to store the computed statistics.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Stat_> 
void model_gene_variances(
    const tatami::Matrix<Value_, Index_>& mat, 
    ModelGeneVariancesBuffers<Stat_> buffers, // yes, the lack of a const ref here is deliberate, we need to move it into bbuffers anyway.
    const ModelGeneVariancesOptions& options)
{
    ModelGeneVariancesBlockedBuffers<Stat_> bbuffers;
    bbuffers.per_block.emplace_back(std::move(buffers));

    bbuffers.average.means = NULL;
    bbuffers.average.variances = NULL;
    bbuffers.average.fitted = NULL;
    bbuffers.average.residuals = NULL;

    model_gene_variances_blocked(mat, static_cast<Index_*>(NULL), bbuffers, options);
}

/** 
 * Overload of `model_gene_variances()` that allocates space for the output statistics.
 *
 * @tparam Stat_ Floating-point type of the output statistics.
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Matrix of expression values, typically after normalization and log-transformation.
 * Rows should be genes while columns should be cells.
 * @param options Further options.
 *
 * @return Results of the variance modelling.
 */
template<typename Stat_ = double, typename Value_, typename Index_>
ModelGeneVariancesResults<Stat_> model_gene_variances(const tatami::Matrix<Value_, Index_>& mat, const ModelGeneVariancesOptions& options) {
    ModelGeneVariancesResults<Stat_> output(mat.nrow()); // cast is safe, as any tatami Index_ can always fit into a size_t.

    ModelGeneVariancesBuffers<Stat_> buffers;
    buffers.means = output.means.data();
    buffers.variances = output.variances.data();
    buffers.fitted = output.fitted.data();
    buffers.residuals = output.residuals.data();

    model_gene_variances(mat, std::move(buffers), options);
    return output;
}

/** 
 * Overload of `model_gene_variances_blocked()` that allocates space for the output statistics.
 *
 * @tparam Stat_ Floating-point type of the output statistics.
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Block_ Integer type of the block IDs.
 *
 * @param mat Matrix of expression values, typically after normalization and log-transformation.
 * Rows should be genes while columns should be cells.
 * @param[in] block Pointer to an array of length equal to the number of cells, containing 0-based block identifiers.
 * This may also be a `nullptr` in which case all cells are assumed to belong to the same block.
 * @param options Further options.
 *
 * @return Results of the variance modelling in each block.
 * An average for each statistic is also computed if `ModelGeneVariancesOptions::average_policy` is not `BlockAveragePolicy::NONE`.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Block_>
ModelGeneVariancesBlockedResults<Stat_> model_gene_variances_blocked(const tatami::Matrix<Value_, Index_>& mat, const Block_* const block, const ModelGeneVariancesOptions& options) {
    const auto nblocks = (block ? tatami_stats::total_groups(block, mat.ncol()) : 1);

    const bool do_average = options.compute_average /* for back-compatibility */ && options.block_average_policy != BlockAveragePolicy::NONE;
    ModelGeneVariancesBlockedResults<Stat_> output(mat.nrow(), nblocks, do_average); // cast is safe, any tatami Index_ can always fit into a size_t.

    ModelGeneVariancesBlockedBuffers<Stat_> buffers;
    sanisizer::resize(buffers.per_block, nblocks);
    for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
        auto& current = buffers.per_block[b];
        current.means = output.per_block[b].means.data();
        current.variances = output.per_block[b].variances.data();
        current.fitted = output.per_block[b].fitted.data();
        current.residuals = output.per_block[b].residuals.data();
    }

    if (!do_average) {
        buffers.average.means = NULL;
        buffers.average.variances = NULL;
        buffers.average.fitted = NULL;
        buffers.average.residuals = NULL;
    } else {
        buffers.average.means = output.average.means.data();
        buffers.average.variances = output.average.variances.data();
        buffers.average.fitted = output.average.fitted.data();
        buffers.average.residuals = output.average.residuals.data();
    }

    model_gene_variances_blocked(mat, block, buffers, options);
    return output;
}

}

#endif
