#ifndef SCRAN_MODEL_GENE_VARIANCES_HPP
#define SCRAN_MODEL_GENE_VARIANCES_HPP

#include <algorithm>
#include <vector>
#include <limits>
#include <cstddef>
#include <cassert>
#include <optional>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "quickstats/quickstats.hpp"
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
     * Whether to fit a mean-variance trend.
     * It can be occasionally desirable to disable the trend fitting if the trend is to be obtained from some other source,
     * e.g., using a separate set of spike-in transcripts to estimate the technical component of each gene's variance.
     */
    bool trend = true;

    /**
     * Options for fitting the mean-variance trend.
     * Ignored if `ModelGeneVariancesOptions::trend = false`.
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
     * By default, we define equal weights for blocks beyond a certain size (see `ModelGeneVariancesOptions::variable_block_weight_parameters`).
     * For smaller blocks, the weight is linearly proportional to its size to avoid putting too much confidence in estimates from very small blocks.
     *
     * Other options include `scran_blocks::WeightPolicy::EQUAL`, where all blocks are equally weighted regardless of size;
     * and `scran_blocks::WeightPolicy::NONE`, where the contribution of each block is proportional to its size.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights, including the threshold at which blocks are considered to be large enough to have equal weight.
     * Only relevant to `model_gene_variances_blocked()` when `ModelGeneVariancesOptions::average_policy = BlockAveragePolicy::MEAN`
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
    Stat_* mean;

    /**
     * Pointer to an array of length equal to the number of genes, containing the variance in the log-expression for each gene.
     */
    Stat_* variance;

    /**
     * Pointer to an array of length equal to the number of genes, containing the fitted value of the mean-variance trend for each gene.
     *
     * If this or `ModelGeneVariancesBuffers::residual` is `NULL`, no trend is fitted.
     */
    Stat_* fitted;

    /**
     * Vector of length equal to the number of genes, containing the residuals of the mean-variance trend for each gene.
     *
     * If this or `ModelGeneVariancesBuffers::fitted` is `NULL`, no trend is fitted.
     */
    Stat_* residual;
};

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
    const ModelGeneVariancesBuffers<Stat_> buffers,
    const ModelGeneVariancesOptions& options
) {
    tatami_stats::VarianceBuffers<Stat_> vbuf;
    vbuf.mean = buffers.mean;
    vbuf.variance = buffers.variance;

    tatami_stats::VarianceOptions<Stat_> vopt;
    vopt.num_threads = options.num_threads;
    tatami_stats::variance(true, mat, vbuf, vopt);

    FitVarianceTrendWorkspace<Stat_> work;
    auto fopt = options.fit_variance_trend_options;
    fopt.num_threads = options.num_threads;

    if (buffers.fitted != NULL && buffers.residual != NULL) {
        const auto NR = mat.nrow();
        if (mat.ncol() >= 2) {
            fit_variance_trend(NR, buffers.mean, buffers.variance, buffers.fitted, buffers.residual, work, fopt);
        } else {
            std::fill_n(buffers.fitted, NR, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(buffers.residual, NR, std::numeric_limits<double>::quiet_NaN());
        }
    }
}

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

    ModelGeneVariancesResults(const std::size_t ngenes, const bool trend) :
        mean(sanisizer::cast<I<decltype(mean.size())> >(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        variance(sanisizer::cast<I<decltype(variance.size())> >(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        fitted(sanisizer::cast<I<decltype(fitted.size())> >(trend ? ngenes : 0)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        residual(sanisizer::cast<I<decltype(residual.size())> >(trend ? ngenes : 0)
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
    std::vector<Stat_> mean;

    /**
     * Vector of length equal to the number of genes, containing the variance in the log-expression for each gene.
     */
    std::vector<Stat_> variance;

    /**
     * Vector of length equal to the number of genes, containing the fitted value of the mean-variance trend for each gene.
     *
     * This will be empty if `ModelGeneVariancesOptions::trend = false`.
     */
    std::vector<Stat_> fitted;

    /**
     * Vector of length equal to the number of genes, containing the residuals of the mean-variance trend for each gene.
     *
     * This will be empty if `ModelGeneVariancesOptions::trend = false`.
     */
    std::vector<Stat_> residual;
};

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
    ModelGeneVariancesResults<Stat_> output(mat.nrow(), options.trend); // cast is safe, as any tatami Index_ can always fit into a size_t.

    ModelGeneVariancesBuffers<Stat_> buffers;
    buffers.mean = output.mean.data();
    buffers.variance = output.variance.data();

    if (options.trend) {
        buffers.fitted = output.fitted.data();
        buffers.residual = output.residual.data();
    } else {
        buffers.fitted = NULL;
        buffers.residual = NULL;
    }

    model_gene_variances(mat, std::move(buffers), options);
    return output;
}

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
     *
     * Any of the pointers in this object may be `NULL`, in which case the corresponding average is not computed.
     *
     * If `average.fitted` or `average.residual` is not `NULL`, all entries of `per_block` should have non-`NULL` pointers for their own `fitted` and `residual`.
     * (That is, a mean-variance trend should have been fitted in each block.) 
     * Otherwise, an error will be thrown.
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

    ModelGeneVariancesBlockedResults(const std::size_t ngenes, const std::size_t nblocks, const bool do_average, const bool do_trend) :
        average(do_average ? ngenes : 0, do_trend)
    {
        per_block.reserve(nblocks);
        for (I<decltype(nblocks)> b = 0; b < nblocks; ++b) {
            per_block.emplace_back(ngenes, do_trend);
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
template<typename Stat_, typename Index_>
void extract_blocked_weights(
    const std::size_t num_blocks,
    const std::vector<Stat_>& block_weights,
    const std::vector<Index_>& block_sizes,
    const Index_ min_size,
    std::vector<Stat_>& tmp_weights
) {
    assert(sanisizer::is_equal(num_blocks, block_weights.size()));
    assert(sanisizer::is_equal(num_blocks, block_sizes.size()));
    tmp_weights.clear();
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (block_sizes[b] < min_size) { // skip blocks with insufficient cells.
            continue;
        }
        tmp_weights.push_back(block_weights[b]);
    }
}

template<typename Stat_, typename Index_, class Function_>
void extract_blocked_pointers(
    const std::size_t num_blocks,
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& per_block, 
    const std::vector<Index_>& block_sizes,
    const Index_ min_size,
    const Function_ fun,
    std::vector<Stat_*>& tmp_pointers
) {
    assert(sanisizer::is_equal(num_blocks, per_block.size()));
    assert(sanisizer::is_equal(num_blocks, block_sizes.size()));
    tmp_pointers.clear();
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (block_sizes[b] < min_size) { // skip blocks with insufficient cells.
            continue;
        }
        tmp_pointers.push_back(fun(per_block[b]));
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
 * @param num_blocks Total number of blocks, a.k.a., \f$B\f$.
 * @param[out] buffers Collection of pointers of arrays in which to store the output statistics.
 * The length of `ModelGeneVariancesBlockedResults::per_block` should be equal to the number of blocks.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Block_, typename Stat_>
void model_gene_variances_blocked(
    const tatami::Matrix<Value_, Index_>& mat, 
    const Block_* const block, 
    const std::size_t num_blocks,
    const ModelGeneVariancesBlockedBuffers<Stat_>& buffers,
    const ModelGeneVariancesOptions& options
) {
    if (!sanisizer::is_equal(num_blocks, buffers.per_block.size())) {
        throw std::runtime_error("length of 'buffers.per_block' is not equal to 'num_blocks'");
    }

    const Index_ NR = mat.nrow();
    const Index_ NC = mat.ncol();
    assert(NC == 0 || sanisizer::is_less_than(*std::max_element(block, block + NC), num_blocks));

    tatami_stats::GroupVarianceBuffers<Stat_> vbuf;
    vbuf.mean.reserve(num_blocks);
    vbuf.variance.reserve(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        vbuf.mean.push_back(buffers.per_block[b].mean);
        vbuf.variance.push_back(buffers.per_block[b].variance);
    }

    auto block_sizes = sanisizer::create<std::vector<Index_> >(num_blocks);
    for (Index_ c = 0; c < NC; ++c) {
        block_sizes[block[c]] += 1;
    }

    tatami_stats::GroupVarianceOptions<Stat_> vopt;
    vopt.num_threads = options.num_threads;
    tatami_stats::group_variance(true, mat, block, num_blocks, block_sizes.data(), vbuf, vopt);

    FitVarianceTrendWorkspace<Stat_> work;
    auto fopt = options.fit_variance_trend_options;
    fopt.num_threads = options.num_threads;
    bool all_trends_fitted = true;

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const auto& current = buffers.per_block[b];
        if (current.fitted == NULL || current.residual == NULL) {
            all_trends_fitted = false;
            continue;
        }
        if (block_sizes[b] >= 2) {
            fit_variance_trend(NR, current.mean, current.variance, current.fitted, current.residual, work, fopt);
        } else {
            std::fill_n(current.fitted, NR, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(current.residual, NR, std::numeric_limits<double>::quiet_NaN());
        }
    }

    const auto ave_means = buffers.average.mean;
    const auto ave_variances = buffers.average.variance;
    const auto ave_fitted = buffers.average.fitted;
    const auto ave_residuals = buffers.average.residual;

    if ((ave_fitted || ave_residuals) && !all_trends_fitted) {
        throw std::runtime_error("cannot compute average fitted values/residuals without per-block trend fits");
    }

    std::vector<Stat_*> tmp_pointers;
    tmp_pointers.reserve(num_blocks);

    if (options.block_average_policy == BlockAveragePolicy::MEAN) {
        const auto block_weight = scran_blocks::compute_weights<Stat_>(block_sizes, options.block_weight_policy, options.variable_block_weight_parameters);
        std::vector<Stat_> tmp_weights;
        tmp_weights.reserve(num_blocks);

        if (ave_means) {
            extract_blocked_weights(num_blocks, block_weight, block_sizes, static_cast<Index_>(1), tmp_weights);
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(1), [](const auto& x) -> Stat_* { return x.mean; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_means, /* skip_nan = */ false);
        }

        // Skip blocks without enough cells to compute the variance.
        extract_blocked_weights(num_blocks, block_weight, block_sizes, static_cast<Index_>(2), tmp_weights);

        if (ave_variances) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.variance; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_variances, /* skip_nan = */ false);
        }

        if (ave_fitted) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.fitted; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_fitted, /* skip_nan = */ false);
        }

        if (ave_residuals) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.residual; }, tmp_pointers);
            scran_blocks::parallel_weighted_means(NR, tmp_pointers, tmp_weights.data(), ave_residuals, /* skip_nan = */ false);
        }

    } else if (options.block_average_policy == BlockAveragePolicy::QUANTILE) {
        if (ave_means) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(1), [](const auto& x) -> Stat_* { return x.mean; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_means, /* skip_nan = */ false);
        }

        // Again, skip blocks without enough cells to compute the variance.

        if (ave_variances) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.variance; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_variances, /* skip_nan = */ false);
        }

        if (ave_fitted) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.fitted; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_fitted, /* skip_nan = */ false);
        }

        if (ave_residuals) {
            extract_blocked_pointers(num_blocks, buffers.per_block, block_sizes, static_cast<Index_>(2), [](const auto& x) -> Stat_* { return x.residual; }, tmp_pointers);
            scran_blocks::parallel_quantiles(NR, tmp_pointers, options.block_quantile, ave_residuals, /* skip_nan = */ false);
        }
    }
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
 * @param num_blocks Total number of blocks.
 * @param options Further options.
 *
 * @return Results of the variance modelling in each block.
 * An average for each statistic is also computed if `ModelGeneVariancesOptions::average_policy` is not `BlockAveragePolicy::NONE`.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Block_>
ModelGeneVariancesBlockedResults<Stat_> model_gene_variances_blocked(
    const tatami::Matrix<Value_, Index_>& mat,
    const Block_* const block,
    const std::size_t num_blocks,
    const ModelGeneVariancesOptions& options
) {
    const bool do_average = options.compute_average /* for back-compatibility */ && options.block_average_policy != BlockAveragePolicy::NONE;
    ModelGeneVariancesBlockedResults<Stat_> output(
        mat.nrow(), // cast is safe, any tatami Index_ can always fit into a size_t.
        num_blocks,
        do_average,
        options.trend
    );

    ModelGeneVariancesBlockedBuffers<Stat_> buffers;
    sanisizer::resize(buffers.per_block, num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        auto& current = buffers.per_block[b];
        current.mean = output.per_block[b].mean.data();
        current.variance = output.per_block[b].variance.data();

        if (options.trend) {
            current.fitted = output.per_block[b].fitted.data();
            current.residual = output.per_block[b].residual.data();
        } else {
            current.fitted = NULL;
            current.residual = NULL;
        }
    }

    if (!do_average) {
        buffers.average.mean = NULL;
        buffers.average.variance = NULL;
        buffers.average.fitted = NULL;
        buffers.average.residual = NULL;
    } else {
        buffers.average.mean = output.average.mean.data();
        buffers.average.variance = output.average.variance.data();

        if (options.trend) {
            buffers.average.fitted = output.average.fitted.data();
            buffers.average.residual = output.average.residual.data();
        } else {
            buffers.average.fitted = NULL;
            buffers.average.residual = NULL;
        }
    }

    model_gene_variances_blocked(mat, block, num_blocks, buffers, options);
    return output;
}

}

#endif
