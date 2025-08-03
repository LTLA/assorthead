#ifndef SCRAN_MODEL_GENE_VARIANCES_H
#define SCRAN_MODEL_GENE_VARIANCES_H

#include "fit_variance_trend.hpp"

#include <algorithm>
#include <vector>
#include <limits>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "scran_blocks/scran_blocks.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file model_gene_variances.hpp
 * @brief Model the per-gene variances. 
 */

namespace scran_variances {

/**
 * @brief Options for `model_gene_variances()` and friends.
 */
struct ModelGeneVariancesOptions {
    /**
     * Options for fitting the mean-variance trend.
     */
    FitVarianceTrendOptions fit_variance_trend_options;

    /**
     * Policy to use for weighting the contribution from each block when computing the average for each statistic.
     * Only relevant to `model_gene_variances_blocked()` overloads where averaged outputs are requested.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights.
     * Only relevant to `model_gene_variances_blocked()` overloads where averaged outputs are requested
     * and `ModelGeneVariancesOptions::block_weight_policy = scran_blocks::WeightPolicy::VARIABLE`.
     */
    scran_blocks::VariableWeightParameters variable_block_weight_parameters; 

    /**
     * Whether to compute the average of each statistic across blocks.
     * Note that this only affects the `model_gene_variances_blocked()` method that returns a `ModelGeneVariancesBlockedResults` object.
     */
    bool compute_average = true;

    /**
     * Number of threads to use. 
     * The parallelization scheme is defined by `tatami::parallelize()` and `FitVarianceTrendOptions::num_threads`.
     */
    int num_threads = 1;
};

/**
 * @brief Buffers for `model_gene_variances()` and friends.
 * @tparam Stat_ Floating-point type for the output statistics.
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
 * @tparam Stat_ Floating-point type for the output statistics.
 */
template<typename Stat_>
struct ModelGeneVariancesResults {
    /**
     * @cond
     */
    ModelGeneVariancesResults() = default;

    template<typename Ngenes_>
    ModelGeneVariancesResults(Ngenes_ ngenes) :
        means(sanisizer::cast<decltype(means.size())>(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        variances(sanisizer::cast<decltype(variances.size())>(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        fitted(sanisizer::cast<decltype(fitted.size())>(ngenes)
#ifdef SCRAN_VARIANCES_TEST_INIT
            , SCRAN_VARIANCES_TEST_INIT
#endif
        ),
        residuals(sanisizer::cast<decltype(residuals.size())>(ngenes)
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
 * @tparam Stat_ Floating-point type for the output statistics.
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
 * @tparam Stat_ Floating-point type for the output statistics.
 */
template<typename Stat_>
struct ModelGeneVariancesBlockedResults {
    /**
     * @cond
     */
    ModelGeneVariancesBlockedResults() = default;

    template<typename Ngenes_, typename Nblocks_>
    ModelGeneVariancesBlockedResults(Ngenes_ ngenes, Nblocks_ nblocks, bool compute_average) : average(compute_average ? ngenes : 0) {
        per_block.reserve(nblocks);
        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
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
     * This is only populated if `ModelGeneVariancesOptions::compute_average = true`.
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
    const Block_* block,
    const std::vector<Index_>& block_size,
    int num_threads)
{
    bool blocked = (block != NULL);
    auto nblocks = block_size.size();
    auto NR = mat.nrow(), NC = mat.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
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
                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                    buffers[b].means[r] = tmp_means[b];
                    buffers[b].variances[r] = tmp_vars[b];
                }
            } else {
                auto stat = tatami_stats::variances::direct(ptr, NC, false);
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
    const Block_* block,
    const std::vector<Index_>& block_size,
    int num_threads)
{
    bool blocked = (block != NULL);
    auto nblocks = block_size.size();
    auto NR = mat.nrow(), NC = mat.ncol();

    tatami::parallelize([&](int, Index_ start, Index_ length) -> void {
        auto tmp_means = sanisizer::create<std::vector<Stat_> >(nblocks);
        auto tmp_vars = sanisizer::create<std::vector<Stat_> >(nblocks);
        auto tmp_nzero = sanisizer::create<std::vector<Index_> >(nblocks);

        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
        tatami::Options opt;
        opt.sparse_ordered_index = false;
        auto ext = tatami::consecutive_extractor<true>(mat, true, start, length, opt);

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
                for (decltype(nblocks) b = 0; b < nblocks; ++b) {
                    buffers[b].means[r] = tmp_means[b];
                    buffers[b].variances[r] = tmp_vars[b];
                }
            } else {
                auto stat = tatami_stats::variances::direct(range.value, range.number, NC, false);
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
    const Block_* block,
    const std::vector<Index_>& block_size,
    int num_threads)
{
    bool blocked = (block != NULL);
    auto nblocks = block_size.size();
    auto NR = mat.nrow(), NC = mat.ncol();

    tatami::parallelize([&](int thread, Index_ start, Index_ length) -> void {
        auto buffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        auto ext = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), NC, start, length);

        auto get_var = [&](Index_ b) -> Stat_* { return buffers[b].variances; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_var)> local_vars(thread, nblocks, start, length, std::move(get_var));
        auto get_mean = [&](Index_ b) -> Stat_* { return buffers[b].means; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_mean)> local_means(thread, nblocks, start, length, std::move(get_mean));

        std::vector<tatami_stats::variances::RunningDense<Stat_, Value_, Index_> > runners;
        runners.reserve(nblocks);
        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            runners.emplace_back(length, local_means.data(b), local_vars.data(b), false);
        }

        if (blocked) {
            for (decltype(NC) c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(buffer.data());
                runners[block[c]].add(ptr);
            }
        } else {
            for (decltype(NC) c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(buffer.data());
                runners[0].add(ptr);
            }
        }

        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
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
    const Block_* block,
    const std::vector<Index_>& block_size,
    int num_threads) 
{
    bool blocked = (block != NULL);
    auto nblocks = block_size.size();
    auto NR = mat.nrow(), NC = mat.ncol();
    auto nonzeros = sanisizer::create<std::vector<std::vector<Index_> > >(
        nblocks,
        tatami::create_container_of_Index_size<std::vector<Index_> >(NR)
    );

    tatami::parallelize([&](int thread, Index_ start, Index_ length) -> void {
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(length);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);
        tatami::Options opt;
        opt.sparse_ordered_index = false;
        auto ext = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), NC, start, length, opt);

        auto get_var = [&](Index_ b) -> Stat_* { return buffers[b].variances; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_var)> local_vars(thread, nblocks, start, length, std::move(get_var));
        auto get_mean = [&](Index_ b) -> Stat_* { return buffers[b].means; };
        tatami_stats::LocalOutputBuffers<Stat_, decltype(get_mean)> local_means(thread, nblocks, start, length, std::move(get_mean));

        std::vector<tatami_stats::variances::RunningSparse<Stat_, Value_, Index_> > runners;
        runners.reserve(nblocks);
        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
            runners.emplace_back(length, local_means.data(b), local_vars.data(b), false, start);
        }

        if (blocked) {
            for (decltype(NC) c = 0; c < NC; ++c) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                runners[block[c]].add(range.value, range.index, range.number);
            }
        } else {
            for (decltype(NC) c = 0; c < NC; ++c) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                runners[0].add(range.value, range.index, range.number);
            }
        }

        for (decltype(nblocks) b = 0; b < nblocks; ++b) {
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
    const Block_* block,
    const std::vector<Index_>& block_size,
    int num_threads) 
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

template<typename Index_, typename Stat_, class Function_>
void compute_average(
    Index_ ngenes, 
    const std::vector<ModelGeneVariancesBuffers<Stat_> >& per_block, 
    const std::vector<Index_>& block_size,
    const std::vector<Stat_>& block_weights,
    Index_ min_size,
    Function_ fun,
    std::vector<Stat_*>& tmp_pointers,
    std::vector<Stat_>& tmp_weights,
    Stat_* output) 
{
    if (!output) {
        return;
    }

    tmp_pointers.clear();
    tmp_weights.clear();
    for (decltype(per_block.size()) b = 0, nblocks = per_block.size(); b < nblocks; ++b) {
        if (block_size[b] < min_size) { // skip blocks with insufficient cells.
            continue;
        }
        tmp_weights.push_back(block_weights[b]);
        tmp_pointers.push_back(fun(per_block[b]));
    }

    scran_blocks::average_vectors_weighted(ngenes, tmp_pointers, tmp_weights.data(), output, /* skip_nan = */ false);
}

}
/**
 * @endcond
 */

/** 
 * Compute and model the per-feature variances from a log-expression matrix with blocking.
 * The mean and variance of each gene is computed separately for all cells in each block, 
 * and a separate trend is fitted to each block to obtain residuals (see `model_gene_variances()`).
 * This ensures that sample and batch effects do not confound the variance estimates.
 *
 * We also compute the average of each statistic across blocks, using the weighting strategy specified in `ModelGeneVariancesOptions::block_weight_policy`.
 * The average residual is particularly useful for feature selection with `choose_highly_variable_genes()`.
 *
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Block_ Integer type to hold the block IDs.
 * @tparam Stat_ Floating-point type for the output statistics.
 *
 * @param mat A **tatami** matrix containing log-expression values.
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
    const Block_* block, 
    const ModelGeneVariancesBlockedBuffers<Stat_>& buffers,
    const ModelGeneVariancesOptions& options)
{
    Index_ NR = mat.nrow(), NC = mat.ncol();
    std::vector<Index_> block_size;

    if (block) {
        block_size = tatami_stats::tabulate_groups(block, NC);
        internal::compute_variances(mat, buffers.per_block, block, block_size, options.num_threads);
    } else {
        block_size.push_back(NC); // everything is one big block.
        internal::compute_variances(mat, buffers.per_block, block, block_size, options.num_threads);
    }
    auto nblocks = block_size.size();

    FitVarianceTrendWorkspace<Stat_> work;
    auto fopt = options.fit_variance_trend_options;
    fopt.num_threads = options.num_threads;
    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        const auto& current = buffers.per_block[b];
        if (block_size[b] >= 2) {
            fit_variance_trend(NR, current.means, current.variances, current.fitted, current.residuals, work, fopt);
        } else {
            std::fill_n(current.fitted, NR, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(current.residuals, NR, std::numeric_limits<double>::quiet_NaN());
        }
    }

    auto ave_means = buffers.average.means;
    auto ave_variances = buffers.average.variances;
    auto ave_fitted = buffers.average.fitted;
    auto ave_residuals = buffers.average.residuals;

    if (ave_means || ave_variances || ave_fitted || ave_residuals) {
        auto block_weight = scran_blocks::compute_weights<Stat_>(block_size, options.block_weight_policy, options.variable_block_weight_parameters);

        std::vector<Stat_*> tmp_pointers;
        std::vector<Stat_> tmp_weights;
        tmp_pointers.reserve(nblocks);
        tmp_weights.reserve(nblocks);

        internal::compute_average(NR, buffers.per_block, block_size, block_weight,
            /* min_size = */ static_cast<Index_>(1),  // skip blocks without enough cells to compute the mean.
            [](const auto& x) -> Stat_* { return x.means; }, tmp_pointers, tmp_weights, ave_means);

        internal::compute_average(NR, buffers.per_block, block_size, block_weight,
            /* min_size = */ static_cast<Index_>(2), // skip blocks without enough cells to compute the variance.
            [](const auto& x) -> Stat_* { return x.variances; }, tmp_pointers, tmp_weights, ave_variances);

        internal::compute_average(NR, buffers.per_block, block_size, block_weight, 
            /* min_size = */ static_cast<Index_>(2), // ditto.
            [](const auto& x) -> Stat_* { return x.fitted; }, tmp_pointers, tmp_weights, ave_fitted);

        internal::compute_average(NR, buffers.per_block, block_size, block_weight, 
            /* min_size = */ static_cast<Index_>(2), // ditto.
            [](const auto& x) -> Stat_* { return x.residuals; }, tmp_pointers, tmp_weights, ave_residuals);
    }
}

/** 
 * Here, we scan through a log-transformed normalized expression matrix and compute per-gene means and variances.
 * We then fit a trend to the variances with respect to the means using `fit_variance_trend()`.
 * We assume that most genes at any given abundance are not highly variable, such that the fitted value of the trend is interpreted as the "uninteresting" variance - 
 * this is mostly attributed to technical variation like sequencing noise, but can also represent constitutive biological noise like transcriptional bursting.
 * Under this assumption, the residual can be treated as a measure of biologically interesting variation, and can be used to identify relevant features for downstream analyses.
 *
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Stat_ Floating-point type for the output statistics.
 *
 * @param mat A **tatami** matrix containing log-expression values.
 * Rows should be genes while columns should be cells.
 * @param buffers Collection of buffers in which to store the computed statistics.
 * @param options Further options.
 */
template<typename Value_, typename Index_, typename Stat_> 
void model_gene_variances(const tatami::Matrix<Value_, Index_>& mat, 
    ModelGeneVariancesBuffers<Stat_> buffers, // yes, the lack of a const ref here is deliberate, we need to move it into bbuffers anyway.
    const ModelGeneVariancesOptions& options) {

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
 * @tparam Stat_ Floating-point type for the output statistics.
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type for the row/column indices.
 *
 * @param mat A **tatami** matrix containing log-expression values.
 * Rows should be genes while columns should be cells.
 * @param options Further options.
 *
 * @return Results of the variance modelling.
 */
template<typename Stat_ = double, typename Value_, typename Index_>
ModelGeneVariancesResults<Stat_> model_gene_variances(const tatami::Matrix<Value_, Index_>& mat, const ModelGeneVariancesOptions& options) {
    ModelGeneVariancesResults<Stat_> output(mat.nrow());

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
 * @tparam Stat_ Floating-point type for the output statistics.
 * @tparam Value_ Data type of the matrix.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Block_ Integer type, to hold the block IDs.
 *
 * @param mat A **tatami** matrix containing log-expression values.
 * Rows should be genes while columns should be cells.
 * @param[in] block Pointer to an array of length equal to the number of cells, containing 0-based block identifiers.
 * This may also be a `nullptr` in which case all cells are assumed to belong to the same block.
 * @param options Further options.
 *
 * @return Results of the variance modelling in each block.
 * An average for each statistic is also computed if `ModelGeneVariancesOptions::compute_average = true`.
 */
template<typename Stat_ = double, typename Value_, typename Index_, typename Block_>
ModelGeneVariancesBlockedResults<Stat_> model_gene_variances_blocked(const tatami::Matrix<Value_, Index_>& mat, const Block_* block, const ModelGeneVariancesOptions& options) {
    auto nblocks = (block ? tatami_stats::total_groups(block, mat.ncol()) : 1);
    ModelGeneVariancesBlockedResults<Stat_> output(mat.nrow(), nblocks, options.compute_average);

    ModelGeneVariancesBlockedBuffers<Stat_> buffers;
    buffers.per_block.resize(nblocks);
    for (decltype(nblocks) b = 0; b < nblocks; ++b) {
        auto& current = buffers.per_block[b];
        current.means = output.per_block[b].means.data();
        current.variances = output.per_block[b].variances.data();
        current.fitted = output.per_block[b].fitted.data();
        current.residuals = output.per_block[b].residuals.data();
    }

    if (!options.compute_average) {
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
