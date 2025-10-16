#ifndef SCRAN_MARKERS_SUMMARIZE_EFFECTS_HPP
#define SCRAN_MARKERS_SUMMARIZE_EFFECTS_HPP

#include <vector>
#include <cstddef>

#include "summarize_comparisons.hpp"
#include "utils.hpp"

/**
 * @file summarize_effects.hpp
 * @brief Summarize the effect sizes from pairwise comparisons.
 */

namespace scran_markers {

/**
 * @brief Options for `summarize_effects()`.
 */
struct SummarizeEffectsOptions {
    /**
     * Number of threads to use. 
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;

    /**
     * Whether to report the minimum of the effect sizes for each group.
     * Only affects the `summarize_effects()` overload that returns a vector of `SummaryResults`.
     */
    bool compute_min = true;

    /**
     * Whether to report the mean of the effect sizes for each group.
     * Only affects the `summarize_effects()` overload that returns a vector of `SummaryResults`.
     */
    bool compute_mean = true;

    /**
     * Whether to report the median of the effect sizes for each group.
     * Only affects the `summarize_effects()` overload that returns a vector of `SummaryResults`.
     */
    bool compute_median = true;

    /**
     * Whether to report the maximum of the effect sizes for each group.
     * Only affects the `summarize_effects()` overload that returns a vector of `SummaryResults`.
     */
    bool compute_max = true;

    /**
     * Whether to report the min-rank of the effect sizes for each group.
     * Only affects the `summarize_effects()` overload that returns a vector of `SummaryResults`.
     */
    bool compute_min_rank = true;

    /**
     * Whether to preserve ties when computing the minimum rank.
     * If `true`, tied genes with equal effect sizes receive the same rank within each pairwise comparison. 
     * Otherwise, ties are broken in a stable manner, i.e., genes in earlier rows will receive a higher rank.
     */
    bool min_rank_preserve_ties = false;
};

/**
 * Given \f$N\f$ groups, each group is involved in \f$N - 1\f$ pairwise comparisons and thus has \f$N - 1\f$ effect sizes (e.g., as computed by `score_markers_pairwise()`).
 * We summarize each group's effect sizes into a small set of desriptive statistics like the mininum, median or mean.
 * Users can then sort genes by any of these summaries to obtain a ranking of potential markers for that group.
 *
 * The choice of summary statistic determines the interpretation of the ranking.
 * Given a group \f$X\f$:
 * 
 * - A large mean effect size indicates that the gene is upregulated in \f$X\f$ compared to the average of the other groups.
 *   A small value indicates that the gene is downregulated in \f$X\f$ instead.
 *   This is a good general-purpose summary statistic that can be ranked in descending order to obtain the strongest upregulated markers in \f$X\f$.
 * - A large median effect size indicates that the gene is upregulated in \f$X\f$ compared to most (>50%) other groups.
 *   A small value indicates that the gene is downregulated in \f$X\f$ instead.
 *   This is also a good general-purpose summary like the mean, with the advantage of being more robust to outlier effects compared to the mean.
 *   However, it also has the disadvantage of being less sensitive to strong effects in a minority of comparisons.
 * - A large minimum effect size indicates that the gene is upregulated in \f$X\f$ compared to all other groups.
 *   A small value indicates that the gene is downregulated in \f$X\f$ compared to at least one other group.
 *   For upregulation, this is the most stringent summary as markers will only have extreme values if they are _uniquely_ upregulated in \f$X\f$ compared to every other group.
 *   However, it may not be effective if \f$X\f$ is closely related to any of the groups.
 * - A large maximum effect size indicates that the gene is upregulated in \f$X\f$ compared to at least one other group.
 *   A small value indicates that the gene is downregulated in \f$X\f$ compared to all other groups.
 *   For downregulation, this is the most stringent summary as markers will only have extreme values if they are _uniquely_ downregulated in \f$X\f$ compared to every other group.
 *   However, it may not be effective if \f$X\f$ is closely related to any of the groups.
 * - The "minimum rank" (a.k.a., min-rank) is defined by ranking genes based on decreasing effect size _within_ each comparison involving \f$X\f$,
 *   and then taking the smallest rank for each gene _across_ all comparisons involving \f$X\f$.
 *   A minimum rank of 1 means that the gene is the top upregulated gene in at least one comparison to another group.
 *   More generally, a minimum rank of \f$T\f$ indicates that the gene is the \f$T\f$-th upregulated gene in at least one comparison. 
 *   Applying a threshold on the minimum rank is useful for obtaining a set of genes that, in combination, are guaranteed to distinguish \f$X\f$ from every other group.
 *
 * The exact definition of "large" and "small" depends on the choice of effect size. 
 * For signed effects like Cohen's d, delta-mean and delta-detected, the value must be positive to be considered "large", and negative to be considered "small".
 * For the AUC, a value greater than 0.5 is considered "large" and less than 0.5 is considered "small".
 *
 * The interpretation above is also contingent on the threshold used (see `score_markers_pairwise()` for details).
 * For positive thresholds, small summary statistics cannot be unambiguously interpreted as downregulation, as the effect is already adjusted to account for the threshold.
 * Only large summary statistics can be safely interpreted, i.e., as evidence for upregulation.
 *
 * NaN effect sizes are allowed, e.g., if two groups do not exist in the same block for a blocked analysis in `score_markers_pairwise_blocked()`.
 * This class will ignore NaN values when computing each summary.
 * If all effects are NaN for a particular group, the summary statistic will also be `NaN`.
 *
 * All choices of summary statistics are enumerated by `Summary`.
 *
 * @tparam Gene_ Integer type of the number of genes.
 * @tparam Stat_ Floating-point type of the statistics.
 * @tparam Rank_ Numeric type of the minimum rank.
 *
 * @param ngenes Number of genes.
 * @param ngroups Number of groups.
 * @param[in] effects Pointer to a 3-dimensional array containing the pairwise statistics, see `ScoreMarkersPairwiseBuffers::cohens_d` for the expected contents.
 * The entry \f$(i, j, k)\f$ (i.e., `effects[i * N * N + j * N + k]`) represents the effect size of gene \f$i\f$ upon comparing group \f$j\f$ against group \f$k\f$.
 * @param[out] summaries Vector of length equal to the number of groups.
 * Each entry corresponds to a group and is used to store the summary statistics for that group.
 * Each pointer in any given `SummaryBuffers` should either point to an array of length equal to the number of genes, 
 * or be `NULL` to indicate that the corresponding summary statistic should not be computed for that group.
 * @param options Further options.
 */
template<typename Gene_, typename Stat_, typename Rank_>
void summarize_effects(
    const Gene_ ngenes,
    const std::size_t ngroups,
    const Stat_* const effects,
    const std::vector<SummaryBuffers<Stat_, Rank_> >& summaries,
    const SummarizeEffectsOptions& options
) {
    internal::compute_min_rank_pairwise(ngenes, ngroups, effects, summaries, options.min_rank_preserve_ties, options.num_threads);
    internal::summarize_comparisons(ngenes, ngroups, effects, summaries, options.num_threads); 
}

/**
 * Overload of `summarize_effects()` that allocates memory for the output summary statistics.
 *
 * @tparam Gene_ Integer type of the number of genes.
 * @tparam Stat Floating point type of the statistics.
 * @tparam Rank_ Numeric type of the minimum rank.
 *
 * @param ngenes Number of genes.
 * @param ngroups Number of groups.
 * @param[in] effects Pointer to a 3-dimensional array containing the pairwise statistics, see `ScoreMarkersPairwiseBuffers::cohens_d` for the expected contents.
 * The entry \f$(i, j, k)\f$ (i.e., `effects[i * N * N + j * N + k]`) represents the effect size of gene \f$i\f$ upon comparing group \f$j\f$ against group \f$k\f$.
 * @param options Further options.
 *
 * @return A vector of length equal to the number of groups.
 * Each `SummaryResults` corresponds to a group and contains the summary statistics (depending on `options`) for that group.
 */
template<typename Stat_ = double, typename Rank_ = int, typename Gene_>
std::vector<SummaryResults<Stat_, Rank_> > summarize_effects(
    const Gene_ ngenes,
    const std::size_t ngroups,
    const Stat_* const effects,
    const SummarizeEffectsOptions& options)
{
    std::vector<SummaryResults<Stat_, Rank_> > output;
    const auto ptrs = internal::fill_summary_results(
        ngenes,
        ngroups,
        output,
        options.compute_min,
        options.compute_mean,
        options.compute_median,
        options.compute_max,
        options.compute_min_rank
    );
    summarize_effects(ngenes, ngroups, effects, ptrs, options);
    return output;
}

}

#endif
