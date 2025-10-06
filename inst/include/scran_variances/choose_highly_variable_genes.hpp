#ifndef SCRAN_VARIANCES_CHOOSE_HIGHLY_VARIABLE_GENES_HPP
#define SCRAN_VARIANCES_CHOOSE_HIGHLY_VARIABLE_GENES_HPP

#include <vector>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"
#include "topicks/topicks.hpp"

/**
 * @file choose_highly_variable_genes.hpp
 * @brief Choose highly variable genes for downstream analyses.
 */

namespace scran_variances {

/**
 * @brief Options for `choose_highly_variable_genes()`.
 */
struct ChooseHighlyVariableGenesOptions {
    /**
     * Number of top genes to choose.
     * Note that the actual number of chosen genes may be: 
     *
     * - smaller than `top`, if the latter is greater than the total number of genes in the dataset. 
     * - smaller than `top`, if `ChooseHighlyVariableGenesOptions::use_bound = true` and `top` is greater than `N`,
     *   where `N` is the number of genes in the dataset with statistics greater than `ChooseHighlyVariableGenesOptions::bound`
     *   (or less than the bound, if `ChosenHighlyVariableGenesOptions::larger = false`).
     * - larger than `top`, if `ChooseHighlyVariableGenesOptions::keep_ties = true` and there are multiple ties with the `top`-th chosen gene.
     */
    std::size_t top = 4000;

    /**
     * Whether larger statistics correspond to higher variability, i.e., the genes with the top `top` largest statistics should be chosen.
     */
    bool larger = true;

    /**
     * Whether to consider an absolute bound on the statistic when choosing HVGs.
     * The value of the bound is determined by `ChooseHighlyVariableGenesOptions::bound`.
     */
    bool use_bound = true;

    /**
     * A lower bound for the statistic, at or below which a gene will not be considered as highly variable even if it is among the top `top` genes.
     * Only used if `ChooseHighlyVariableGenesOptions::use_bound = true`.
     * If `ChooseHighlyVariableGenesOptions::larger = false`, this is an upper bound instead.
     *
     * The default of zero assumes that the input statistics are residuals from a mean-variance trend (see `fit_variance_trend()`),
     * in which case all chosen HVGs should have positive residuals.
     */
    double bound = 0;

    /**
     * Whether to keep all genes with statistics that are tied with the `ChooseHighlyVariableGenesOptions::top`-th gene.
     * If `false`, ties are arbitrarily broken but the number of retained genes will not be greater than `ChooseHighlyVariableGenesOptions::top`.
     */
    bool keep_ties = true;
};

/**
 * @cond
 */
namespace internal {

template<typename Stat_>
topicks::PickTopGenesOptions<Stat_> translate_options(const ChooseHighlyVariableGenesOptions& chvg_options) {
    topicks::PickTopGenesOptions<Stat_> opt;
    opt.keep_ties = chvg_options.keep_ties;
    if (chvg_options.use_bound) {
        opt.bound = chvg_options.bound;
    }
    return opt;
}

}
/**
 * @endcond
 */

/**
 * @tparam Stat_ Type of the variance statistic.
 * @tparam Bool_ Type to be used as a boolean.
 *
 * @param n Number of genes.
 * @param[in] statistic Pointer to an array of length `n` containing the per-gene variance statistics.
 * This is typically the residuals from `model_gene_variances()`.
 * @param[out] output Pointer to an array of length `n`.
 * On output, the `i`-th entry is `true` if the `i`-th gene is to be retained and `false` otherwise.
 * @param options Further options.
 */
template<typename Stat_, typename Bool_>
void choose_highly_variable_genes(const std::size_t n, const Stat_* const statistic, Bool_* const output, const ChooseHighlyVariableGenesOptions& options) {
    topicks::pick_top_genes(n, statistic, options.top, options.larger, output, internal::translate_options<Stat_>(options));
}

/**
 * @tparam Bool_ Type to be used as a boolean.
 * @tparam Stat_ Type of the variance statistic.
 *
 * @param n Number of genes.
 * @param[in] statistic Pointer to an array of length `n` containing the per-gene variance statistics.
 * This is typically the residuals from `model_gene_variances()`.
 * @param options Further options.
 *
 * @return A vector of booleans of length `n`, indicating whether each gene is to be retained.
 */
template<typename Bool_ = char, typename Stat_>
std::vector<Bool_> choose_highly_variable_genes(const std::size_t n, const Stat_* const statistic, const ChooseHighlyVariableGenesOptions& options) {
    auto output = sanisizer::create<std::vector<Bool_> >(n
#ifdef SCRAN_VARIANCES_TEST_INIT
        , SCRAN_VARIANCES_TEST_INIT
#endif
    );
    choose_highly_variable_genes(n, statistic, output.data(), options);
    return output;
}

/**
 * @tparam Index_ Type of the indices.
 * @tparam Stat_ Type of the variance statistic.
 *
 * @param n Number of genes.
 * @param[in] statistic Pointer to an array of length `n` containing the per-gene variance statistics.
 * This is typically the residuals from `model_gene_variances()`.
 * @param options Further options.
 *
 * @return Vector of sorted and unique indices for the chosen genes.
 * All indices are guaranteed to be non-negative and less than `n`.
 */
template<typename Index_, typename Stat_>
std::vector<Index_> choose_highly_variable_genes_index(const Index_ n, const Stat_* const statistic, const ChooseHighlyVariableGenesOptions& options) {
    return topicks::pick_top_genes_index<Index_>(n, statistic, options.top, options.larger, internal::translate_options<Stat_>(options));
}

}

#endif
