#ifndef SCRAN_MARKERS_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_MARKERS_SUMMARIZE_COMPARISONS_HPP

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <cstddef>

#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file summarize_comparisons.hpp
 * @brief Utilities for effect summarization.
 */

namespace scran_markers {

/**
 * @brief Pointers to arrays to hold the summary statistics.
 *
 * @tparam Stat_ Floating-point type of the statistics.
 * @tparam Rank_ Numeric type of the rank.
 */
template<typename Stat_ = double, typename Rank_ = int>
struct SummaryBuffers {
    /**
     * Pointer to an array of length equal to the number of genes,
     * to be filled with the minimum effect size for each gene.
     * If `NULL`, the minimum is not computed.
     */ 
    Stat_* min = NULL;

    /**
     * Pointer to an array of length equal to the number of genes,
     * to be filled with the mean effect size for each gene.
     * If `NULL`, the mean is not computed.
     */ 
    Stat_* mean = NULL;

    /**
     * Pointer to an array of length equal to the number of genes,
     * to be filled with the median effect size for each gene.
     * If `NULL`, the median is not computed.
     */ 
    Stat_* median = NULL;

    /**
     * Pointer to an array of length equal to the number of genes,
     * to be filled with the maximum effect size for each gene.
     * If `NULL`, the maximum is not computed.
     */ 
    Stat_* max = NULL;

    /**
     * Pointer to an array of length equal to the number of genes,
     * to be filled with the minimum rank of the effect sizes for each gene.
     * If `NULL`, the minimum rank is not computed.
     */ 
    Rank_* min_rank = NULL;
};

/**
 * @brief Container for the summary statistics.
 *
 * @tparam Stat_ Floating-point type of the statistics.
 * @tparam Rank_ Numeric type of the rank.
 */
template<typename Stat_ = double, typename Rank_ = int>
struct SummaryResults {
    /**
     * Vector of length equal to the number of genes,
     * to be filled with the minimum effect size for each gene.
     */ 
    std::vector<Stat_> min;

    /**
     * Vector of length equal to the number of genes,
     * to be filled with the mean effect size for each gene.
     */ 
    std::vector<Stat_> mean;

    /**
     * Vector of length equal to the number of genes,
     * to be filled with the median effect size for each gene.
     */
    std::vector<Stat_> median;

    /**
     * Vector of length equal to the number of genes,
     * to be filled with the maximum effect size for each gene.
     */
    std::vector<Stat_> max;

    /**
     * Vector of length equal to the number of genes,
     * to be filled with the minimum rank of the effect sizes for each gene.
     */ 
    std::vector<Rank_> min_rank;
};

/**
 * @cond
 */
namespace internal {

template<typename Stat_, typename Gene_, typename Rank_>
void summarize_comparisons(
    const std::size_t ngroups,
    const Stat_* const effects,
    const std::size_t group,
    const Gene_ gene,
    const SummaryBuffers<Stat_, Rank_>& output,
    std::vector<Stat_>& buffer)
{
    // Ignoring the self comparison and pruning out NaNs.
    std::size_t ncomps = 0;
    for (decltype(I(ngroups)) r = 0; r < ngroups; ++r) {
        if (r == group || std::isnan(effects[r])) {
            continue;
        }
        buffer[ncomps] = effects[r];
        ++ncomps;
    }

    if (ncomps <= 1) {
        Stat_ val = (ncomps == 0 ? std::numeric_limits<Stat_>::quiet_NaN() : buffer[0]);
        if (output.min) {
            output.min[gene] = val;
        }
        if (output.mean) {
            output.mean[gene] = val;
        }
        if (output.max) {
            output.max[gene] = val;
        }
        if (output.median) {
            output.median[gene] = val;
        }

    } else {
        const auto ebegin = buffer.data(), elast = ebegin + ncomps;
        if (output.min) {
            output.min[gene] = *std::min_element(ebegin, elast);
        }
        if (output.mean) {
            output.mean[gene] = std::accumulate(ebegin, elast, static_cast<Stat_>(0)) / ncomps;
        }
        if (output.max) {
            output.max[gene] = *std::max_element(ebegin, elast);
        }
        if (output.median) { // this mutates the buffer, so we put this last to avoid surprises.
            output.median[gene] = tatami_stats::medians::direct(ebegin, ncomps, /* skip_nan = */ false); 
        }
    }
}

template<typename Gene_, typename Stat_, typename Rank_>
void summarize_comparisons(
    const Gene_ ngenes,
    const std::size_t ngroups,
    const Stat_* const effects,
    const std::vector<SummaryBuffers<Stat_, Rank_> >& output,
    const int threads)
{
    tatami::parallelize([&](const int, const Gene_ start, const Gene_ length) -> void {
        auto buffer = sanisizer::create<std::vector<Stat_> >(ngroups);
        for (Gene_ gene = start, end = start + length; gene < end; ++gene) {
            for (decltype(I(ngroups)) l = 0; l < ngroups; ++l) {
                const auto current_effects = effects + sanisizer::nd_offset<std::size_t>(0, ngroups, l, ngroups, gene);
                summarize_comparisons(ngroups, current_effects, l, gene, output[l], buffer);
            }
        }
    }, ngenes, threads);
}

template<typename Stat_, typename Gene_>
Gene_ fill_and_sort_rank_buffer(const Stat_* const effects, const std::size_t stride, std::vector<std::pair<Stat_, Gene_> >& buffer) {
    Gene_ counter = 0;
    for (Gene_ i = 0, end = buffer.size(); i < end; ++i) {
        const auto cureffect = effects[sanisizer::product_unsafe<std::size_t>(i, stride)];
        if (!std::isnan(cureffect)) {
            auto& current = buffer[counter];
            current.first = cureffect;
            current.second = i;
            ++counter;
        }
    }

    std::sort(
        buffer.begin(),
        buffer.begin() + counter,
        [&](const std::pair<Stat_, Gene_>& left, const std::pair<Stat_, Gene_>& right) -> bool {
            // Sort by decreasing first element, then break ties by increasing second element. 
            if (left.first == right.first) {
                return left.second < right.second;
            } else {
                return left.first > right.first;
            }
        }
    );

    return counter;
}

template<typename Stat_, typename Gene_, typename Rank_>
void compute_min_rank_pairwise(
    const Gene_ ngenes,
    const std::size_t ngroups,
    const Stat_* const effects,
    const std::vector<SummaryBuffers<Stat_, Rank_> >& output,
    const bool preserve_ties, 
    const int threads
) {
    const auto ngroups2 = sanisizer::product_unsafe<std::size_t>(ngroups, ngroups);
    const auto maxrank_placeholder = sanisizer::cast<Rank_>(ngenes); // using the maximum possible rank (i.e., 'ngenes') as the default.

    tatami::parallelize([&](const int, const std::size_t start, const std::size_t length) -> void {
        auto buffer = sanisizer::create<std::vector<std::pair<Stat_, Gene_> > >(ngenes);
        for (decltype(I(start)) g = start, end = start + length; g < end; ++g) { 
            const auto target = output[g].min_rank;
            if (target == NULL) {
                continue;
            }
            std::fill_n(target, ngenes, maxrank_placeholder);

            for (decltype(I(ngroups)) g2 = 0; g2 < ngroups; ++g2) {
                if (g == g2) {
                    continue;
                }
                const auto offset = sanisizer::nd_offset<std::size_t>(g2, ngroups, g);
                const auto used = fill_and_sort_rank_buffer(effects + offset, ngroups2, buffer);

                if (!preserve_ties) {
                    Rank_ counter = 1;
                    for (Gene_ i = 0; i < used; ++i) {
                        auto& current = target[buffer[i].second];
                        if (counter < current) {
                            current = counter;
                        }
                        ++counter;
                    }
                } else {
                    Rank_ counter = 1;
                    Gene_ i = 0;
                    while (i < used) {
                        const auto original = i;
                        const auto val = buffer[i].first;

                        auto& current = target[buffer[i].second];
                        if (counter < current) {
                            current = counter;
                        }

                        while (++i < used && buffer[i].first == val) {
                            auto& current = target[buffer[i].second];
                            if (counter < current) {
                                current = counter;
                            }
                        }

                        counter += i - original;
                    }
                }
            }
        }
    }, ngroups, threads);
}

template<typename Gene_, typename Stat_, typename Rank_>
SummaryBuffers<Stat_, Rank_> fill_summary_results(
    const Gene_ ngenes,
    SummaryResults<Stat_, Rank_>& out, 
    const bool compute_min,
    const bool compute_mean,
    const bool compute_median,
    const bool compute_max,
    const bool compute_min_rank) 
{
    SummaryBuffers<Stat_, Rank_> ptr;
    const auto out_len = sanisizer::cast<typename std::vector<Stat_>::size_type>(ngenes);

    if (compute_min) {
        out.min.resize(out_len
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        ptr.min = out.min.data();
    }
    if (compute_mean) {
        out.mean.resize(out_len
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        ptr.mean = out.mean.data();
    }
    if (compute_median) {
        out.median.resize(out_len
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        ptr.median = out.median.data();
    }
    if (compute_max) {
        out.max.resize(out_len
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        ptr.max = out.max.data();
    }
    if (compute_min_rank) {
        out.min_rank.resize(out_len
#ifdef SCRAN_MARKERS_TEST_INIT
            , SCRAN_MARKERS_TEST_INIT
#endif
        );
        ptr.min_rank = out.min_rank.data();
    }

    return ptr;
}

template<typename Gene_, typename Stat_, typename Rank_>
std::vector<SummaryBuffers<Stat_, Rank_> > fill_summary_results(
    Gene_ ngenes,
    const std::size_t ngroups,
    std::vector<SummaryResults<Stat_, Rank_> >& outputs, 
    const bool compute_min,
    const bool compute_mean,
    const bool compute_median,
    const bool compute_max,
    const bool compute_min_rank) 
{
    sanisizer::resize(outputs, ngroups);
    std::vector<SummaryBuffers<Stat_, Rank_> > ptrs;
    ptrs.reserve(ngroups);
    for (decltype(I(ngroups)) g = 0; g < ngroups; ++g) {
        ptrs.emplace_back(fill_summary_results(ngenes, outputs[g], compute_min, compute_mean, compute_median, compute_max, compute_min_rank));
    }
    return ptrs;
}

}
/**
 * @endcond
 */

}

#endif
