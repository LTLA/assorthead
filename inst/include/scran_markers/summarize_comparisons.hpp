#ifndef SCRAN_MARKERS_SUMMARIZE_COMPARISONS_HPP
#define SCRAN_MARKERS_SUMMARIZE_COMPARISONS_HPP

#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <cstddef>
#include <optional>
#include <cassert>

#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"
#include "scran_blocks/scran_blocks.hpp"

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
     * Optional vector of pointers to arrays of length equal to the number of genes.
     * Each pointer corresponds to a quantile probability from `SummarizeEffects::compute_quantiles`.
     * The array is to be filled with the corresponding quantile of effect sizes for each gene.
     *
     * If unset, no quantiles will be computed computed.
     * If set, each pointer should be non-`NULL`, and the length of the vector should be equal to:
     *
     * - the vector stored in `SummarizeEffectsOptions::compute_quantiles`, if the `SummaryBuffers` object is to be passed to `summarize_effects()`.
     *   If no vector was stored, `quantiles` is ignored.
     * - the vector stored in `ScoreMarkersSummaryOptions::compute_summary_quantiles`, if the `SummaryBuffers` object is to be passed to `score_markers_summary()`.
     *   If no vector was stored, `quantiles` is ignored.
     */ 
    std::optional<std::vector<Stat_*> > quantiles;

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
     * Optional vector of vectors of length equal to the number of genes.
     * Each inner vector corresponds to a quantile probability from `SummarizeEffects::compute_quantiles`.
     * Each entry of the inner vector contains the corresponding quantile of effect sizes for each gene.
     * If not set, no quantiles were computed.
     */ 
    std::optional<std::vector<std::vector<Stat_> > > quantiles;

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

inline void validate_quantiles(const std::optional<std::vector<double> >& probs) {
    if (!probs.has_value()) { 
        return;
    }

    const auto val = probs->front();
    if (val < 0 || val > 1) {
        throw std::runtime_error("quantile probabilities should be in [0, 1]");
    }

    const auto nprobs = probs->size();
    for (I<decltype(nprobs)> i = 1; i < nprobs; ++i) {
        const auto val = (*probs)[i];
        if (val < 0 || val > 1) {
            throw std::runtime_error("quantile probabilities should be in [0, 1]");
        }
        if (val < (*probs)[i - 1]) {
            throw std::runtime_error("quantile probabilities should be sorted");
        }
    }
}

template<typename Stat_, class Iterator_>
class MultipleQuantiles {
public:
    MultipleQuantiles(const std::vector<double>& probs, const std::size_t max_len) :
        my_stacks(sanisizer::cast<I<decltype(my_stacks.size())> >(max_len)),
        my_probs(&probs)
    {
        sanisizer::can_ptrdiff<Iterator_>(max_len);
    }

    struct Details { 
        std::vector<std::size_t> index;
        std::vector<double> lower_fraction;
        std::vector<double> upper_fraction;
    };

private:
    std::vector<std::optional<Details> > my_stacks;
    const std::vector<double>* my_probs; // avoid unnecessary copies of the quantile probabilities.

private:
    Details& initialize_stack(const std::size_t len) {
        // len is guaranteed to be > 1 from summarize_comparisons(). 
        assert(len > 1);
        auto& raw_stack = my_stacks[len - 1];
        if (raw_stack.has_value()) {
            return *raw_stack;
        }

        raw_stack.emplace();
        auto& stack = *raw_stack;
        const auto nprobs = my_probs->size();
        stack.index.reserve(nprobs);
        stack.lower_fraction.reserve(nprobs);
        stack.upper_fraction.reserve(nprobs);

        for (const auto prob : *my_probs) {
            const double frac = static_cast<double>(len - 1) * static_cast<double>(prob);
            const double base = std::floor(frac);
            stack.index.push_back(base); // cast is known-safe if can_ptrdiff passes and 0 <= quantile <= 1.
            stack.upper_fraction.push_back(frac - base);
            stack.lower_fraction.push_back(static_cast<double>(1) - stack.upper_fraction.back());
        }

        return stack;
    }

public:
    template<class OutputFun_>
    void compute(const std::size_t len, Iterator_ begin, Iterator_ end, OutputFun_ output) {
        // len is assumed to be > 1 and equal to 'end - begin'.
        // We just accept it as an argument to avoid recomputing it.
        assert(len > 1);
        assert(len == static_cast<std::size_t>(end - begin));
        auto& stack = initialize_stack(len);

        // Here, the assumption is that we're computing quantiles by increasing probability.
        // We keep track of how much we've already sorted so that we don't have to call
        // nth_element for a given index if we already computed for a previous probability.
        // This allows us to avoid near-redundant sorts for similar probabilities.
        std::size_t sorted_up_to = 0;

        const auto nprobs = stack.index.size();
        for (I<decltype(nprobs)> p = 0; p < nprobs; ++p) {
            const auto curindex = stack.index[p];
            const auto curlower = stack.lower_fraction[p];
            const auto curupper = stack.upper_fraction[p];

            const auto target = begin + curindex;
            if (curindex >= sorted_up_to) { // avoid re-searching for the nth element if we already found what we wanted for a lower probability.
                std::nth_element(begin + sorted_up_to, target, end);
                sorted_up_to = curindex + 1;
            }
            const auto lower_val = *target;

            if (curupper != 0) {
                const auto curindex_p1 = curindex + 1;
                const auto target_p1 = begin + curindex_p1;
                if (curindex_p1 >= sorted_up_to) {
                    // Basically mimics nth_element(target_p1, target_p1, end).
                    std::swap(*target_p1, *std::min_element(target_p1, end));
                    sorted_up_to = curindex_p1 + 1;
                }
                const auto upper_val = *target_p1;
                output(p, lower_val * curlower + upper_val * curupper);
            } else {
                output(p, lower_val);
            }
        }
    }
};

template<typename Stat_>
using MaybeMultipleQuantiles = std::optional<MultipleQuantiles<Stat_, Stat_*> >;

template<typename Stat_>
MaybeMultipleQuantiles<Stat_> setup_multiple_quantiles(const std::optional<std::vector<double> >& requested, const std::size_t ngroups) {
    MaybeMultipleQuantiles<Stat_> output;
    if (requested.has_value()) {
        output.emplace(*requested, ngroups);
    }
    return output;
}

template<typename Stat_, typename Gene_, typename Rank_>
void summarize_comparisons(
    const std::size_t ngroups,
    const Stat_* const effects,
    const std::size_t group,
    const Gene_ gene,
    const SummaryBuffers<Stat_, Rank_>& output,
    MaybeMultipleQuantiles<Stat_>& quantile_calculators,
    std::vector<Stat_>& buffer
) {
    // Ignoring the self comparison and pruning out NaNs.
    std::size_t ncomps = 0;
    for (I<decltype(ngroups)> r = 0; r < ngroups; ++r) {
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
        if (output.quantiles.has_value()) {
            for (const auto& quan : *(output.quantiles)) {
                quan[gene] = val;
            }
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
        // This following calculations mutate the buffer, so we put this last to avoid surprises.
        if (output.median) {
            output.median[gene] = tatami_stats::medians::direct(ebegin, ncomps, /* skip_nan = */ false); 
        }
        if (output.quantiles.has_value()) {
            quantile_calculators->compute(
                ncomps,
                ebegin,
                elast, 
                [&](const std::size_t i, const Stat_ value) -> void {
                    (*output.quantiles)[i][gene] = value;
                }
            );
        }
    }
}

template<typename Gene_, typename Stat_, typename Rank_>
void summarize_comparisons(
    const Gene_ ngenes,
    const std::size_t ngroups,
    const Stat_* const effects,
    const std::optional<std::vector<double> >& compute_quantiles,
    const std::vector<SummaryBuffers<Stat_, Rank_> >& output,
    const int threads
) {
    tatami::parallelize([&](const int, const Gene_ start, const Gene_ length) -> void {
        auto summary_qcalcs = setup_multiple_quantiles<Stat_>(compute_quantiles, ngroups);
        auto buffer = sanisizer::create<std::vector<Stat_> >(ngroups);

        for (Gene_ gene = start, end = start + length; gene < end; ++gene) {
            for (I<decltype(ngroups)> l = 0; l < ngroups; ++l) {
                const auto current_effects = effects + sanisizer::nd_offset<std::size_t>(0, ngroups, l, ngroups, gene);
                summarize_comparisons(ngroups, current_effects, l, gene, output[l], summary_qcalcs, buffer);
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
        for (I<decltype(start)> g = start, end = start + length; g < end; ++g) { 
            const auto target = output[g].min_rank;
            if (target == NULL) {
                continue;
            }
            std::fill_n(target, ngenes, maxrank_placeholder);

            for (I<decltype(ngroups)> g2 = 0; g2 < ngroups; ++g2) {
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
    const std::optional<std::vector<double> >& compute_quantiles,
    const bool compute_min_rank
) {
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

    if (compute_quantiles.has_value()) {
        out.quantiles.emplace();
        ptr.quantiles.emplace();
        out.quantiles->reserve(compute_quantiles->size());
        ptr.quantiles->reserve(compute_quantiles->size());
        for ([[maybe_unused]] const auto quan : *compute_quantiles) {
            out.quantiles->emplace_back(out_len
#ifdef SCRAN_MARKERS_TEST_INIT
                , SCRAN_MARKERS_TEST_INIT
#endif
            );
            ptr.quantiles->push_back(out.quantiles->back().data());
        }
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
    const std::optional<std::vector<double> >& compute_quantiles,
    const bool compute_min_rank
) {
    sanisizer::resize(outputs, ngroups);
    std::vector<SummaryBuffers<Stat_, Rank_> > ptrs;
    ptrs.reserve(ngroups);
    for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
        ptrs.emplace_back(
            fill_summary_results(
                ngenes,
                outputs[g],
                compute_min,
                compute_mean,
                compute_median,
                compute_max,
                compute_quantiles,
                compute_min_rank
            )
        );
    }
    return ptrs;
}

}
/**
 * @endcond
 */

}

#endif
