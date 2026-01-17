#ifndef SCRAN_AGGREGATE_AGGREGATE_ACROSS_GENES_HPP
#define SCRAN_AGGREGATE_AGGREGATE_ACROSS_GENES_HPP

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file aggregate_across_genes.hpp
 * @brief Aggregate expression values across genes.
 */

namespace scran_aggregate {

/**
 * @brief Options for `aggregate_across_genes()`.
 */
struct AggregateAcrossGenesOptions {
    /**
     * Number of threads to use. 
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;

    /**
     * Whether to compute the average expression within each gene set.
     * If the gene set contains weights, a weighted average is computed.
     */
    bool average = false;
};

/**
 * @brief Buffers for `aggregate_across_genes()`.
 * @tparam Sum_ Floating-point type of the sum/mean.
 */
template <typename Sum_>
struct AggregateAcrossGenesBuffers {
    /**
     * Vector of length equal to the number of gene sets.
     * Each element is a pointer to an array of length equal to the number of cells,
     * to be filled with the (weighted) sum/mean of expression values for each gene set.
     */
    std::vector<Sum_*> sum;
};

/**
 * @brief Results of `aggregate_across_genes()`.
 * @tparam Sum_ Floating-point type of the sum/mean.
 */
template <typename Sum_>
struct AggregateAcrossGenesResults {
    /**
     * Vector of length equal to the number of gene sets.
     * Each inner vector is of length equal to the number of cells.
     * Each entry contains the (weighted) sum/mean of expression values across all genes in the corresponding gene set.
     */
    std::vector<std::vector<Sum_> > sum;
};

/**
 * @cond
 */
template<typename Index_, typename Gene_, typename Weight_>
std::vector<Gene_> create_subset(const std::vector<std::tuple<std::size_t, const Gene_*, const Weight_*> >& gene_sets, const Index_ nrow) {
    std::unordered_set<Gene_> of_interest;
    for (const auto& set : gene_sets) {
        const auto set_size = std::get<0>(set);
        const auto set_genes = std::get<1>(set);
        of_interest.insert(set_genes, set_genes + set_size);
    }

    std::vector<Index_> subset(of_interest.begin(), of_interest.end());
    if (!subset.empty()) {
        std::sort(subset.begin(), subset.end());
        if (subset.front() < 0 || subset.back() >= nrow) {
            throw std::runtime_error("set indices are out of range");
        }
    }

    return subset;
}

template<typename Index_>
std::pair<std::vector<Index_>, Index_> create_subset_mapping(const std::vector<Index_>& subset) {
    const Index_ offset = subset.front();
    const Index_ span = subset.back() - offset + 1;
    auto mapping = tatami::create_container_of_Index_size<std::vector<Index_> >(span);
    const auto nsubs = subset.size();
    for (I<decltype(nsubs)> i = 0; i < nsubs; ++i) {
        mapping[subset[i] - offset] = i;
    }
    return std::make_pair(std::move(mapping), offset);
}

template<typename Data_, typename Index_, typename Gene_, typename Weight_, typename Sum_>
void aggregate_across_genes_by_column(
    const tatami::Matrix<Data_, Index_>& p,
    const std::vector<std::tuple<std::size_t, const Gene_*, const Weight_*> >& gene_sets,
    const AggregateAcrossGenesBuffers<Sum_>& buffers,
    const AggregateAcrossGenesOptions& options)
{
    // Identifying the subset of rows that actually need to be extracted.
    const tatami::VectorPtr<Index_> subset_of_interest = std::make_shared<std::vector<Index_> >(create_subset<Index_>(gene_sets, p.nrow()));
    const auto& subset = *subset_of_interest;
    const Index_ nsubs = subset.size();

    // Creating a mapping back to the gene indices in the subset.
    const auto num_sets = gene_sets.size();
    auto remapping = sanisizer::create<std::vector<std::pair<std::vector<Index_>, const Weight_*> > >(num_sets);
    if (nsubs) {
        const auto sub_mapping = create_subset_mapping(subset);
        const auto& mapping = sub_mapping.first;
        const Gene_ offset = sub_mapping.second;

        for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
            const auto& set = gene_sets[s];
            const auto set_size = std::get<0>(set);
            const auto set_genes = std::get<1>(set);

            auto& remapped = remapping[s].first;
            remapped.reserve(set_size);
            for (I<decltype(set_size)> g = 0; g < set_size; ++g) {
                remapped.push_back(mapping[set_genes[g] - offset]);
            }
            remapping[s].second = std::get<2>(set);
        }
    }

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        // We extract as sparse even if it is dense, as it's just
        // easier to index from a dense vector.
        auto ext = tatami::consecutive_extractor<false>(p, false, start, length, subset_of_interest);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(nsubs);

        for (Index_ x = start, end = start + length; x < end; ++x) {
            const auto ptr = ext->fetch(vbuffer.data());
            for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
                const auto& set = remapping[s];

                Sum_ value = 0;
                if (set.second) {
                    for (I<decltype(set.first.size())> i = 0, send = set.first.size(); i < send; ++i) {
                        value += ptr[set.first[i]] * set.second[i];
                    }
                } else {
                    for (const auto ix : set.first) {
                        value += ptr[ix];
                    }
                }

                buffers.sum[s][x] = value;
            }
        }

    }, p.ncol(), options.num_threads);
}

template<typename Data_, typename Index_, typename Gene_, typename Weight_, typename Sum_>
void aggregate_across_genes_by_row(
    const tatami::Matrix<Data_, Index_>& p,
    const std::vector<std::tuple<std::size_t, const Gene_*, const Weight_*> >& gene_sets,
    const AggregateAcrossGenesBuffers<Sum_>& buffers,
    const AggregateAcrossGenesOptions& options)
{
    // Identifying the subset of rows that actually need to be extracted.
    const auto subset = create_subset<Index_>(gene_sets, p.nrow());
    const Index_ nsubs = subset.size();
    const auto sub_oracle = std::make_shared<tatami::FixedViewOracle<Index_> >(subset.data(), nsubs);

    const auto num_sets = gene_sets.size();
    auto remapping = tatami::create_container_of_Index_size<std::vector<std::vector<std::pair<std::size_t, Weight_> > > >(nsubs);
    if (nsubs) {
        const auto sub_mapping = create_subset_mapping(subset);
        const auto& mapping = sub_mapping.first;
        const Gene_ offset = sub_mapping.second;

        for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
            const auto& set = gene_sets[s];
            const auto set_size = std::get<0>(set);
            const auto set_genes = std::get<1>(set);
            const auto set_weights = std::get<2>(set);

            if (set_weights) {
                for (I<decltype(set_size)> g = 0; g < set_size; ++g) {
                    remapping[mapping[set_genes[g] - offset]].emplace_back(s, set_weights[g]); 
                }
            } else {
                for (I<decltype(set_size)> g = 0; g < set_size; ++g) {
                    remapping[mapping[set_genes[g] - offset]].emplace_back(s, 1);
                }
            }
        }
    }

    tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        auto get_sum = [&](Index_ i) -> Sum_* { return buffers.sum[i]; };
        tatami_stats::LocalOutputBuffers<Sum_, I<decltype(get_sum)>> local_sums(t, num_sets, start, length, std::move(get_sum));

        if (p.sparse()) {
            auto ext = tatami::new_extractor<true, true>(p, true, sub_oracle, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(length);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(length);

            for (Index_ sub = 0; sub < nsubs; ++sub) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());

                for (const auto& sw : remapping[sub]) {
                    const auto outptr = local_sums.data(sw.first);
                    const auto wt = sw.second;
                    for (Index_ c = 0; c < range.number; ++c) {
                        outptr[range.index[c] - start] += range.value[c] * wt;
                    }
                }
            }

        } else {
            auto ext = tatami::new_extractor<false, true>(&p, true, sub_oracle, start, length);
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(length);

            for (Index_ sub = 0; sub < nsubs; ++sub) {
                const auto ptr = ext->fetch(vbuffer.data());
                for (const auto& sw : remapping[sub]) {
                    const auto outptr = local_sums.data(sw.first);
                    const auto wt = sw.second;
                    for (Index_ cell = 0; cell < length; ++cell) {
                        outptr[cell] += ptr[cell] * wt;
                    }
                }
            }
        }

        local_sums.transfer();
    }, p.ncol(), options.num_threads);
}
/**
 * @endcond
 */

/**
 * Aggregate expression values across gene sets for each cell.
 * This involves computing the sum/mean of expression values for any number of gene sets.
 * The aim is to quantify the activity of signatures, pathways or regulons in each cell.
 * Each gene in each set can also be weighted based on any _a priori_ assumptions of their importance to the corresponding pathway.
 *
 * @tparam Data_ Type of data in the input matrix, should be numeric.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Gene_ Integer type of the indices of genes in each set.
 * @tparam Weight_ Floating-point type of the weights of genes in each set.
 * @tparam Sum_ Floating-point type of the sum.
 *
 * @param input Matrix of expression values where rows are features and columns are cells.
 * This is usually normalized and possibly log-transformed, but the exact nature of the values depends on the application.
 * @param gene_sets Vector of gene sets.
 * Each tuple corresponds to a set and contains (i) the number of genes in the set,
 * (ii) a pointer to the row indices of the genes in the set, and
 * (iii) a pointer to the weights of the genes in the set.
 * The weight pointer may be `NULL`, in which case all weights are set to 1.
 * @param[out] buffers Collection of buffers in which to store the sum/mean for each gene set and cell.
 * @param options Further options.
 */
template<typename Data_, typename Index_, typename Gene_, typename Weight_, typename Sum_>
void aggregate_across_genes(
    const tatami::Matrix<Data_, Index_>& input,
    const std::vector<std::tuple<std::size_t, const Gene_*, const Weight_*> >& gene_sets,
    const AggregateAcrossGenesBuffers<Sum_>& buffers,
    const AggregateAcrossGenesOptions& options)
{
    if (input.prefer_rows()) {
        aggregate_across_genes_by_row(input, gene_sets, buffers, options);
    } else {
        aggregate_across_genes_by_column(input, gene_sets, buffers, options);
    }

    if (options.average) {
        const auto nsets = gene_sets.size();
        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            const Index_ NC = input.ncol();
            for (Index_ s = start, end = start + length; s < end; ++s) {
                const auto& set = gene_sets[s];
                const auto set_size = std::get<0>(set);

                Sum_ denom = 0;
                const auto set_weights = std::get<2>(set);
                if (set_weights) {
                    denom = std::accumulate(set_weights, set_weights + set_size, static_cast<Sum_>(0)); 
                } else {
                    denom = set_size;
                }

                const auto current = buffers.sum[s];
                for (Index_ c = 0; c < NC; ++c) {
                    current[c] /= denom;
                }
            }
        }, nsets, options.num_threads);
    }
} 

/**
 * Overload of `aggregate_across_genes()` that allocates memory for the results.
 *
 * @tparam Sum_ Floating-point type of the sum.
 * @tparam Data_ Type of data in the input matrix, should be numeric.
 * @tparam Index_ Integer type of index in the input matrix.
 * @tparam Gene_ Integer type of the indices of genes in each set.
 * @tparam Weight_ Floating-point type of the weights of genes in each set.
 *
 * @param input Matrix of expression values where rows are features and columns are cells.
 * @param gene_sets Vector of gene sets.
 * Each tuple corresponds to a set and contains (i) the number of genes in the set,
 * (ii) a pointer to the row indices of the genes in the set, and
 * (iii) a pointer to the weights of the genes in the set.
 * The weight pointer may be `NULL`, in which case all weights are set to 1.
 * @param options Further options.
 *
 * @return Results of the aggregation.
 */
template<typename Sum_ = double, typename Data_, typename Index_, typename Gene_, typename Weight_>
AggregateAcrossGenesResults<Sum_> aggregate_across_genes(
    const tatami::Matrix<Data_, Index_>& input,
    const std::vector<std::tuple<std::size_t, const Gene_*, const Weight_*> >& gene_sets,
    const AggregateAcrossGenesOptions& options)
{
    AggregateAcrossGenesResults<Sum_> output;
    AggregateAcrossGenesBuffers<Sum_> buffers;

    const Index_ NC = input.ncol();
    const auto nsets = gene_sets.size();
    sanisizer::resize(output.sum, nsets);
    sanisizer::resize(buffers.sum, nsets);

    for (I<decltype(nsets)> s = 0; s < nsets; ++s) {
        tatami::resize_container_to_Index_size(
            output.sum[s],
            NC
#ifdef SCRAN_AGGREGATE_TEST_INIT
            , SCRAN_AGGREGATE_TEST_INIT
#endif
        );
        buffers.sum[s] = output.sum[s].data();
    }

    aggregate_across_genes(input, gene_sets, buffers, options);
    return output;
} 

}

#endif
