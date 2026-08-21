#ifndef SCRAN_AGGREGATE_AGGREGATE_ACROSS_GENES_HPP
#define SCRAN_AGGREGATE_AGGREGATE_ACROSS_GENES_HPP

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "quickstats/quickstats.hpp"
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
 * @brief Gene set to use in `aggregate_across_genes()`.
 * @tparam Gene_ Integer type of the indices of genes in each set.
 * @tparam Weight_ Floating-point type of the weights of genes in each set.
 */
template<typename Gene_, typename Weight_>
struct AggregateAcrossGenesSet {
    /**
     * Default constructor.
     */
    AggregateAcrossGenesSet() = default;

    /**
     * @param number Number of genes in the set.
     * @param gene Pointer to an array of length `number`, containing the row indices of the genes in the set.
     * Each entry should be a non-negative integer that is less than `input.nrow()` in `aggregate_across_genes()`.
     * Values should be unique.
     * @param weight Pointer to an array of length `number`, containing the weight for each gene in the set.
     * Each entry corresponds to an entry of `gene` and specifies the weight for that gene.
     * For unweighted sets, this should be set to NULL.
     */
    AggregateAcrossGenesSet(std::size_t number, const Gene_* gene, const Weight_* weight) :
        number(number),
        gene(gene),
        weight(weight)
    {}

    /**
     * Number of genes in the set.
     */
    std::size_t number = 0;

    /**
     * Pointer to an array of length `number`, containing the unique row indices of all genes in the set.
     */
    const Gene_* gene = NULL;

    /**
     * Pointer to an array of length `number`, containing the weights for all genes in the set.
     * This may also be set to NULL if no weights are available.
     */
    const Weight_* weight = NULL;
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
template<typename Data_, typename Index_, typename Gene_, typename Weight_, typename Sum_>
void aggregate_across_genes_by_column(
    const tatami::Matrix<Data_, Index_>& p,
    const std::vector<AggregateAcrossGenesSet<Gene_, Weight_> >& gene_sets,
    const AggregateAcrossGenesBuffers<Sum_>& buffers,
    const AggregateAcrossGenesOptions& options
) {
    const auto NR = p.nrow();
    const auto num_sets = gene_sets.size();

    // Identifying the subset of rows that actually need to be extracted.
    std::vector<Index_> subset;
    {
        auto occupancy = tatami::create_container_of_Index_size<std::vector<char> >(NR);
        Index_ used = 0;
        for (const auto& set : gene_sets) {
            for (std::size_t i = 0; i < set.number; ++i) {
                const auto g = set.gene[i];
                if (g < 0 || sanisizer::is_greater_than_or_equal(g, NR)) {
                    throw std::runtime_error("set indices are out of range");
                }
                if (!occupancy[g]) {
                    ++used;
                    occupancy[g] = true;
                }
            }
        }

        subset.reserve(used);
        for (Index_ r = 0; r < NR; ++r) {
            if (occupancy[r]) {
                subset.push_back(r);
            }
        }
    }

    // Remapping the row indices to the subset of genes across all sets.
    // However, we only do this if the subset of genes does not consist of all genes.
    // This choice requires some care to produce an alternative to 'gene_sets' with the remapped indices.
    const std::vector<AggregateAcrossGenesSet<Gene_, Weight_> >* gene_sets_ptr = &gene_sets;
    std::optional<std::vector<AggregateAcrossGenesSet<Gene_, Weight_> > > remapped_gene_sets;

    class RemappedGeneSetLiberator {
    public:
        RemappedGeneSetLiberator(std::optional<std::vector<AggregateAcrossGenesSet<Gene_, Weight_> > >& host) : my_host(host) {}
        ~RemappedGeneSetLiberator() {
            // The only purpose of this class is to wipe out the dynamically allocated memory for the remapped indices.
            if (my_host.has_value()) {
                for (auto& rset : *my_host) {
                    if (rset.gene) {
                        delete [] rset.gene;
                    }
                }
            }
        }
    private:
        std::optional<std::vector<AggregateAcrossGenesSet<Gene_, Weight_> > >& my_host;
    };
    RemappedGeneSetLiberator lib(remapped_gene_sets);

    const auto nsubs = subset.size();
    if (nsubs) {
        const Index_ offset = subset.front();
        const Index_ span = subset.back() - offset + 1;

        if (offset || !sanisizer::is_equal(span, nsubs)) { // i.e., not a consecutive block starting at zero.
            auto mapping = tatami::create_container_of_Index_size<std::vector<Index_> >(span);
            for (I<decltype(nsubs)> i = 0; i < nsubs; ++i) {
                mapping[subset[i] - offset] = i;
            }

            remapped_gene_sets.emplace(num_sets); // type is already correct, no need for sanisizer protection.
            gene_sets_ptr = &(*remapped_gene_sets);

            for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
                const auto& set = gene_sets[s];
                auto& remapped = (*remapped_gene_sets)[s];
                remapped.number = set.number;
                remapped.weight = set.weight;

                const auto rgene = new Gene_ [set.number]; // set.number is already size_t, no need to cast.
                remapped.gene = rgene; // set it here ASAP to avoid memory leak if there are any exceptions.
                for (std::size_t g = 0; g < set.number; ++g) {
                    rgene[g] = mapping[set.gene[g] - offset];
                }
            }
        }
    }

    tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
        // We extract as dense even if it is sparse, as it's just easier to index from a dense vector.
        auto ext = [&]{
            if (nsubs) {
                const Index_ offset = subset.front();
                if (offset == 0 && sanisizer::is_equal(nsubs, NR)) {
                    return tatami::consecutive_extractor<false>(p, false, start, length);
                }
                const Index_ span = subset.back() - offset + 1;
                if (sanisizer::is_equal(span, nsubs)) {
                    return tatami::consecutive_extractor<false>(p, false, start, length, offset, span);
                }
            }
            return tatami::consecutive_extractor<false>(p, false, start, length, tatami::VectorPtr<Index_>(tatami::VectorPtr<Index_>{}, &subset));
        }();
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(nsubs);

        // Using a pairwise sum for a more-or-less free improvement to accuracy.
        quickstats::PairwiseSumWorkspace<Sum_> pswrk;
        quickstats::PairwiseSumOptions psopt;

        for (Index_ x = start, end = start + length; x < end; ++x) {
            const auto ptr = ext->fetch(vbuffer.data());
            for (std::size_t s = 0; s < num_sets; ++s) {
                const auto& set = (*gene_sets_ptr)[s];

                if (set.weight) {
                    buffers.sum[s][x] = quickstats::pairwise_sum_abstract(
                        set.number,
                        [&](std::size_t i) -> Sum_ {
                            return ptr[set.gene[i]] * set.weight[i];
                        },
                        pswrk,
                        psopt
                    );
                } else {
                    buffers.sum[s][x] = quickstats::pairwise_sum_abstract(
                        set.number, 
                        [&](std::size_t i) -> Sum_ {
                            return ptr[set.gene[i]];
                        },
                        pswrk,
                        psopt
                    );
                }
            }
        }

    }, p.ncol(), options.num_threads);
}

template<typename Data_, typename Index_, typename Gene_, typename Weight_, typename Sum_>
void aggregate_across_genes_by_row(
    const tatami::Matrix<Data_, Index_>& p,
    const std::vector<AggregateAcrossGenesSet<Gene_, Weight_> >& gene_sets,
    const AggregateAcrossGenesBuffers<Sum_>& buffers,
    const AggregateAcrossGenesOptions& options
) {
    const auto NR = p.nrow();
    const auto NC = p.ncol();
    const auto num_sets = gene_sets.size();
    typedef I<decltype(num_sets)> SetIndex;

    // Identifying the subset of rows that actually need to be extracted.
    std::vector<Index_> subset;
    std::vector<std::pair<std::vector<SetIndex>, std::vector<Weight_> > > revmapping; 
    {
        auto occupancy = tatami::create_container_of_Index_size<std::vector<SetIndex> >(NR);
        Index_ used = 0;
        for (const auto& set : gene_sets) {
            for (std::size_t i = 0; i < set.number; ++i) {
                const auto g = set.gene[i];
                if (g < 0 || sanisizer::is_greater_than_or_equal(g, NR)) {
                    throw std::runtime_error("set indices are out of range");
                }
                used += (occupancy[g] == 0);
                occupancy[g] += 1; 
            }
        }

        subset.reserve(used);
        tatami::resize_container_to_Index_size(revmapping, used);
        for (Index_ r = 0; r < NR; ++r) {
            if (occupancy[r]) {
                auto& revmap_dest = revmapping[subset.size()];
                revmap_dest.first.reserve(occupancy[r]);
                revmap_dest.second.reserve(occupancy[r]);
                subset.push_back(r);
            }
        }
    }

    // Reverse the mapping to get genes->sets.
    const Index_ nsubs = subset.size();
    if (nsubs) {
        const Index_ offset = subset.front();
        const Index_ span = subset.back() - offset + 1;

        if (!sanisizer::is_equal(span, nsubs)) { // i.e., not a consecutive block. 
            auto mapping = tatami::create_container_of_Index_size<std::vector<Index_> >(span);
            for (I<decltype(nsubs)> i = 0; i < nsubs; ++i) {
                mapping[subset[i] - offset] = i;
            }

            for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
                const auto& set = gene_sets[s];
                if (set.weight) {
                    for (std::size_t g = 0; g < set.number; ++g) {
                        auto& dest = revmapping[mapping[set.gene[g] - offset]];
                        dest.first.push_back(s);
                        dest.second.push_back(set.weight[g]);
                    }
                } else {
                    for (std::size_t g = 0; g < set.number; ++g) {
                        auto& dest = revmapping[mapping[set.gene[g] - offset]];
                        dest.first.push_back(s);
                        dest.second.push_back(1);
                    }
                }
            }

        } else {
            for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
                const auto& set = gene_sets[s];
                if (set.weight) {
                    for (std::size_t g = 0; g < set.number; ++g) {
                        auto& dest = revmapping[set.gene[g] - offset];
                        dest.first.push_back(s);
                        dest.second.push_back(set.weight[g]);
                    }
                } else {
                    for (std::size_t g = 0; g < set.number; ++g) {
                        auto& dest = revmapping[set.gene[g] - offset];
                        dest.first.push_back(s);
                        dest.second.push_back(1);
                    }
                }
            }
        }
    }

    for (I<decltype(num_sets)> s = 0; s < num_sets; ++s) {
        std::fill_n(buffers.sum[s], NC, 0);
    }

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<std::vector<Sum_> > > > > per_thread_sums;
    if (do_parallel) {
        per_thread_sums.emplace(sanisizer::cast<I<decltype(per_thread_sums->size())> >(options.num_threads - 1));
    }

    const bool is_sparse = p.is_sparse();
    const auto nused = tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        auto sub_oracle = std::make_shared<tatami::FixedViewOracle<Index_> >(subset.data() + start, length);
        std::optional<std::vector<std::vector<Sum_> > > tmp_sums;
        if (t > 0) {
            tmp_sums.emplace(sanisizer::cast<I<decltype(tmp_sums->size())> >(num_sets));
        }

        auto get_output_ptr = [&](SetIndex curset) -> Sum_* {
            if (t == 0) {
                return buffers.sum[curset];
            }
            // Only allocate each set's memory if we actually need it in the current thread.
            // Recall that we split the gene subset across multiple threads.
            // A thread could get a sub-subset where certain gene sets are not represented.
            // In such cases, there's no point allocating temporary storage for those gene sets in that thread.
            if ((*tmp_sums)[curset].empty()) {
                tatami::resize_container_to_Index_size((*tmp_sums)[curset], NC);
            }
            return (*tmp_sums)[curset].data();
        };

        if (is_sparse){
            auto ext = tatami::new_extractor<true, true>(p, true, std::move(sub_oracle));
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            for (Index_ g = 0; g < length; ++g) {
                const auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                const auto& cursets = revmapping[start + g];
                const auto ncursets = cursets.first.size();

                for (I<decltype(ncursets)> s = 0; s < ncursets; ++s) {
                    const auto curset = cursets.first[s];
                    const auto curw = cursets.second[s];
                    const auto outptr = get_output_ptr(curset);
                    if (curw != 1) {
                        for (Index_ i = 0; i < range.number; ++i) {
                            outptr[range.index[i]] += range.value[i] * curw;
                        }
                    } else {
                        for (Index_ i = 0; i < range.number; ++i) {
                            outptr[range.index[i]] += range.value[i];
                        }
                    }
                }
            }

        } else {
            auto ext = tatami::new_extractor<false, true>(p, true, std::move(sub_oracle));
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<Data_> >(NC);

            for (Index_ g = 0; g < length; ++g) {
                const auto ptr = ext->fetch(vbuffer.data());
                const auto& cursets = revmapping[start + g];
                const auto ncursets = cursets.first.size();

                for (I<decltype(ncursets)> s = 0; s < ncursets; ++s) {
                    const auto curset = cursets.first[s];
                    const auto curw = cursets.second[s];
                    const auto outptr = get_output_ptr(curset);
                    if (curw != 1) {
                        for (Index_ c = 0; c < NC; ++c) {
                            outptr[c] += ptr[c] * curw;
                        }
                    } else {
                        for (Index_ c = 0; c < NC; ++c) {
                            outptr[c] += ptr[c];
                        }
                    }
                }
            }
        }

        if (t > 0) {
            (*per_thread_sums)[t - 1] = std::move(tmp_sums);
        }
    }, static_cast<Index_>(nsubs), options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < nused; ++u) { 
            const auto& thread_sums = *((*per_thread_sums)[u - 1]);
            for (SetIndex s = 0; s < num_sets; ++s) {
                const auto& thread_sum = thread_sums[s];
                if (thread_sum.empty()) {
                    continue;                    
                }
                const auto outptr = buffers.sum[s];
                for (Index_ c = 0; c < NC; ++c) {
                    outptr[c] += thread_sum[c];
                }
            }
        }
    }
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
 * @param gene_sets Vector of (possibly weighted) gene sets.
 * @param[out] buffers Collection of buffers in which to store the sum/mean for each gene set and cell.
 * @param options Further options.
 */
template<typename Data_, typename Index_, typename Gene_, typename Weight_, typename Sum_>
void aggregate_across_genes(
    const tatami::Matrix<Data_, Index_>& input,
    const std::vector<AggregateAcrossGenesSet<Gene_, Weight_> >& gene_sets,
    const AggregateAcrossGenesBuffers<Sum_>& buffers,
    const AggregateAcrossGenesOptions& options
) {
    if (input.prefer_rows()) {
        aggregate_across_genes_by_row(input, gene_sets, buffers, options);
    } else {
        aggregate_across_genes_by_column(input, gene_sets, buffers, options);
    }

    if (options.average) {
        const auto nsets = gene_sets.size();
        tatami::parallelize([&](const int, const Index_ start, const Index_ length) -> void {
            const Index_ NC = input.ncol();
            quickstats::PairwiseSumWorkspace<Sum_> pswrk;
            quickstats::PairwiseSumOptions psopt;

            for (Index_ s = start, end = start + length; s < end; ++s) {
                const auto& set = gene_sets[s];
                Sum_ denom = 0;
                if (set.weight) {
                    denom = quickstats::pairwise_sum(set.number, set.weight, pswrk, psopt); 
                } else {
                    denom = set.number;
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
 * @param gene_sets Vector of (possibly weighted) gene sets.
 * @param options Further options.
 *
 * @return Results of the aggregation.
 */
template<typename Sum_ = double, typename Data_, typename Index_, typename Gene_, typename Weight_>
AggregateAcrossGenesResults<Sum_> aggregate_across_genes(
    const tatami::Matrix<Data_, Index_>& input,
    const std::vector<AggregateAcrossGenesSet<Gene_, Weight_> >& gene_sets,
    const AggregateAcrossGenesOptions& options
) {
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
