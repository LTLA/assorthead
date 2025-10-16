#ifndef SCRAN_MARKERS_SCORE_MARKERS_BEST_HPP
#define SCRAN_MARKERS_SCORE_MARKERS_BEST_HPP

#include <vector>
#include <cstddef>

#include "scran_blocks/scran_blocks.hpp"
#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"
#include "topicks/topicks.hpp"

#include "scan_matrix.hpp"
#include "average_group_stats.hpp"
#include "PrecomputedPairwiseWeights.hpp"
#include "create_combinations.hpp"
#include "cohens_d.hpp"
#include "simple_diff.hpp"
#include "utils.hpp"

/**
 * @file score_markers_best.hpp
 * @brief Find the best markers in each pairwise comparison between groups of cells.
 */

namespace scran_markers {

/**
 * @brief Options for `score_markers_best()` and friends.
 */
struct ScoreMarkersBestOptions {
    /**
     * Threshold on the differences in expression values between groups, used to adjust the Cohen's d and AUC calculations.
     * This should be non-negative.
     * Higher thresholds will favor genes with large differences at the expense of those with low variance. 
     */
    double threshold = 0;

    /**
     * Number of threads to use. 
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;

    /**
     * Whether to compute the mean expression in each group.
     */
    bool compute_group_mean = true;

    /**
     * Whether to compute the proportion of cells with detected expression in each group.
     */
    bool compute_group_detected = true;

    /**
     * Whether to compute Cohen's d. 
     */
    bool compute_cohens_d = true;

    /**
     * Whether to compute the AUC.
     */
    bool compute_auc = true;

    /**
     * Whether to compute the delta-mean, i.e., difference in means.
     */
    bool compute_delta_mean = true;

    /**
     * Whether to compute the delta-detected, i.e., difference in the detected proportion.
     */
    bool compute_delta_detected = true;

    /**
     * Whether to keep the genes with the largest Cohen's d.
     * If false, the genes with the smallest Cohen's d are retained instead.
     */
    bool largest_cohens_d = true;

    /**
     * Whether to keep the genes with the largest AUC.
     * If false, the genes with the smallest AUC are retained instead.
     */
    bool largest_auc = true;

    /**
     * Whether to keep the genes with the largest delta-mean.
     * If false, the genes with the smallest delta-mean are retained instead.
     */
    bool largest_delta_mean = true;

    /**
     * Whether to keep the genes with the largest delta-detected.
     * If false, the genes with the smallest delta-detected are retained instead.
     */
    bool largest_delta_detected = true;

    /**
     * The threshold on the Cohen's d.
     * Genes are only retained if their Cohen's d is greater (if `ScoreMarkersBestOptions::largest_cohens_d = true`) or less than this bound (otherwise).
     * If missing, no threshold is applied on the Cohen's d.
     */
    std::optional<double> threshold_cohens_d = 0;

    /**
     * The threshold on the AUC.
     * Genes are only retained if their AUC is greater (if `ScoreMarkersBestOptions::largest_auc = true`) or less than this bound (otherwise).
     * If missing, no threshold is applied on the AUC.
     */
    std::optional<double> threshold_auc = 0.5;

    /**
     * The threshold on the delta-mean.
     * Genes are only retained if their delta-mean is greater (if `ScoreMarkersBestOptions::largest_delta_mean = true`) or less than this bound (otherwise).
     * If missing, no threshold is applied on the delta-mean.
     */
    std::optional<double> threshold_delta_mean = 0;

    /**
     * The threshold on the delta-detected.
     * Genes are only retained if their delta-detected is greater (if `ScoreMarkersBestOptions::largest_delta_detected = true`) or less than this bound (otherwise).
     * If missing, no threshold is applied on the delta-detected.
     */
    std::optional<double> threshold_delta_detected = 0;

    /**
     * Whether to report genes with effect sizes that are tied with the `top`-th gene.
     */
    bool keep_ties = false;

    /**
     * Policy to use for weighting blocks when computing average statistics/effect sizes across blocks.
     *
     * The default of `scran_blocks::WeightPolicy::VARIABLE` is to define equal weights for blocks once they reach a certain size
     * (see `ScoreMarkersBestOptions::variable_block_weight_parameters`).
     * For smaller blocks, the weight is linearly proportional to its size to avoid outsized contributions from very small blocks.
     *
     * Other options include `scran_blocks::WeightPolicy::EQUAL`, where all blocks are equally weighted regardless of size;
     * and `scran_blocks::WeightPolicy::NONE`, where the contribution of each block is proportional to its size.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights, including the threshold at which blocks are considered to be large enough to have equal weight.
     * Only used when `ScoreMarkersBestOptions::block_weight_policy = scran_blocks::WeightPolicy::VARIABLE`.
     */
    scran_blocks::VariableWeightParameters variable_block_weight_parameters;
};

/**
 * @brief Results for `score_markers_best()` and friends.
 * @tparam Stat_ Floating-point type of the output statistics.
 * @tparam Index_ Integer type of the matrix row indices.
 */
template<typename Stat_, typename Index_>
struct ScoreMarkersBestResults {
    /**
     * Vector of length equal to the number of groups.
     * Each inner vector corresponds to a group and contains the mean expression of each gene in that group. 
     */
    std::vector<std::vector<Stat_> > mean;

    /**
     * Vector of length equal to the number of groups.
     * Each inner vector corresponds to a group and contains the mean expression of each gene in that group. 
     */
    std::vector<std::vector<Stat_> > detected;

    /**
     * Vector containing the genes with the largest Cohen's d for each pairwise comparison between groups.
     * Specifically, `cohens_d[i][j][k]` represents the `k`-th largest Cohen's d for the comparison of group `i` to group `j`
     * (i.e., a positive value indicates upregulation in `i` over `j`).
     * Each pair contains the index of the gene in the input matrix and the value of the Cohen's d.
     *
     * If `ScoreMarkersBestOptions::largest_cohens_d = false`, this instead contains the markers with the smallest Cohens'd d.
     *
     * This vector will be empty if `ScoreMarkersBestOptions::compute_cohens_d = false`.
     */
    std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > > cohens_d;

    /**
     * Vector containing the genes with the largest AUCs for each pairwise comparison between groups.
     * Specifically, `auc[i][j][k]` represents the `k`-th largest AUC for the comparison of group `i` to group `j`
     * (i.e., a positive value indicates upregulation in `i` over `j`).
     * Each pair contains the index of the gene in the input matrix and the value of the AUC.
     *
     * If `ScoreMarkersBestOptions::largest_auc = false`, this instead contains the markers with the smallest AUCs.
     *
     * This vector will be empty if `ScoreMarkersBestOptions::compute_auc = false`.
     */
    std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > > auc;

    /**
     * Vector containing the genes with the largest delta-means for each pairwise comparison between groups.
     * Specifically, `delta_mean[i][j][k]` represents the `k`-th largest delta-mean for the comparison of group `i` to group `j`
     * (i.e., a positive value indicates upregulation in `i` over `j`).
     * Each pair contains the index of the gene in the input matrix and the value of the delta-mean.
     *
     * If `ScoreMarkersBestOptions::largest_delta_mean = false`, this instead contains the markers with the smallest delta-means.
     *
     * This vector will be empty if `ScoreMarkersBestOptions::compute_delta_mean = false`.
     */
    std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > > delta_mean;

    /**
     * Vector containing the genes with the largest delta-detecteds for each pairwise comparison between groups.
     * Specifically, `delta_detected[i][j][k]` represents the `k`-th largest delta-detected for the comparison of group `i` to group `j`
     * (i.e., a positive value indicates upregulation in `i` over `j`).
     * Each pair contains the index of the gene in the input matrix and the value of the delta-detected.
     *
     * If `ScoreMarkersBestOptions::largest_delta_detected = false`, this instead contains the markers with the smallest delta-detecteds.
     *
     * This vector will be empty if `ScoreMarkersBestOptions::compute_delta_detected = false`.
     */
    std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > > delta_detected;
};

/**
 * @cond
 */
namespace internal {

template<typename Stat_, typename Index_>
using PairwiseTopQueues = std::vector<std::vector<topicks::TopQueue<Stat_, Index_> > >;

template<typename Stat_, typename Index_>
void allocate_best_top_queues(
    PairwiseTopQueues<Stat_, Index_>& pqueues,
    const std::size_t ngroups,
    const int top,
    const bool larger,
    const bool keep_ties,
    const std::optional<Stat_>& bound
) {
    topicks::TopQueueOptions<Stat_> opt;
    opt.check_nan = true;
    opt.keep_ties = keep_ties;
    if (bound.has_value()) {
        opt.bound = *bound;
    }

    sanisizer::resize(pqueues, ngroups);
    for (auto& x : pqueues) {
        x.reserve(ngroups);
        for (decltype(I(ngroups)) g = 0; g < ngroups; ++g) {
            x.emplace_back(top, larger, opt);
        }
    }
}

template<typename Stat_, typename Index_>
void add_best_top_queues(
    PairwiseTopQueues<Stat_, Index_>& pqueues,
    const Index_ gene,
    std::size_t ngroups,
    const std::vector<Stat_>& effects
) {
    for (decltype(I(ngroups)) g1 = 0; g1 < ngroups; ++g1) {
        for (decltype(I(ngroups)) g2 = 0; g2 < ngroups; ++g2) {
            const auto val = effects[sanisizer::nd_offset<std::size_t>(g2, ngroups, g1)];
            if (g1 != g2) {
                pqueues[g1][g2].emplace(val, gene);
            }
        }
    }
}

template<typename Stat_, typename Index_>
void report_best_top_queues(
    std::vector<PairwiseTopQueues<Stat_, Index_> >& pqueues,
    std::size_t ngroups,
    std::vector<std::vector<std::vector<std::pair<Index_, Stat_> > > >& output
) {
    // We know it fits int an 'int' as this is what we got originally.
    const int num_threads = pqueues.size();

    // Consolidating all of the thread-specific queues into a single queue.
    auto& true_pqueue = pqueues.front(); // we better have at least one thread.
    for (int t = 1; t < num_threads; ++t) {
        for (decltype(I(ngroups)) g1 = 0; g1 < ngroups; ++g1) {
            for (decltype(I(ngroups)) g2 = 0; g2 < ngroups; ++g2) {
                auto& current_in = pqueues[t][g1][g2];
                auto& current_out = true_pqueue[g1][g2];
                while (!current_in.empty()) {
                    current_out.push(current_in.top());
                    current_in.pop();
                }
            }
        }
    }

    // Now spilling them out into a single vector.
    sanisizer::resize(output, ngroups);
    for (decltype(I(ngroups)) g1 = 0; g1 < ngroups; ++g1) {
        sanisizer::resize(output[g1], ngroups);
        for (decltype(I(ngroups)) g2 = 0; g2 < ngroups; ++g2) {
            if (g1 == g2) {
                continue;
            }
            auto& current_in = true_pqueue[g1][g2];
            auto& current_out = output[g1][g2];
            while (!current_in.empty()) {
                const auto& best = current_in.top();
                current_out.emplace_back(best.second, best.first);
                current_in.pop();
            }
            std::reverse(current_out.begin(), current_out.end()); // earliest element should have the strongest effect sizes.
        }
    }
}

template<typename Index_, typename Stat_>
void find_best_simple_best_effects(
    const Index_ ngenes,
    const std::size_t ngroups,
    const std::size_t nblocks,
    const std::size_t ncombos,
    const std::vector<Stat_>& combo_weights,
    std::vector<Stat_>& combo_means,
    std::vector<Stat_>& combo_vars,
    std::vector<Stat_>& combo_detected,
    ScoreMarkersBestResults<Stat_, Index_>& output,
    int top,
    const ScoreMarkersBestOptions& options
) {
    std::vector<Stat_> total_weights_per_group;
    const Stat_* total_weights_ptr = combo_weights.data();
    if (nblocks > 1) {
        total_weights_per_group = compute_total_weight_per_group(ngroups, nblocks, combo_weights.data());
        total_weights_ptr = total_weights_per_group.data();
    }
    PrecomputedPairwiseWeights<Stat_> preweights(ngroups, nblocks, combo_weights.data());

    std::vector<Stat_*> mptrs;
    if (options.compute_group_mean) {
        mptrs.reserve(ngroups);
        sanisizer::resize(output.mean, ngroups);
        for (auto& x : output.mean) {
            sanisizer::resize(x, ngenes);
            mptrs.push_back(x.data());
        }
    }

    std::vector<Stat_*> dptrs;
    if (options.compute_group_detected) {
        dptrs.reserve(ngroups);
        sanisizer::resize(output.detected, ngroups);
        for (auto& x : output.detected) {
            sanisizer::resize(x, ngenes);
            dptrs.push_back(x.data());
        }
    }

    // Setting up the output queues.
    std::vector<PairwiseTopQueues<Stat_, Index_> > cohens_d_queues, delta_detected_queues, delta_mean_queues;
    if (options.compute_cohens_d) {
        sanisizer::resize(cohens_d_queues, options.num_threads);
    }
    if (options.compute_delta_mean) {
        sanisizer::resize(delta_mean_queues, options.num_threads);
    }
    if (options.compute_delta_detected) {
        sanisizer::resize(delta_detected_queues, options.num_threads);
    }

    const auto ngroups2 = sanisizer::product<typename std::vector<Stat_>::size_type>(ngroups, ngroups);

    tatami::parallelize([&](const int t, const Index_ start, const Index_ length) -> void {
        if (options.compute_cohens_d) {
            allocate_best_top_queues(cohens_d_queues[t], ngroups, top, options.largest_cohens_d, options.keep_ties, options.threshold_cohens_d);
        }
        if (options.compute_delta_mean) {
            allocate_best_top_queues(delta_mean_queues[t], ngroups, top, options.largest_delta_mean, options.keep_ties, options.threshold_delta_mean);
        }
        if (options.compute_delta_detected) {
            allocate_best_top_queues(delta_detected_queues[t], ngroups, top, options.largest_delta_detected, options.keep_ties, options.threshold_delta_detected);
        }
        std::vector<Stat_> buffer;
        if (options.compute_cohens_d || options.compute_delta_mean || options.compute_delta_detected) {
            buffer.resize(ngroups2);
        }

        for (Index_ gene = start, end = start + length; gene < end; ++gene) {
            auto in_offset = sanisizer::product_unsafe<std::size_t>(gene, ncombos);

            if (options.compute_group_mean) {
                const auto tmp_means = combo_means.data() + in_offset;
                average_group_stats(gene, ngroups, nblocks, tmp_means, combo_weights.data(), total_weights_ptr, mptrs);
            }
            if (options.compute_group_detected) {
                const auto tmp_detected = combo_detected.data() + in_offset;
                average_group_stats(gene, ngroups, nblocks, tmp_detected, combo_weights.data(), total_weights_ptr, dptrs);
            }

            // Computing the effect sizes.
            if (options.compute_cohens_d) {
                const auto tmp_means = combo_means.data() + in_offset;
                const auto tmp_variances = combo_vars.data() + in_offset;
                compute_pairwise_cohens_d(tmp_means, tmp_variances, ngroups, nblocks, preweights, options.threshold, buffer.data());
                add_best_top_queues(cohens_d_queues[t], gene, ngroups, buffer);
            }

            if (options.compute_delta_mean) {
                const auto tmp_means = combo_means.data() + in_offset;
                compute_pairwise_simple_diff(tmp_means, ngroups, nblocks, preweights, buffer.data());
                add_best_top_queues(delta_mean_queues[t], gene, ngroups, buffer);
            }

            if (options.compute_delta_detected) {
                const auto tmp_detected = combo_detected.data() + in_offset;
                compute_pairwise_simple_diff(tmp_detected, ngroups, nblocks, preweights, buffer.data());
                add_best_top_queues(delta_detected_queues[t], gene, ngroups, buffer);
            }
        }
    }, ngenes, options.num_threads);

    // Now figuring out which of these are the top dogs.
    if (options.compute_cohens_d) {
        report_best_top_queues(cohens_d_queues, ngroups, output.cohens_d);
    }

    if (options.compute_delta_mean) {
        report_best_top_queues(delta_mean_queues, ngroups, output.delta_mean);
    }

    if (options.compute_delta_detected) {
        report_best_top_queues(delta_detected_queues, ngroups, output.delta_detected);
    }
}

template<
    bool single_block_,
    typename Stat_,
    typename Value_,
    typename Index_,
    typename Group_,
    typename Block_
>
ScoreMarkersBestResults<Stat_, Index_> score_markers_best(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const std::size_t ngroups,
    const Group_* const group, 
    const std::size_t nblocks,
    const Block_* const block,
    const std::size_t ncombos,
    const std::size_t* const combo,
    const std::vector<Index_>& combo_sizes,
    int top,
    const ScoreMarkersBestOptions& options
) {
    const auto ngenes = matrix.nrow();
    const auto payload_size = sanisizer::product<typename std::vector<Stat_>::size_type>(ngenes, ncombos);
    std::vector<Stat_> combo_means, combo_vars, combo_detected;
    if (options.compute_group_mean || options.compute_cohens_d || options.compute_delta_mean) {
        combo_means.resize(payload_size);
    }
    if (options.compute_cohens_d) {
        combo_vars.resize(payload_size);
    }
    if (options.compute_group_detected || options.compute_delta_detected) {
        combo_detected.resize(payload_size);
    }

    // For a single block, this usually doesn't really matter, but we do it for consistency with the multi-block case,
    // and to account for variable weighting where non-zero block sizes get zero weight.
    const auto combo_weights = scran_blocks::compute_weights<Stat_>(
        combo_sizes,
        options.block_weight_policy,
        options.variable_block_weight_parameters
    );

    ScoreMarkersBestResults<Stat_, Index_> output;

    if (options.compute_auc) {
        auto auc_queues = sanisizer::create<std::vector<PairwiseTopQueues<Stat_, Index_> > >(options.num_threads);

        struct AucResultWorkspace {
            AucResultWorkspace() = default;
            AucResultWorkspace(const std::size_t ngroups, PairwiseTopQueues<Stat_, Index_>& pqueue) :
                pairwise_buffer(sanisizer::product<typename std::vector<Stat_>::size_type>(ngroups, ngroups)),
                queue_ptr(&pqueue)
            {};
            std::vector<Stat_> pairwise_buffer;
            PairwiseTopQueues<Stat_, Index_>* queue_ptr;
        };

        scan_matrix_by_row_custom_auc<single_block_>(
            matrix, 
            ngroups,
            group,
            nblocks,
            block,
            ncombos,
            combo,
            combo_means,
            combo_vars,
            combo_detected,
            /* do_auc = */ true,
            /* auc_result_initialize = */ [&](int t) -> AucResultWorkspace {
                allocate_best_top_queues(auc_queues[t], ngroups, top, options.largest_auc, options.keep_ties, options.threshold_auc);
                return AucResultWorkspace(ngroups, auc_queues[t]);
            },
            /* auc_result_process = */ [&](const Index_ gene, AucScanWorkspace<Value_, Group_, Index_, Stat_>& auc_work, AucResultWorkspace& res_work) -> void {
                process_auc_for_rows(auc_work, ngroups, nblocks, options.threshold, res_work.pairwise_buffer.data());
                add_best_top_queues(*(res_work.queue_ptr), gene, ngroups, res_work.pairwise_buffer);
            },
            combo_sizes,
            combo_weights,
            options.num_threads
        );

        report_best_top_queues(auc_queues, ngroups, output.auc);

    } else if (matrix.prefer_rows()) {
        scan_matrix_by_row_full_auc<single_block_>(
            matrix, 
            ngroups,
            group,
            nblocks,
            block,
            ncombos,
            combo,
            combo_means,
            combo_vars,
            combo_detected,
            static_cast<Stat_*>(NULL),
            combo_sizes,
            combo_weights,
            options.threshold,
            options.num_threads
        );

    } else {
        scan_matrix_by_column(
            matrix,
            [&]{
                if constexpr(single_block_) {
                    return ngroups;
                } else {
                    return ncombos;
                }
            }(),
            [&]{
                if constexpr(single_block_) {
                    return group;
                } else {
                    return combo;
                }
            }(),
            combo_means,
            combo_vars,
            combo_detected,
            combo_sizes,
            options.num_threads
        );
    }

    find_best_simple_best_effects(
        matrix.nrow(),
        ngroups,
        nblocks,
        ncombos,
        combo_weights,
        combo_means,
        combo_vars,
        combo_detected,
        output,
        top,
        options
    );

    return output;
}

}
/**
 * @endcond
 */

/**
 * Find potential marker genes with the largest effect sizes in each pairwise comparison between groups.
 * This function is equivalent to (but more efficient than) running `score_markers_pairwise()`
 * and then using `topicks::pick_top_genes()` on the effect sizes from each pairwise comparison.
 * The idea is to identify the top markers without a large memory allocation to hold the 3D array of effect sizes.
 *
 * @tparam Stat_ Floating-point type of the statistics.
 * @tparam Value_ Matrix data type.
 * @tparam Index_ Matrix index type.
 * @tparam Group_ Integer type of the group assignments.
 *
 * @param matrix A matrix of expression values, typically normalized and log-transformed.
 * Rows should contain genes while columns should contain cells.
 * @param[in] group Pointer to an array of length equal to the number of columns in `matrix`, containing the group assignments.
 * Group identifiers should be 0-based and should contain all integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param top Number of top genes to retain from each pairwise comparison.
 * The actual number of retained genes may be less than or greater than `top` depending on the number of rows in `matrix`
 * and the choices of `ScoreMarkersBestOptions::keep_ties`, `ScoreMarkersBestOptions::threshold_cohens_d`, etc.
 * @param options Further options.
 *
 * @return Object containing the top markers from each pairwise comparison.
 */
template<typename Stat_, typename Value_, typename Index_, typename Group_>
ScoreMarkersBestResults<Stat_, Index_> score_markers_best(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Group_* const group, 
    int top,
    const ScoreMarkersBestOptions& options
) {
    const Index_ NC = matrix.ncol();
    const auto group_sizes = tatami_stats::tabulate_groups(group, NC); 
    const auto ngroups = sanisizer::cast<std::size_t>(group_sizes.size());

    return internal::score_markers_best<true, Stat_>(
        matrix,
        ngroups,
        group,
        1,
        static_cast<int*>(NULL),
        ngroups,
        static_cast<std::size_t*>(NULL),
        group_sizes,
        top,
        options
    );
}

/**
 * Find potential marker genes with the largest effect sizes in each pairwise comparison between groups,
 * after accounting for any blocking factor in the dataset.
 * This function is equivalent to (but more efficient than) running `score_markers_pairwise_blocked()`
 * and then using `topicks::pick_top_genes()` on the effect sizes from each pairwise comparison.
 * The idea is to identify the top markers without a large memory allocation to hold the 3D array of effect sizes.
 *
 * @tparam Stat_ Floating-point type of the statistics.
 * @tparam Value_ Matrix data type.
 * @tparam Index_ Matrix index type.
 * @tparam Group_ Integer type of the group assignments.
 * @tparam Block_ Integer type of the block assignments.
 *
 * @param matrix A matrix of expression values, typically normalized and log-transformed.
 * Rows should contain genes while columns should contain cells.
 * @param[in] group Pointer to an array of length equal to the number of columns in `matrix`, containing the group assignments.
 * Group identifiers should be 0-based and should contain all integers in \f$[0, N)\f$ where \f$N\f$ is the number of unique groups.
 * @param[in] block Pointer to an array of length equal to the number of columns in `matrix`, containing the blocking factor.
 * Block identifiers should be 0-based and should contain all integers in \f$[0, B)\f$ where \f$B\f$ is the number of unique blocking levels.
 * @param top Number of top genes to retain from each pairwise comparison.
 * The actual number of retained genes may be less than or greater than `top` depending on the number of rows in `matrix`
 * and the choices of `ScoreMarkersBestOptions::keep_ties`, `ScoreMarkersBestOptions::threshold_cohens_d`, etc.
 * @param options Further options.
 *
 * @return Object containing the top markers with the largest effect sizes from each pairwise comparison.
 */
template<typename Stat_, typename Value_, typename Index_, typename Group_, typename Block_>
ScoreMarkersBestResults<Stat_, Index_> score_markers_best_blocked(
    const tatami::Matrix<Value_, Index_>& matrix, 
    const Group_* const group, 
    const Block_* const block,
    int top,
    const ScoreMarkersBestOptions& options
) {
    const Index_ NC = matrix.ncol();
    const auto ngroups = tatami_stats::total_groups(group, NC);
    const auto nblocks = tatami_stats::total_groups(block, NC); 

    const auto combinations = internal::create_combinations(ngroups, group, block, NC);
    const auto combo_sizes = internal::tabulate_combinations<Index_>(ngroups, nblocks, combinations);
    const auto ncombos = combo_sizes.size();

    return internal::score_markers_best<false, Stat_>(
        matrix,
        sanisizer::cast<std::size_t>(ngroups),
        group,
        sanisizer::cast<std::size_t>(nblocks),
        block,
        sanisizer::cast<std::size_t>(ncombos),
        combinations.data(),
        combo_sizes,
        top,
        options
    );
}

}

#endif
