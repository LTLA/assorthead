#ifndef SINGLER_CLASSIC_MARKERS_QUEUE_HPP
#define SINGLER_CLASSIC_MARKERS_QUEUE_HPP

#include <vector>
#include <cstddef>
#include <optional>

#include "topicks/topicks.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace singler_classic_markers {

template<typename Stat_, typename Index_>
using PairwiseTopQueues = std::vector<std::vector<topicks::TopQueue<Stat_, Index_> > >;

template<typename Stat_, typename Index_>
void allocate_pairwise_queues(
    PairwiseTopQueues<Stat_, Index_>& pqueues,
    const Index_ num_keep,
    const std::size_t ngroups,
    const bool keep_ties,
    const bool check_nan
) {
    topicks::TopQueueOptions<Stat_> opt;
    opt.check_nan = check_nan;
    opt.keep_ties = keep_ties;
    opt.bound = 0;
    sanisizer::resize(pqueues, ngroups);
    for (auto& x : pqueues) {
        x.reserve(ngroups);
        for (I<decltype(ngroups)> g = 0; g < ngroups; ++g) {
            x.emplace_back(num_keep, true, opt);
        }
    }
}

template<bool include_stat_, typename Stat_, typename Index_>
void report_best_top_queues(
    std::vector<PairwiseTopQueues<Stat_, Index_> >& pqueues,
    const std::size_t ngroups,
    Markers<include_stat_, Index_, Stat_>& output
) {
    // We know it fits into an 'int' as this is what we got originally.
    const int num_threads = pqueues.size();

    // Consolidating all of the thread-specific queues into a single queue.
    auto& true_pqueue = pqueues.front(); // we better have at least one thread.
    for (int t = 1; t < num_threads; ++t) {
        for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
            for (I<decltype(ngroups)> g2 = 0; g2 < ngroups; ++g2) {
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
    for (I<decltype(ngroups)> g1 = 0; g1 < ngroups; ++g1) {
        sanisizer::resize(output[g1], ngroups);
        for (I<decltype(ngroups)> g2 = 0; g2 < ngroups; ++g2) {
            if (g1 == g2) {
                continue;
            }
            auto& current_in = true_pqueue[g1][g2];
            auto& current_out = output[g1][g2];
            while (!current_in.empty()) {
                const auto& best = current_in.top();
                if constexpr(include_stat_) { 
                    current_out.emplace_back(best.second, best.first);
                } else {
                    current_out.emplace_back(best.second);
                }
                current_in.pop();
            }
            std::reverse(current_out.begin(), current_out.end()); // earliest element should have the strongest effect sizes.
        }
    }
}

}

#endif
