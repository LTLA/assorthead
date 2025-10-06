#ifndef TOPICKS_TOP_QUEUE_HPP
#define TOPICKS_TOP_QUEUE_HPP

#include <vector>
#include <algorithm>
#include <optional>
#include <limits>
#include <cmath>

#include "sanisizer/sanisizer.hpp"

/**
 * @file TopQueue.hpp
 * @brief Priority queue for the top genes.
 */

namespace topicks {

/**
 * @brief Options for `TopQueue`.
 * @tparam Stat_ Numeric type of the statistic for ranking top genes.
 */
template<typename Stat_>
struct TopQueueOptions {
    /**
     * A absolute bound for the statistic.
     * A gene will not be inserted in the queue if its statistic is:
     *
     * - equal to or lower than the bound, when `larger = true` and `TopQueueOptions::open_bound = true`.
     * - lower than the bound, when `larger = true` and `TopQueueOptions::open_bound = false`.
     * - equal to or greater than the bound, when `larger = false` and `TopQueueOptions::open_bound = true`.
     * - greater than the bound, when `larger = false` and `TopQueueOptions::open_bound = false`.
     *
     * If unset, no absolute bound is applied to the statistic.
     */
    std::optional<Stat_> bound;

    /**
     * Whether `TopQueue::bound` is an open interval, i.e., genes with statistics equal to the bound will not be inserted in the queue.
     * Only relevant if `TopQueue::bound` is set.
     */
    bool open_bound = true;

    /**
     * Whether to keep all genes with statistics that are tied with the `top`-th gene.
     * If `false`, the number of genes in the queue will not be greater than `top`; ties are broken by retaining genes with a lower index.
     */
    bool keep_ties = true;

    /**
     * Whether to check for NaN values and ignore them.
     * If `false`, it is assumed that no NaNs will be added to the queue.
     */
    bool check_nan = false;
};

/**
 * @brief Priority queue for retaining top genes.
 * @tparam Stat_ Numeric type of the statistic for ranking top genes.
 * @tparam Index_ Integer type of gene index.
 *
 * The `TopQueue` class retains the indices of the top genes, analogous to `pick_top_genes_index()`.
 * This is useful if the user does not want to store an array of statistics for all genes;
 * instead, the statistics can be computed for each gene and added to this queue to reduce memory usage.
 */
template<typename Stat_, typename Index_>
class TopQueue {
private:
    typedef std::pair<Stat_, Index_> Element;
    std::vector<Element> my_queue, my_ties;

    struct SortGreater {
        bool operator()(const Element& left, const Element& right) const {
            if (left.first == right.first) {
                return left.second < right.second;
            } else {
                return left.first > right.first;
            }
        }
    };

    struct SortLess {
        bool operator()(const Element& left, const Element& right) const {
            if (left.first == right.first) {
                return left.second < right.second;
            } else {
                return left.first < right.first;
            }
        }
    };

    struct SortTies {
        bool operator()(const Element& left, const Element& right) const {
            return left.second < right.second;
        }
    };

    Index_ my_top;
    bool my_larger;
    std::optional<Stat_> my_bound; 
    bool my_open_bound;
    bool my_keep_ties;
    bool my_check_nan;

private:
    template<class Compare_, class Skip_>
    void push_internal(std::pair<Stat_, Index_> gene, const Compare_ cmp, const Skip_ skip) {
        if (my_bound.has_value()) {
            if (skip(*my_bound, gene.first)) {
                return;
            } else if (my_open_bound && *my_bound == gene.first) {
                return;
            }
        }

        if constexpr(std::numeric_limits<Stat_>::has_quiet_NaN) {
            if (my_check_nan && std::isnan(gene.first)) {
                return;
            }
        }

        if (sanisizer::is_less_than(my_queue.size(), my_top)) {
            my_queue.push_back(std::move(gene));
            std::push_heap(my_queue.begin(), my_queue.end(), cmp);
            return;
        }

        if (my_top == 0) {
            return;
        }
        const auto& top = my_queue.front();
        if (skip(top.first, gene.first)) {
            return;
        }

        if (!my_keep_ties) {
            if (top.first != gene.first || gene.second < top.second) {
                std::pop_heap(my_queue.begin(), my_queue.end(), cmp);
                my_queue.back() = std::move(gene);
                std::push_heap(my_queue.begin(), my_queue.end(), cmp);
            }
            return;
        }

        if (top.first == gene.first) {
            // Always making sure that all elements in my_ties are of equal or lower priority than the tied element in my_queue.
            if (gene.second >= top.second) {
                my_ties.push_back(std::move(gene));
                std::push_heap(my_ties.begin(), my_ties.end(), SortTies());
            } else {
                auto retop = top;
                std::pop_heap(my_queue.begin(), my_queue.end(), cmp);
                my_queue.back() = std::move(gene);
                std::push_heap(my_queue.begin(), my_queue.end(), cmp);
                my_ties.push_back(std::move(retop));
                std::push_heap(my_ties.begin(), my_ties.end(), SortTies());
            }
            return;
        }

        if (my_top == 1) {
            my_queue.front() = std::move(gene);
            return;
        }

        // The idea is to check whether the top 2 elements of the queue are tied,
        // and if so, shift the top value into the tie queue.
        auto retop = top;
        std::pop_heap(my_queue.begin(), my_queue.end(), cmp);
        if (retop.first == my_queue.front().first) {
            my_ties.push_back(std::move(retop));
            std::push_heap(my_ties.begin(), my_ties.end(), SortTies());
        } else {
            my_ties.clear();
        }
        my_queue.back() = std::move(gene);
        std::push_heap(my_queue.begin(), my_queue.end(), cmp);
    }

    template<class Compare_>
    void pop_internal(Compare_ cmp) {
        if (my_ties.size()) {
            std::pop_heap(my_ties.begin(), my_ties.end(), SortTies());
            my_ties.pop_back();
        } else {
            std::pop_heap(my_queue.begin(), my_queue.end(), cmp);
            my_queue.pop_back();
        }
    }

public:
    /**
     * @param top Number of top genes to choose.
     * Note that the actual number of chosen genes may be smaller/larger than `top`, depending on the number of genes and `options`.
     * @param larger Whether the top genes are defined as those with larger statistics.
     * @param options Further options.
     */
    TopQueue(const Index_ top, const bool larger, const TopQueueOptions<Stat_>& options) : 
        my_top(top),
        my_larger(larger),
        my_bound(options.bound),
        my_open_bound(options.open_bound),
        my_keep_ties(options.keep_ties),
        my_check_nan(options.check_nan)
    {}

    /**
     * @param gene The statistic and the index of a gene.
     * This is inserted into the queue if it is among the `top` current genes, otherwise no modification is performed.
     */
    void push(std::pair<Stat_, Index_> gene) {
        if (my_larger) {
            push_internal(
                std::move(gene), 
                SortGreater(),
                [](const Stat_ existing, const Stat_ latest) -> bool { return existing > latest; }
            );
        } else {
            push_internal(
                std::move(gene),
                SortLess(),
                [](const Stat_ existing, const Stat_ latest) -> bool { return existing < latest; }
            );
        }
    }

    /**
     * @param statistic Statistic for the `index`-th gene.
     * @param index Index of the gene.
     *
     * The gene is inserted into the queue if it is among the `top` current genes, otherwise no modification is performed.
     */
    void emplace(Stat_ statistic, Index_ index) {
        push({ statistic, index });
    }

    /**
     * @return Size of the queue.
     * This may be more than `top` if ties are included.
     */
    Index_ size() const {
        return my_queue.size() + my_ties.size();
    }

    /**
     * @return Whether the queue is empty.
     */
    bool empty() const {
        return size() == 0;
    }

    /**
     * @return Top item in the queue, i.e., the gene with the smallest (if `larger = true`) or largest (otherwise) statistic.
     * In case of ties, the gene with the largest index is returned.
     *
     * This should only be called if `empty()` returns false.
     */
    const std::pair<Stat_, Index_>& top() const {
        if (my_ties.size()) {
            return my_ties.front();
        } else {
            return my_queue.front();
        }
    }

    /**
     * Removes the top item in the queue.
     * This should only be called if `empty()` returns false.
     */
    void pop() {
        if (my_larger) {
            pop_internal(SortGreater());
        } else {
            pop_internal(SortLess());
        }
    }
};

}

#endif
