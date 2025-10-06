#ifndef TOPICKS_PICK_TOP_GENES_HPP
#define TOPICKS_PICK_TOP_GENES_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>
#include <type_traits>
#include <limits>
#include <cmath>

#include "sanisizer/sanisizer.hpp"

/**
 * @file pick_top_genes.hpp
 * @brief Pick top genes from an array of statistics.
 */

namespace topicks {

/**
 * @brief Options for `pick_top_genes()`.
 * @tparam Stat_ Numeric type of the statistic for picking top genes.
 */
template<typename Stat_>
struct PickTopGenesOptions {
    /**
     * A absolute bound for the statistic, to be considered when choosing the top genes.
     * A gene will not be picked, even if it is among the top `top` genes, if its statistic is:
     *
     * - equal to or lower than the bound, when `larger = true` and `PickTopGenesOptions::open_bound = true`.
     * - lower than the bound, when `larger = true` and `PickTopGenesOptions::open_bound = false`.
     * - equal to or greater than the bound, when `larger = false` and `PickTopGenesOptions::open_bound = true`.
     * - greater than the bound, when `larger = false` and `PickTopGenesOptions::open_bound = false`.
     *
     * If unset, no absolute bound is applied to the statistic.
     */
    std::optional<Stat_> bound;

    /**
     * Whether `PickTopGenesOptions::bound` is an open interval, i.e., genes with statistics equal to the bound will not be picked. 
     * Only relevant if `PickTopGenesOptions::bound` is set.
     */
    bool open_bound = true;

    /**
     * Whether to keep all genes with statistics that are tied with the `top`-th gene.
     * If `false`, ties are arbitrarily broken but the number of retained genes will not be greater than `top`.
     */
    bool keep_ties = true;

    /**
     * Whether to check for NaN values and ignore them.
     * If `false`, it is assumed that no NaNs are present in `statistic`.
     */
    bool check_nan = false;
};

/**
 * @cond
 */
namespace internal {

template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(Input_ x) {
    return x;
}

template<bool keep_index_, typename Index_, typename Stat_, class Output_, class Cmp_>
void filter_genes_by_threshold(const Index_ n, const Stat_* statistic, Output_& output, const Cmp_ cmp, const Stat_ threshold) {
    // This function is inherently safe as 'ok' will always be false for any comparison involving NaNs.
    for (Index_ i = 0; i < n; ++i) {
        const bool ok = cmp(statistic[i], threshold);
        if constexpr(keep_index_) {
            if (ok) {
                output.push_back(i);
            }
        } else {
            output[i] = ok;
        }
    }
}

template<bool keep_index_, typename Index_, typename Stat_, class Output_, class Cmp_>
void select_top_genes_by_threshold(const Index_ top, const Stat_* statistic, Output_& output, const Cmp_ cmp, const Stat_ threshold, const std::vector<Index_>& semi_sorted) {
    Index_ counter = top;
    while (counter > 0) {
        --counter;
        const auto pos = semi_sorted[counter];
        if (cmp(statistic[pos], threshold)) {
            if constexpr(keep_index_) {
                output.push_back(pos);
            } else {
                output[pos] = true;
            }
        }
    }
}

template<bool keep_index_, typename Index_, typename Stat_, class Output_, class CmpNotEqual_, class CmpEqual_>
void pick_top_genes(const Index_ n, const Stat_* statistic, const Index_ top, Output_& output, const CmpNotEqual_ cmpne, const CmpEqual_ cmpeq, const PickTopGenesOptions<Stat_>& options) {
    if (top == 0) {
        if constexpr(keep_index_) {
            ; // no-op, we assume it's already empty.
        } else {
            std::fill_n(output, n, false);
        }
        return;
    }

    Index_ num_nan = 0;
    if constexpr(std::numeric_limits<Stat_>::has_quiet_NaN) {
        if (options.check_nan) {
            for (Index_ i = 0; i < n; ++i) {
                num_nan += std::isnan(statistic[i]);
            }
        }
    }

    if (top >= n - num_nan) {
        if (options.bound.has_value()) {
            if (options.open_bound) {
                filter_genes_by_threshold<keep_index_>(n, statistic, output, cmpne, *(options.bound));
            } else {
                filter_genes_by_threshold<keep_index_>(n, statistic, output, cmpeq, *(options.bound));
            }
        } else if (num_nan == 0) {
            if constexpr(keep_index_) {
                sanisizer::resize(output, n);
                std::iota(output.begin(), output.end(), static_cast<Index_>(0));
            } else {
                std::fill_n(output, n, true);
            }
        } else {
            if constexpr(keep_index_) {
                output.reserve(n - num_nan);
                for (Index_ i = 0; i < n; ++i) {
                    if (!std::isnan(statistic[i])) {
                        output.push_back(i);
                    }
                }
            } else {
                for (Index_ i = 0; i < n; ++i) {
                    output[i] = !std::isnan(statistic[i]);
                }
            }
        }
        return;
    }

    std::vector<Index_> semi_sorted;
    if (num_nan == 0) {
        sanisizer::resize(semi_sorted, n);
        std::iota(semi_sorted.begin(), semi_sorted.end(), static_cast<Index_>(0));
    } else {
        semi_sorted.reserve(n - num_nan);
        for (Index_ i = 0; i < n; ++i) {
            if (!std::isnan(statistic[i])) {
                semi_sorted.push_back(i);
            }
        }
    }

    const auto cBegin = semi_sorted.begin(), cMid = cBegin + top - 1, cEnd = semi_sorted.end();
    std::nth_element(cBegin, cMid, cEnd, [&](const Index_ l, const Index_ r) -> bool { 
        const auto L = statistic[l], R = statistic[r];
        if (L == R) {
            return l < r; // always favor the earlier index for a stable sort, even if larger = false.
        } else {
            return cmpne(L, R);
        }
    });
    const Stat_ threshold = statistic[*cMid];

    if (options.keep_ties) {
        if (options.bound.has_value()) {
            const auto bound = *(options.bound);

            if (options.open_bound) {
                if (!cmpne(threshold, bound)) {
                    filter_genes_by_threshold<keep_index_>(n, statistic, output, cmpne, *(options.bound));
                    return;
                }
            } else {
                if (!cmpeq(threshold, bound)) {
                    filter_genes_by_threshold<keep_index_>(n, statistic, output, cmpeq, *(options.bound));
                    return;
                }
            }
        }

        filter_genes_by_threshold<keep_index_>(n, statistic, output, cmpeq, threshold);
        return;
    }

    if constexpr(keep_index_) {
        output.reserve(sanisizer::cast<decltype(I(output.size()))>(top));
    } else {
        std::fill_n(output, n, false);
    }

    if (options.bound.has_value()) {
        // cast of 'top' to Index_ is known safe as top <= n by this point.
        if (options.open_bound) {
            select_top_genes_by_threshold<keep_index_>(static_cast<Index_>(top), statistic, output, cmpne, *(options.bound), semi_sorted);
        } else {
            select_top_genes_by_threshold<keep_index_>(static_cast<Index_>(top), statistic, output, cmpeq, *(options.bound), semi_sorted);
        }
    } else {
        if constexpr(keep_index_) {
            output.insert(output.end(), semi_sorted.begin(), semi_sorted.begin() + top);
        } else {
            for (decltype(I(top)) i = 0; i < top; ++i) {
                output[semi_sorted[i]] = true;
            }
        }
    }

    if constexpr(keep_index_) {
        std::sort(output.begin(), output.end());
    }
}

}
/**
 * @endcond
 */

/**
 * @tparam Stat_ Numeric type of the statistic for picking top genes.
 * @tparam Bool_ Output boolean type. 
 *
 * @param n Number of genes.
 * @param[in] statistic Pointer to an array of length `n`, containing the statistics with which to rank genes.
 * @param top Number of top genes to choose.
 * @param larger Whether the top genes are defined as those with larger statistics.
 * @param[out] output Pointer to an array of length `n`. 
 * On output, the `i`-th element will be `true` if gene `i` is one of the top genes and `false` otherwise.
 * Note that the actual number of chosen genes may be smaller/larger than `top`, depending on `n` and `options`.
 * @param options Further options.
 */
template<typename Stat_, typename Bool_>
void pick_top_genes(const std::size_t n, const Stat_* const statistic, const std::size_t top, const bool larger, Bool_* const output, const PickTopGenesOptions<Stat_>& options) {
    if (larger) {
        internal::pick_top_genes<false>(
            n, 
            statistic, 
            top,
            output, 
            [](Stat_ l, Stat_ r) -> bool { return l > r; },
            [](Stat_ l, Stat_ r) -> bool { return l >= r; },
            options
        );
    } else {
        internal::pick_top_genes<false>(
            n, 
            statistic, 
            top,
            output, 
            [](Stat_ l, Stat_ r) -> bool { return l < r; },
            [](Stat_ l, Stat_ r) -> bool { return l <= r; },
            options
        );
    }
}

/**
 * @tparam Bool_ Output boolean type. 
 * @tparam Stat_ Numeric type of the statistic for picking top genes.
 *
 * @param n Number of genes.
 * @param[in] statistic Pointer to an array of length `n`, containing the statistics with which to rank genes.
 * @param top Number of top genes to choose.
 * @param larger Whether the top genes are defined as those with larger statistics.
 * @param options Further options.
 *
 * @return A vector of booleans of length `n`, indicating whether each gene is to be retained.
 * Note that the actual number of chosen genes may be smaller/larger than `top`, depending on `n` and `options`.
 */
template<typename Bool_, typename Stat_>
std::vector<Bool_> pick_top_genes(const std::size_t n, const Stat_* const statistic, const std::size_t top, const bool larger, const PickTopGenesOptions<Stat_>& options) {
    auto output = sanisizer::create<std::vector<Bool_> >(n
#ifdef SCRAN_VARIANCES_TEST_INIT
        , SCRAN_VARIANCES_TEST_INIT
#endif
    );
    pick_top_genes(n, statistic, top, larger, output.data(), options);
    return output;
}

/**
 * @tparam Index_ Integer type of the output indices.
 * @tparam Stat_ Numeric type of the statistic for picking top genes.
 *
 * @param n Number of genes.
 * @param[in] statistic Pointer to an array of length `n` containing the statistics with which to rank genes.
 * @param top Number of top genes to choose.
 * @param larger Whether the top genes are defined as those with larger statistics.
 * @param options Further options.
 *
 * @return Vector of sorted and unique indices for the chosen genes.
 * All indices are guaranteed to be non-negative and less than `n`.
 * Note that the actual number of chosen genes may be smaller/larger than `top`, depending on `n` and `options`.
 */
template<typename Index_, typename Stat_>
std::vector<Index_> pick_top_genes_index(const Index_ n, const Stat_* const statistic, const Index_ top, const bool larger, const PickTopGenesOptions<Stat_>& options) {
    std::vector<Index_> output;
    if (larger) {
        internal::pick_top_genes<true>(
            n, 
            statistic, 
            top,
            output,
            [](Stat_ l, Stat_ r) -> bool { return l > r; },
            [](Stat_ l, Stat_ r) -> bool { return l >= r; },
            options
        );
    } else {
        internal::pick_top_genes<true>(
            n, 
            statistic, 
            top,
            output,
            [](Stat_ l, Stat_ r) -> bool { return l < r; },
            [](Stat_ l, Stat_ r) -> bool { return l <= r; },
            options
        );
    }
    return output;
}

}

#endif
