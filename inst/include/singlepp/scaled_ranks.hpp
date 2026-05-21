#ifndef SINGLEPP_SCALED_RANKS_HPP
#define SINGLEPP_SCALED_RANKS_HPP

#include "utils.hpp"

#include "sanisizer/sanisizer.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cassert>

namespace singlepp {

template<typename Stat_, typename Index_>
using RankedVector = std::vector<std::pair<Stat_, Index_> >;

// Compute the centered rank of each observation, returning the sum of squares.
// This can be converted into scaled ranks by multiplying the contents of 'output' with 'sum_squares_to_mult(sum_squares)'.
// Used in both 'scaled_ranks_dense()' and an overload of 'scaled_ranks_sparse_l2()'.
template<typename Index_, typename Stat_, typename Float_>
Float_ centered_ranks_dense(const Index_ num_markers, const RankedVector<Stat_, Index_>& collected, Float_* output) {
    static_assert(std::is_floating_point<Float_>::value);
    assert(sanisizer::is_equal(num_markers, collected.size()));
    if (num_markers == 0) {
        return 0;
    }

    const Float_ center_rank = static_cast<Float_>(num_markers - 1) / static_cast<Float_>(2); 
    Float_ sum_squares = 0;

    // Computing tied ranks. 
    Index_ cur_rank = 0;
    auto cIt = collected.begin();
    auto cEnd = collected.end();

    while (cIt != cEnd) {
        auto copy = cIt;
        do {
            ++copy;
        } while (copy != cEnd && copy->first == cIt->first);

        const Float_ jump = copy - cIt;
        const Float_ mean_rank = cur_rank + static_cast<Float_>(jump - 1) / static_cast<Float_>(2) - center_rank;
        while (cIt != copy) {
            output[cIt->second] = mean_rank;
            ++cIt;
        }

        sum_squares += mean_rank * mean_rank * jump;
        cur_rank += jump;
    }
    
    return sum_squares;
}

// Obtain the multiplier for converting the centered ranks into scaled ranks.
template<typename Float_>
Float_ sum_squares_to_mult(const Float_ sum_squares) {
    assert(sum_squares > 0);
    return 0.5 / std::sqrt(sum_squares);
}

// Compute scaled ranks from a dense ranked vector (i.e., 'collected' contains all indices in '[0, collected.size())').
// Each scaled rank is passed along with its index to the 'process' function.
// The return value is whether there are any non-zero values in the scaled ranks.
template<typename Index_, typename Stat_, typename Float_, class Process_>
bool scaled_ranks_dense(const Index_ num_markers, const RankedVector<Stat_, Index_>& collected, Float_* workspace, Process_ process) {
    const auto sum_squares = centered_ranks_dense(num_markers, collected, workspace);

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    if (sum_squares == 0) {
        for (Index_ i = 0; i < num_markers; ++i) {
            process(i, 0);
        }
        return false;
    }

    const Float_ mult = sum_squares_to_mult(sum_squares);
    for (Index_ i = 0; i < num_markers; ++i) {
        process(i, workspace[i] * mult);
    }
    return true;
}

// Compute scaled ranks from a dense ranked vector (i.e., 'collected' contains all indices in '[0, collected.size())') and store them in 'output'.
// The return value is whether there are any non-zero values in the scaled ranks.
template<typename Index_, typename Stat_, typename Float_>
bool scaled_ranks_dense(const Index_ num_markers, const RankedVector<Stat_, Index_>& collected, Float_* output) { 
    return scaled_ranks_dense(
        num_markers,
        collected, 
        output, 
        [&](const Index_ i, const Float_ val) -> void {
            output[i] = val;
        }
    );
}

// Compute scaled ranks from two iterator ranges of sparse ranked vectors containing negative and positive values.
// (This also accepts the sizes of each directly to avoid having to compute it from the difference of iterators if we already knew it.)
// It calls 'zprocess()' with the scaled rank of the zero values.
// It also calls 'nzprocess' on each of the scaled ranks of the non-zero values.
// Return value is whether there are any non-zero values in the scaled ranks.
template<typename Index_, typename Stat_, typename Float_, class ZeroProcess_, class NonzeroProcess_>
bool scaled_ranks_sparse(
    const Index_ num_markers,
    const Index_ num_negative,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_start,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_end,
    const Index_ num_positive,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_start,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_end,
    std::vector<std::pair<Index_, Float_> >& workspace,
    ZeroProcess_ zprocess,
    NonzeroProcess_ nzprocess
) {
    static_assert(std::is_floating_point<Float_>::value);

    assert(sanisizer::is_equal(negative_end - negative_start, num_negative));
    assert(sanisizer::is_equal(positive_end - positive_start, num_positive));
    assert(sanisizer::is_greater_than_or_equal(num_markers, sanisizer::sum<Index_>(num_positive, num_negative)));

    workspace.clear();
    if (num_markers == 0) {
        zprocess(0);
        return false;
    }

    Float_ sum_squares = 0;
    const Float_ center_rank = static_cast<Float_>(num_markers - 1) / static_cast<Float_>(2); 

    // Computing tied ranks: before, at, and after zero.
    Index_ cur_rank = 0;
    auto nIt = negative_start;
    while (nIt != negative_end) {
        auto copy = nIt;
        do {
            ++copy;
        } while (copy != negative_end && copy->first == nIt->first);

        const Float_ jump = copy - nIt;
        const Float_ mean_rank = cur_rank + static_cast<Float_>(jump - 1) / static_cast<Float_>(2) - center_rank;
        while (nIt != copy) {
            workspace.emplace_back(nIt->second, mean_rank);
            ++nIt;
        }

        sum_squares += mean_rank * mean_rank * jump;
        cur_rank += jump;
    }

    Index_ num_zero = num_markers - num_negative - num_positive;
    Float_ zero_rank = 0; 
    if (num_zero) {
        zero_rank = cur_rank + static_cast<Float_>(num_zero - 1) / static_cast<Float_>(2) - center_rank;
        sum_squares += zero_rank * zero_rank * num_zero;
        cur_rank += num_zero;
    }

    auto pIt = positive_start;
    while (pIt != positive_end) {
        auto copy = pIt;
        do {
            ++copy;
        } while (copy != positive_end && copy->first == pIt->first);

        const Float_ jump = copy - pIt;
        const Float_ mean_rank = cur_rank + static_cast<Float_>(jump - 1) / static_cast<Float_>(2) - center_rank;
        while (pIt != copy) {
            workspace.emplace_back(pIt->second, mean_rank);
            ++pIt;
        }

        sum_squares += mean_rank * mean_rank * jump;
        cur_rank += jump;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    if (sum_squares == 0) {
        workspace.clear();
        zprocess(0);
        return false;
    }

    const Float_ mult = sum_squares_to_mult(sum_squares);
    zprocess(zero_rank * mult);
    for (auto& nz : workspace) {
        nzprocess(nz, nz.second * mult);
    }

    return true;
}

// Compute scaled ranks from two iterator ranges of sparse ranked vectors containing negative and positive values.
// It then expands this to a dense vector of scaled ranks in 'output'.
// The return value is whether there are any non-zero values in the scaled ranks.
// Technically, the centered ranks of the non-zero elements are stored in 'workspace', but this is an implementation detail and the workspace should mostly be ignored on output;
// it's only provided to avoid reallocations within multiple calls to this function.
template<typename Index_, typename Stat_, typename Float_>
bool scaled_ranks_sparse(
    const Index_ num_markers,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_start,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_end,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_start,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_end,
    std::vector<std::pair<Index_, Float_> >& workspace,
    Float_* output
) { 
    return scaled_ranks_sparse<Index_, Stat_, Float_>(
        num_markers,
        negative_end - negative_start,
        negative_start,
        negative_end,
        positive_end - positive_start,
        positive_start,
        positive_end,
        workspace,
        [&](const Float_ zval) -> void {
            std::fill_n(output, num_markers, zval);
        },
        [&](std::pair<Index_, Float_>& pair, const Float_ val) -> void {
            output[pair.first] = val;
        }
    );
}

template<typename Index_, typename Float_>
struct SparseScaled {
    std::vector<std::pair<Index_, Float_> > nonzero;
    Float_ zero = 0;
};

// Compute scaled ranks from two iterator ranges of sparse ranked vectors containing negative and positive values.
// Specifically, this stores the scaled ranks of the non-zero elements in 'output.nonzero'.
// It also stores the scaled rank corresponding to the zero values in 'output.zero'.
// The return value is whether there are any non-zero values in the scaled ranks.
template<typename Index_, typename Stat_, typename Float_>
bool scaled_ranks_sparse(
    const Index_ num_markers,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_start,
    const typename RankedVector<Stat_, Index_>::const_iterator negative_end,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_start,
    const typename RankedVector<Stat_, Index_>::const_iterator positive_end,
    SparseScaled<Index_, Float_>& output
) { 
    return scaled_ranks_sparse<Index_, Stat_, Float_>(
        num_markers,
        negative_end - negative_start,
        negative_start,
        negative_end,
        positive_end - positive_start,
        positive_start,
        positive_end,
        output.nonzero,
        [&](const Float_ zval) -> void {
            output.zero = zval;
        },
        [&](std::pair<Index_, Float_>& pair, const Float_ val) -> void {
            pair.second = val;
        }
    );
}

// Overload that accepts two sparse ranked vectors, rather than the iterator ranges.
template<typename Index_, typename Stat_, typename Float_>
bool scaled_ranks_sparse(
    const Index_ num_markers,
    const RankedVector<Stat_, Index_>& negative,
    const RankedVector<Stat_, Index_>& positive,
    SparseScaled<Index_, Float_>& output
) { 
    return scaled_ranks_sparse<Index_, Stat_, Float_>(
        num_markers,
        negative.size(),
        negative.begin(),
        negative.end(),
        positive.size(),
        positive.begin(),
        positive.end(),
        output.nonzero,
        [&](const Float_ zval) -> void {
            output.zero = zval;
        },
        [&](std::pair<Index_, Float_>& pair, const Float_ val) -> void {
            pair.second = val;
        }
    );
}

// Simplifies the ranks to replace arbitrary statistics with an integer for greater memory efficiency.
// The general idea is that Simple_ is smaller than Stat_, to save space.
// As long as we respect ties, everything should be fine.
template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(
    const Index_ size,
    const typename RankedVector<Stat_, Index_>::const_iterator start,
    const typename RankedVector<Stat_, Index_>::const_iterator end,
    RankedVector<Simple_, Index_>& output
) {
    assert(sanisizer::is_equal(size, end - start));
    if (size == 0) {
        return;
    }

    output.reserve(size);

    Simple_ counter = 0;
    auto last = start->first;
    for (auto it = start; it < end; ++it) {
        if (it->first != last) {
            ++counter;
            last = it->first;
        }
        output.emplace_back(counter, it->second);
    }
}

// Overload that accepts an iterator range.
template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(
    const typename RankedVector<Stat_, Index_>::const_iterator start,
    const typename RankedVector<Stat_, Index_>::const_iterator end,
    RankedVector<Simple_, Index_>& output
) {
    simplify_ranks<Stat_, Index_, Simple_>(end - start, start, end, output);
}

// Overload that accepts a ranked vector.
template<typename Stat_, typename Index_, typename Simple_>
void simplify_ranks(const RankedVector<Stat_, Index_>& x, RankedVector<Simple_, Index_>& output) {
    simplify_ranks<Stat_, Index_, Simple_>(x.size(), x.begin(), x.end(), output);
}

}

#endif
