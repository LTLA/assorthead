#ifndef SINGLEPP_L2_HPP
#define SINGLEPP_L2_HPP

#include <cmath>
#include <cassert>
#include <limits>
#include <vector>
#include <algorithm>

#include "utils.hpp"
#include "scaled_ranks.hpp"

namespace singlepp {

// L2 between two dense vectors.
template<typename Index_, typename Float_>
Float_ dense_l2(const Index_ num_markers, const Float_* vec1, const Float_* vec2) {
    Float_ l2 = 0;
    for (Index_ d = 0; d < num_markers; ++d) {
        const Float_ delta = vec1[d] - vec2[d]; 
        l2 += delta * delta;
    }
    return l2;
}

// Compute the scaled ranks from 'ref' and compute the L2 to 'query'.
// This is the same as 'scaled_ranks_dense()' followed by 'dense_l2()' but has fewer passes through 'ref'.
template<typename Index_, typename Float_, typename Stat_>
Float_ scaled_ranks_dense_l2(const Index_ num_markers, const Float_* query, const RankedVector<Stat_, Index_>& ref, Float_* buffer) {
    Float_ l2 = 0;
    scaled_ranks_dense(
        num_markers,
        ref,
        buffer,
        [&](const Index_ i, const Float_ val) -> void {
            const Float_ delta = val - query[i];
            l2 += delta * delta;
        }
    );
    return l2;
}

template<typename Index_, typename Float_>
Index_ get_sparse_num(const SparseScaled<Index_, Float_>& x) { return x.nonzero.size(); }

template<typename Index_, typename Float_>
Index_ get_sparse_index(const SparseScaled<Index_, Float_>& x, const Index_ i) { return x.nonzero[i].first; }

template<typename Index_, typename Float_>
Float_ get_sparse_value(const SparseScaled<Index_, Float_>& x, const Index_ i) { return x.nonzero[i].second; }

template<typename Index_, typename Float_>
Float_ get_sparse_zero(const SparseScaled<Index_, Float_>& x) { return x.zero; }

// Compute the L2 between an abstract dense vector and a sparse vector. 
// The abstract dense vector is represented by the 'get_densified' argument, which is a function that accepts an index and returns the scaled rank at that position. 
// The sparse vector should contain the scaled ranks of the non-zero elements as well as the scaled rank of the zero value.
template<typename Float_, typename Index_, class GetDensified_, class SparseVec_>
Float_ internal_sparse_l2(const Index_ num_markers, const GetDensified_ get_densified, const bool densified_has_nonzero, const SparseVec_& sparse_vec) {
    const auto num_ref = get_sparse_num(sparse_vec);
    const auto zero_ref = get_sparse_zero(sparse_vec);
    assert(sanisizer::is_greater_than_or_equal(num_markers, num_ref));

    Float_ sum = 0;
    for (Index_ ir = 0; ir < num_ref; ++ir) {
        const auto val_ref = get_sparse_value(sparse_vec, ir);
        const auto augmented = val_ref - zero_ref;
        const auto val_densified = get_densified(get_sparse_index(sparse_vec, ir));
        sum += augmented * (augmented - 2 * val_densified);
    }

    return static_cast<Float_>(densified_has_nonzero ? 0.25 : 0) + sum - num_markers * zero_ref * zero_ref;
}

// Compute the L2 between a dense vector and a sparse vector, both of which contain scaled ranks.
template<typename Index_, typename Float_, typename SparseVec_>
Float_ sparse_l2(const Index_ num_markers, const Float_* densified, const bool densified_has_nonzero, const SparseVec_& sparse_vec) {
    return internal_sparse_l2<Float_>(
        num_markers, 
        [&](const Index_ i) -> Float_ { return densified[i]; },
        densified_has_nonzero,
        sparse_vec
    );
}

// Compute the scaled ranks from sparse reference vectors of negative and positive values, and then compute its L2 against a dense 'query' vector of scaled ranks. 
// This is the same as 'scaled_ranks_sparse()' followed by 'sparse_l2()' but avoids an unnecessary pass through the non-zero elements.
// 'workspace' should be ignored on output, it's only provided as an argument here to avoid reallocation in multiple calls.
template<typename Index_, typename Float_, typename Stat_>
Float_ scaled_ranks_sparse_l2(
    const Index_ num_markers,
    const Float_* query,
    const bool query_has_nonzero,
    const RankedVector<Stat_, Index_>& negative_ref,
    const RankedVector<Stat_, Index_>& positive_ref,
    std::vector<std::pair<Index_, Float_> >& workspace
) {
    Float_ zero_rank;
    Float_ sum = 0;

    scaled_ranks_sparse<Index_, Stat_, Float_>(
        num_markers,
        static_cast<Index_>(negative_ref.size()),
        negative_ref.begin(),
        negative_ref.end(),
        static_cast<Index_>(positive_ref.size()),
        positive_ref.begin(),
        positive_ref.end(),
        workspace,
        /* zero processing */ [&](const Float_ zval) -> void {
            zero_rank = zval;
        },
        /* non-zero processing */ [&](std::pair<Index_, Float_>& pair, const Float_ val) -> void {
            const auto augmented = val - zero_rank;
            const auto val_query = query[pair.first];
            sum += augmented * (augmented - 2 * val_query);
        }
    );

    return static_cast<Float_>(query_has_nonzero ? 0.25 : 0) + sum - num_markers * zero_rank * zero_rank;
}

// Compute the scaled ranks from a dense 'ref' vector, and then compute its L2 against a sparse 'query' vector of scaled ranks. 
// This is the same as 'scaled_ranks_dense()' followed by 'sparse_l2()' but avoids an unnecessary pass through the non-zero elements.
// 'workspace' should be ignored on output, it's only provided as an argument here to avoid reallocation in multiple calls.
template<typename Index_, typename Float_, typename Stat_>
Float_ scaled_ranks_sparse_l2(
    const Index_ num_markers,
    const SparseScaled<Index_, Float_>& query,
    const RankedVector<Stat_, Index_>& ref,
    Float_* buffer
) {
    const auto ss = centered_ranks_dense(num_markers, ref, buffer);
    const Float_ mult = (ss ? sum_squares_to_mult(ss) : 0);
    return internal_sparse_l2<Float_>(
        num_markers, 
        [&](const Index_ i) -> Float_ { return mult * buffer[i]; },
        ss > 0,
        query
    );
}

template<typename Index_, class SparseInput_, typename Float_>
bool densify_sparse_vector(const Index_ num_markers, const SparseInput_& vec, std::vector<Float_>& buffer) {
    assert(sanisizer::is_greater_than_or_equal(buffer.size(), num_markers));
    std::fill_n(buffer.data(), num_markers, get_sparse_zero(vec));
    const auto num = get_sparse_num(vec);
    for (Index_ i = 0; i < num; ++i) {
        buffer[get_sparse_index(vec, i)] = get_sparse_value(vec, i);
    }
    return num > 0;
}

}

#endif
