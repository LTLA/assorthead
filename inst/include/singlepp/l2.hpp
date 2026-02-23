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

template<typename Index_, typename Float_>
Float_ dense_l2(const Index_ num_markers, const Float_* vec1, const Float_* vec2) {
    Float_ l2 = 0;
    for (Index_ d = 0; d < num_markers; ++d) {
        const Float_ delta = vec1[d] - vec2[d]; 
        l2 += delta * delta;
    }
    return l2;
}

template<typename Index_, typename Float_, typename Stat_>
Float_ scaled_ranks_dense_l2(const Index_ num_markers, const Float_* query, const RankedVector<Stat_, Index_>& collected, Float_* buffer) {
    Float_ l2 = 0;
    scaled_ranks_dense(
        num_markers,
        collected,
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

template<typename Index_, typename Float_, typename SparseRef_>
Float_ sparse_l2(const Index_ num_markers, const Float_* query, const bool query_has_nonzero, const SparseRef_& ref_vec) {
    const auto num_ref = get_sparse_num(ref_vec);
    const auto zero_ref = get_sparse_zero(ref_vec);
    assert(sanisizer::is_greater_than_or_equal(num_markers, num_ref));

    Float_ sum = 0;
    for (Index_ ir = 0; ir < num_ref; ++ir) {
        const auto val_ref = get_sparse_value(ref_vec, ir);
        const auto augmented = val_ref - zero_ref;
        const auto val_query = query[get_sparse_index(ref_vec, ir)];
        sum += augmented * (augmented - 2 * val_query);
    }

    return static_cast<Float_>(query_has_nonzero ? 0.25 : 0) + sum - num_markers * zero_ref * zero_ref;
}

template<typename Index_, typename Float_, typename Stat_>
Float_ scaled_ranks_sparse_l2(
    const Index_ num_markers,
    const Float_* query,
    const bool query_has_nonzero,
    const RankedVector<Stat_, Index_>& negative_ref,
    const RankedVector<Stat_, Index_>& positive_ref,
    std::vector<std::pair<Index_, Float_> >& buffer_ref
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
        buffer_ref,
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
