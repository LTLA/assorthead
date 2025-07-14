#ifndef TATAMI_CHUNKED_SLAB_CACHE_STATS_HPP
#define TATAMI_CHUNKED_SLAB_CACHE_STATS_HPP

#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

/**
 * @file SlabCacheStats.hpp
 * @brief Slab cache statistics.
 */

namespace tatami_chunked {

/**
 * @brief Statistics for slab caching.
 *
 * This computes the slab size and the number of slabs to be cached, given the dimensions of the slab and the cache size in bytes.
 * The assumption is that all slabs are of the same shape, partitioning the matrix into regular intervals along the target dimension.
 * Developers should check out `CustomDenseChunkedMatrix` for some usage examples.
 *
 * @tparam MaxSlabs_ Integer type of the maximum number of slabs.
 */
template<typename MaxSlabs_>
struct SlabCacheStats {
    /**
     * Size of each slab, in terms of the number of data elements.
     */
    std::size_t slab_size_in_elements;

    /**
     * Number of slabs that can fit in the cache.
     * This is used as `max_slabs` in `LruSlabCache`, `OracularSlabCache` and friends.
     */
    MaxSlabs_ max_slabs_in_cache;

    /**
     * @tparam Index_ Integer type of the dimension extents.
     * @tparam TargetNumSlabs_ Integer type of the number of slabs along the target dimension.
     *
     * @param target_length Length of the target dimension of each slab.
     * For example, if we were iterating through rows of a matrix, `target_length` would be the number of rows spanned by each slab.
     * @param non_target_length Length of the non-target dimension of each slab.
     * For example, if we were iterating through matrix rows, the `non_target_length` would be the number of columns spanned by each slab.
     * @param target_num_slabs Number of slabs required to span the full extent of the target dimension of the matrix.
     * This is used as an upper bound on the value of `max_slabs_in_cache`.
     * For example, if we were iterating through matrix rows, the `target_num_slabs` would be the number of slabs required to span all rows of the matrix.
     * @param cache_size_in_elements Total size of the cache, in terms of the number of data elements.
     * @param require_minimum_cache Whether to enforce a minimum size of the cache for efficient extraction of consecutive dimension elements, even if it exceeds `cache_size_in_elements`.
     */
    template<typename Index_, typename TargetNumSlabs_>
    SlabCacheStats(Index_ target_length, Index_ non_target_length, TargetNumSlabs_ target_num_slabs, std::size_t cache_size_in_elements, bool require_minimum_cache) :
        // Don't be tempted to do unsafe casts of target_length to size_t,
        // as this class might be used outside of the tatami::Matrix contract (i.e., Index_ might store values beyond std::size_t).
        slab_size_in_elements(sanisizer::product<std::size_t>(target_length, non_target_length)),
        max_slabs_in_cache(compute_max_slabs_in_cache(slab_size_in_elements, target_num_slabs, cache_size_in_elements, require_minimum_cache))
    {}

    /**
     * @tparam Index_ Integer type of the dimension extents.
     * @tparam TargetNumSlabs_ Integer type of the number of slabs along the target dimension.
     *
     * @param target_length Length of the target dimension of each slab.
     * For example, if we were iterating through rows of a matrix, `target_length` would be the number of rows spanned by each slab.
     * @param non_target_length Length of the non-target dimension of each slab.
     * For example, if we were iterating through matrix rows, the `non_target_length` would be the number of columns spanned by each slab.
     * @param target_num_slabs Number of slabs required to span the full extent of the target dimension of the matrix.
     * This is used as an upper bound on the value of `max_slabs_in_cache`.
     * For example, if we were iterating through matrix rows, the `target_num_slabs` would be the number of slabs required to span all rows of the matrix.
     * @param cache_size_in_bytes Total size of the cache, in terms of the number of bytes.
     * @param element_size Size of each data element in the cache, in bytes.
     * This may be zero, e.g., when neither the value nor the index are required during sparse extraction.
     * @param require_minimum_cache Whether to enforce a minimum size of the cache for efficient extraction of consecutive dimension elements, even if it exceeds `cache_size_in_bytes`.
     */
    template<typename Index_, typename TargetNumSlabs_>
    SlabCacheStats(Index_ target_length, Index_ non_target_length, TargetNumSlabs_ target_num_slabs, std::size_t cache_size_in_bytes, std::size_t element_size, bool require_minimum_cache) :
        slab_size_in_elements(sanisizer::product<std::size_t>(target_length, non_target_length)),
        max_slabs_in_cache([&]{
            if (element_size == 0) {
                return sanisizer::cap<MaxSlabs_>(target_num_slabs);
            } else {
                return compute_max_slabs_in_cache(slab_size_in_elements, target_num_slabs, cache_size_in_bytes / element_size, require_minimum_cache); 
            }
        }())
    {}

private:
    template<typename NumSlabs_>
    static MaxSlabs_ compute_max_slabs_in_cache(std::size_t slab_size_in_elements, NumSlabs_ num_slabs, std::size_t cache_size_in_elements, bool require_minimum_cache) {
        if (slab_size_in_elements == 0) {
            return sanisizer::cap<MaxSlabs_>(num_slabs);
        }

        auto tmp = cache_size_in_elements / slab_size_in_elements;
        if (tmp == 0 && require_minimum_cache) {
            return 1;
        } 

        if (sanisizer::is_less_than_or_equal(tmp, num_slabs)) {
            return sanisizer::cast<MaxSlabs_>(tmp);
        } else {
            return sanisizer::cast<MaxSlabs_>(num_slabs);
        }
    }
};

}

#endif
