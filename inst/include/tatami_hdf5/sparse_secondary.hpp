#ifndef TATAMI_HDF5_SPARSE_SECONDARY_HPP
#define TATAMI_HDF5_SPARSE_SECONDARY_HPP

#include "sparse_primary.hpp"
#include "serialize.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>
#include <type_traits>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "tatami_chunked/tatami_chunked.hpp"
#include "sanisizer/sanisizer.hpp"

namespace tatami_hdf5 {

// For a compressed sparse matrix, secondary extraction involves extracting elements of the "secondary" dimension, i.e., the one that is not used for grouping non-zero values. 
// So, for a CSC matrix, this would involve extraction of individual rows; for CSR matrices, columns instead.
// This cannot be efficiently done without reading through the entire file, but we can mitigate this pain by caching.
//
// In the myopic case, we allocate one big slab based on what our cache size can afford.
// We then scan through the file and cache all values/indices that fall within the chunk interval spanned by the slab.
// If the next request doesn't fall inside the chunk interval, we define a new chunk interval and repeat this process.
// This strategy reduces the number of full file scans to the extent of the secondary dimension divided by the chunk length.
//
// For the oracular case, we also allocate a big slab, but this time, we know exactly which secondary dimension elements are requested.
// So for each prediction cycle, we perform a full file scan and accumulate the values/indices that correspond to the requested elements.

namespace CompressedSparseMatrix_internal {

template<typename CachedValue_, typename Index_>
Index_ choose_chunk_length_for_myopic_secondary(const MatrixDetails<Index_>& details, Index_ primary_extract_length, bool needs_value, bool needs_index) {
    std::size_t elsize = CompressedSparseMatrix_internal::size_of_cached_element<CachedValue_, Index_>(needs_value, needs_index);
    if (elsize == 0 || primary_extract_length == 0) {
        return details.secondary_dim; // caching the entire secondary dimension, if possible.
    }

    std::size_t secondary_chunkdim = (details.slab_cache_size / elsize) / primary_extract_length;
    if (secondary_chunkdim < 1) {
        return 1; // ensure we have at least one dimension.
    }

    return std::min(secondary_chunkdim, static_cast<std::size_t>(details.secondary_dim));
}

// We don't use CachedIndex_, as this may be too small to store the other
// dimension; use Index_ instead as this is guaranteed.
template<typename Index_, typename CachedValue_>
class MyopicSecondaryCore {
public:
    MyopicSecondaryCore(
        const MatrixDetails<Index_>& details,
        tatami::MaybeOracle<false, Index_>, // oracle, for consistency with the oracular constructor.
        Index_ primary_extract_length,
        bool needs_value,
        bool needs_index) :
        my_pointers(details.pointers),
        my_secondary_dim_stats(details.secondary_dim, choose_chunk_length_for_myopic_secondary<CachedValue_>(details, primary_extract_length, needs_value, needs_index)),
        my_primary_extract_length(primary_extract_length),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {
        initialize(details, my_h5comp);

        auto cache_size_in_elements = sanisizer::product<std::size_t>(my_secondary_dim_stats.chunk_length, primary_extract_length);
        if (my_needs_value) {
            my_cache_data.resize(sanisizer::cast<decltype(my_cache_data.size())>(cache_size_in_elements));
        }
        if (my_needs_index) {
            my_cache_index.resize(sanisizer::cast<decltype(my_cache_index.size())>(cache_size_in_elements));
        }
        tatami::resize_container_to_Index_size(my_cache_count, my_secondary_dim_stats.chunk_length);

        // Precomputing the offsets so we don't have to do the multiplication every time.
        my_cache_offsets.reserve(my_secondary_dim_stats.chunk_length);
        std::size_t current_offset = 0;
        for (Index_ i = 0; i < my_secondary_dim_stats.chunk_length; ++i, current_offset += primary_extract_length) {
            my_cache_offsets.push_back(current_offset);
        }

        // Protect pointer differences against overflow when refining primary limits.
        sanisizer::can_ptrdiff<decltype(my_index_buffer.begin())>(my_secondary_dim_stats.dimension_extent);

        // Check that resizing will work correctly for these vectors.
        tatami::can_cast_Index_to_container_size<decltype(my_index_buffer)>(my_secondary_dim_stats.dimension_extent);
        tatami::can_cast_Index_to_container_size<decltype(my_data_buffer)>(my_secondary_dim_stats.dimension_extent);
    }

    ~MyopicSecondaryCore() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    const std::vector<hsize_t>& my_pointers;
    tatami_chunked::ChunkDimensionStats<Index_> my_secondary_dim_stats;
    Index_ my_primary_extract_length;
    bool my_needs_value;
    bool my_needs_index;

    std::vector<Index_> my_index_buffer;
    std::vector<CachedValue_> my_data_buffer;

    Index_ my_last_chunk_id = 0;
    std::vector<Index_> my_cache_index;
    std::vector<CachedValue_> my_cache_data;
    std::vector<Index_> my_cache_count;
    std::vector<std::size_t> my_cache_offsets;
    bool my_first = true;

private:
    template<class Extract_>
    tatami::SparseRange<CachedValue_, Index_> fetch_raw(Index_ i, Extract_ extract) {
        Index_ chunk_id = i / my_secondary_dim_stats.chunk_length;
        Index_ chunk_offset = i % my_secondary_dim_stats.chunk_length;

        if (chunk_id != my_last_chunk_id || my_first) { 
            Index_ clen = tatami_chunked::get_chunk_length(my_secondary_dim_stats, chunk_id);
            std::fill_n(my_cache_count.begin(), clen, 0);

            serialize([&]() -> void {
                extract(chunk_id * my_secondary_dim_stats.chunk_length, clen);
            });
            my_last_chunk_id = chunk_id;
            my_first = false;
        }

        tatami::SparseRange<CachedValue_, Index_> output(my_cache_count[chunk_offset]);
        auto offset = sanisizer::product_unsafe<std::size_t>(chunk_offset, my_primary_extract_length);
        if (my_needs_value) {
            output.value = my_cache_data.data() + offset;
        }
        if (my_needs_index) {
            output.index = my_cache_index.data() + offset;
        }
        return output;
    }

    // Serial locks should be applied by the callers before calling this.
    void extract_and_append(Index_ primary, Index_ secondary_start, Index_ secondary_length, Index_ primary_to_add) {
        auto left = my_pointers[primary], right = my_pointers[primary + 1];
        hsize_t count = right - left;
        if (count == 0) {
            return;
        }
        my_index_buffer.resize(count);

        my_h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        my_h5comp->memspace.setExtentSimple(1, &count);
        my_h5comp->memspace.selectAll();
        my_h5comp->index_dataset.read(my_index_buffer.data(), define_mem_type<Index_>(), my_h5comp->memspace, my_h5comp->dataspace);

        auto start = my_index_buffer.begin(), end = my_index_buffer.end();
        refine_primary_limits(start, end, my_secondary_dim_stats.dimension_extent, secondary_start, secondary_start + secondary_length);

        if (my_needs_index) {
            for (auto x = start; x != end; ++x) {
                Index_ current = *x - secondary_start;
                my_cache_index[my_cache_offsets[current] + static_cast<std::size_t>(my_cache_count[current])] = primary_to_add;
            }
        }

        if (start != end && my_needs_value) {
            hsize_t better_left = left + (start - my_index_buffer.begin()); // pointer difference won't overflow as we checked can_ptrdiff() in the constructor.
            hsize_t better_count = end - start;
            my_h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &better_count, &better_left);
            my_h5comp->memspace.setExtentSimple(1, &better_count);
            my_h5comp->memspace.selectAll();

            my_data_buffer.resize(better_count);
            my_h5comp->data_dataset.read(my_data_buffer.data(), define_mem_type<CachedValue_>(), my_h5comp->memspace, my_h5comp->dataspace);

            for (auto x = start; x != end; ++x) {
                Index_ current = *x - secondary_start;
                my_cache_data[my_cache_offsets[current] + static_cast<std::size_t>(my_cache_count[current])] = my_data_buffer[x - start];
            }
        }

        for (auto x = start; x != end; ++x) {
            ++(my_cache_count[*x - secondary_start]);
        }
    }

public:
    // The store_index_ flag specifies whether we want to store the index of
    // the primary value (useful for dense extraction) or the primary value
    // itself (useful for sparse extraction).
    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_block(Index_ i, Index_ primary_start, Index_ primary_length) {
        return fetch_raw(
            i,
            [&](Index_ secondary_start, Index_ secondary_length) -> void {
                for (Index_ px = 0; px < primary_length; ++px) {
                    auto primary = px + primary_start;
                    extract_and_append(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
                }
            }
        );
    }

    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_indices(Index_ i, const std::vector<Index_>& primary_indices) {
        return fetch_raw(
            i,
            [&](Index_ secondary_start, Index_ secondary_length) -> void { 
                for (Index_ px = 0, end = primary_indices.size(); px < end; ++px) {
                    auto primary = primary_indices[px];
                    extract_and_append(primary, secondary_start, secondary_length, (store_index_ ? px : primary));
                }
            }
        );
    }
};

template<typename Index_, typename CachedValue_>
class OracularSecondaryCore {
public:
    OracularSecondaryCore(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ primary_extract_length,
        bool needs_value,
        bool needs_index) :
        my_pointers(details.pointers),
        my_secondary_dim(details.secondary_dim),
        my_primary_extract_length(primary_extract_length),
        my_needs_value(needs_value),
        my_needs_index(needs_index),
        my_cache(
            std::move(oracle),
            tatami_chunked::SlabCacheStats<Index_>(
                /* target_length = */ 1,
                /* non_target_length = */ primary_extract_length,
                /* target_num_chunks = */ my_secondary_dim,
                /* cache_size_in_bytes = */ details.slab_cache_size,
                /* element_size = */ size_of_cached_element<CachedValue_, Index_>(needs_value, needs_index),
                /* require_minimum_cache = */ true
            ).max_slabs_in_cache
        )
    {
        initialize(details, my_h5comp);

        auto alloc = sanisizer::product<std::size_t>(my_cache.get_max_slabs(), my_primary_extract_length);
        if (my_needs_index) {
            my_cache_index.resize(sanisizer::cast<decltype(my_cache_index.size())>(alloc));
        }
        if (my_needs_value) {
            my_cache_data.resize(sanisizer::cast<decltype(my_cache_data.size())>(alloc));
        }
        my_slab_ptrs.resize(tatami::cast_Index_to_container_size<decltype(my_slab_ptrs)>(my_secondary_dim), NULL);

        // Protect pointer differences against overflow when refining primary limits.
        sanisizer::can_ptrdiff<decltype(my_index_buffer.begin())>(my_secondary_dim);

        // Check that resizing will work correctly for these vectors.
        tatami::can_cast_Index_to_container_size<decltype(my_index_buffer)>(my_secondary_dim);
        tatami::can_cast_Index_to_container_size<decltype(my_data_buffer)>(my_secondary_dim);
    }

    ~OracularSecondaryCore() {
        destroy(my_h5comp);
    }

protected:
    std::unique_ptr<Components> my_h5comp;
    const std::vector<hsize_t>& my_pointers;

    Index_ my_secondary_dim;
    Index_ my_primary_extract_length;
    bool my_needs_value;
    bool my_needs_index;

    struct Slab {
        CachedValue_* value = NULL;
        Index_* index = NULL;
        Index_ number = 0;
    };
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

    // Contiguous data stores for the Slabs to point to. This avoids
    // the overhead of allocating a lot of little vectors.
    std::vector<Index_> my_cache_index;
    std::vector<CachedValue_> my_cache_data;
    std::size_t my_offset = 0;

    // Temporary buffers for the HDF5 library to read in values/indices for each dimension element.
    std::vector<Index_> my_index_buffer;
    std::vector<CachedValue_> my_data_buffer;

    // Some account-keeping intermediates to move data from the buffers to the cache.
    std::vector<Slab*> my_slab_ptrs;
    std::vector<CachedValue_*> my_value_ptrs;
    std::vector<Index_> my_found;

private:
    template<class Extract_>
    tatami::SparseRange<CachedValue_, Index_> fetch_raw(Extract_ extract) {
        auto out = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current, 0);
            }, 
            /* create = */ [&]() -> Slab {
                Slab latest;
                if (my_needs_value) {
                    latest.value = my_cache_data.data() + my_offset;
                }
                if (my_needs_index) {
                    latest.index = my_cache_index.data() + my_offset;
                }
                my_offset += my_primary_extract_length;
                return latest;
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                if (!chunks.empty()) {
                    Index_ first = my_secondary_dim, last = 0;
                    for (auto& c : chunks) {
                        my_slab_ptrs[c.first] = c.second;
                        first = std::min(first, c.first);
                        last = std::max(last, c.first);
                        c.second->number = 0;
                    }

                    serialize([&]() -> void {
                        extract(first, last + 1);
                    });

                    for (auto& c : chunks) {
                        my_slab_ptrs[c.first] = NULL;
                    }
                }
            }
        );
        return tatami::SparseRange(out.first->number, out.first->value, out.first->index);
    }

    // Serial locks should be applied by the callers before calling this.
    void extract_and_append(Index_ primary, Index_ secondary_first, Index_ secondary_last_plus_one, Index_ primary_to_add) {
        hsize_t left = my_pointers[primary], right = my_pointers[primary + 1];
        hsize_t count = right - left;
        if (count == 0) {
            return;
        }
        my_index_buffer.resize(count);

        my_h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, &count, &left);
        my_h5comp->memspace.setExtentSimple(1, &count);
        my_h5comp->memspace.selectAll();
        my_h5comp->index_dataset.read(my_index_buffer.data(), define_mem_type<Index_>(), my_h5comp->memspace, my_h5comp->dataspace);

        auto start = my_index_buffer.begin(), end = my_index_buffer.end();
        refine_primary_limits(start, end, my_secondary_dim, secondary_first, secondary_last_plus_one);

        if (my_needs_value) {
            my_value_ptrs.clear();
            my_found.clear();
        }

        for (auto x = start; x != end; ++x) {
            auto slab_ptr = my_slab_ptrs[*x];
            if (slab_ptr != NULL) {
                if (my_needs_index) {
                    slab_ptr->index[slab_ptr->number] = primary_to_add;
                }
                if (my_needs_value) {
                    my_value_ptrs.push_back(slab_ptr->value + slab_ptr->number);
                    my_found.push_back(x - start); // pointer subtraction is safe as we checked can_ptrdiff in the constructor.
                }
                ++(slab_ptr->number);
            }
        }

        if (!my_found.empty()) {
            hsize_t new_start = left + (start - my_index_buffer.begin());
            my_h5comp->dataspace.selectNone();
            tatami::process_consecutive_indices<Index_>(
                my_found.data(),
                my_found.size(),
                [&](Index_ start, Index_ length) -> void {
                    hsize_t offset = start + new_start;
                    hsize_t count = length;
                    my_h5comp->dataspace.selectHyperslab(H5S_SELECT_OR, &count, &offset);
                }
            );

            hsize_t new_len = my_found.size();
            my_h5comp->memspace.setExtentSimple(1, &new_len);
            my_h5comp->memspace.selectAll();

            my_data_buffer.resize(new_len);
            my_h5comp->data_dataset.read(my_data_buffer.data(), define_mem_type<CachedValue_>(), my_h5comp->memspace, my_h5comp->dataspace);
            for (hsize_t i = 0; i < new_len; ++i) {
                *(my_value_ptrs[i]) = my_data_buffer[i];
            }
        }
    }

public:
    // The store_index_ flag specifies whether we want to store the index of
    // the primary value (useful for dense extraction) or the primary value
    // itself (useful for sparse extraction).
    //
    // The first argument here is for compile-time polymorphism with the
    // myopic counterpart, and is not actually needed itself.
    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_block(Index_, Index_ primary_start, Index_ primary_length) {
        const auto& info = fetch_raw([&](Index_ secondary_first, Index_ secondary_last_plus_one) -> void {
            for (Index_ px = 0; px < primary_length; ++px) {
                auto primary = px + primary_start;
                extract_and_append(primary, secondary_first, secondary_last_plus_one, (store_index_ ? px : primary));
            }
        });
        return tatami::SparseRange<CachedValue_, Index_>(info.number, info.value, info.index);
    }

    template<bool store_index_>
    tatami::SparseRange<CachedValue_, Index_> fetch_indices(Index_, const std::vector<Index_>& primary_indices) {
        const auto& info = fetch_raw([&](Index_ secondary_first, Index_ secondary_last_plus_one) -> void {
            for (Index_ px = 0, end = primary_indices.size(); px < end; ++px) {
                auto primary = primary_indices[px];
                extract_and_append(primary, secondary_first, secondary_last_plus_one, (store_index_ ? px : primary));
            }
        });
        return tatami::SparseRange<CachedValue_, Index_>(info.number, info.value, info.index);
    }
};

template<bool oracle_, typename Index_, typename CachedValue_>
using ConditionalSecondaryCore = typename std::conditional<oracle_, OracularSecondaryCore<Index_, CachedValue_>, MyopicSecondaryCore<Index_, CachedValue_> >::type;

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryFullSparse final : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SecondaryFullSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), details.primary_dim, needs_value, needs_index),
        my_primary_dim(details.primary_dim)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto cached = my_core.template fetch_block<false>(i, 0, my_primary_dim);
        return slab_to_sparse(cached, value_buffer, index_buffer); 
    }

private:
    ConditionalSecondaryCore<oracle_, Index_, CachedValue_> my_core;
    Index_ my_primary_dim;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryFullDense final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    SecondaryFullDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle) :
        my_core(details, std::move(oracle), details.primary_dim, true, true),
        my_primary_dim(details.primary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = my_core.template fetch_block<true>(i, 0, my_primary_dim);
        return slab_to_dense(cached, buffer, my_primary_dim);
    }

private:
    ConditionalSecondaryCore<oracle_, Index_, CachedValue_> my_core;
    Index_ my_primary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryBlockSparse final : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SecondaryBlockSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), block_length, needs_value, needs_index),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto cached = my_core.template fetch_block<false>(i, my_block_start, my_block_length);
        return slab_to_sparse(cached, value_buffer, index_buffer);
    }

private:
    ConditionalSecondaryCore<oracle_, Index_, CachedValue_> my_core;
    Index_ my_block_start;
    Index_ my_block_length;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryBlockDense final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    SecondaryBlockDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length) :
        my_core(details, std::move(oracle), block_length, true, true),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = my_core.template fetch_block<true>(i, my_block_start, my_block_length);
        return slab_to_dense(cached, buffer, my_block_length);
    }

private:
    ConditionalSecondaryCore<oracle_, Index_, CachedValue_> my_core;
    Index_ my_block_start;
    Index_ my_block_length;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryIndexSparse final : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SecondaryIndexSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), indices_ptr->size(), needs_value, needs_index),
        my_indices_ptr(std::move(indices_ptr))
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto cached = my_core.template fetch_indices<false>(i, *my_indices_ptr);
        return slab_to_sparse(cached, value_buffer, index_buffer);
    }

private:
    ConditionalSecondaryCore<oracle_, Index_, CachedValue_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_>
class SecondaryIndexDense final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    SecondaryIndexDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr) :
        my_core(details, std::move(oracle), indices_ptr->size(), true, true),
        my_indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto cached = my_core.template fetch_indices<true>(i, *my_indices_ptr);
        return slab_to_dense(cached, buffer, static_cast<Index_>(my_indices_ptr->size()));
    }

private:
    ConditionalSecondaryCore<oracle_, Index_, CachedValue_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr;
};

}

}

#endif
