#ifndef TATAMI_HDF5_SPARSE_PRIMARY_HPP
#define TATAMI_HDF5_SPARSE_PRIMARY_HPP

#include "sparse_utils.hpp"
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

// For a compressed sparse matrix, primary extraction involves extracting elements of the "primary" dimension, i.e., that along which non-zero entries are grouped.
// So, for a CSC matrix, this would involve extraction of individual columns; for CSR matrices, rows instead.
// This can be efficiently done as it involves a single contiguous read from the file.
//
// In the myopic case, we cache slabs where each slab contains one primary dimension element.
// The slab itself has enough memory to hold the maximum number of zeros in any dimension element, to ensure that any slab can hold any dimension element.
// This is initially conservative but slabs are re-used across requests, and with enough re-use, eventually all slabs will have capacity equal to the maximum number of zeros.
//
// In the oracular case, we use variable slabs where we read an entire stretch of values/indices from file at once according to the predictions.
// The cache memory pool is contiguous, so in the best case, this amounts to a single read call through the HDF5 library.
// Some shuffling is required to handle situations where the same primary dimension element is re-used or only a subset of the element is required.

namespace CompressedSparseMatrix_internal {

// Unfortunately we can't use tatami::SparseRange, as CachedIndex_ might not be
// large enough to represent the number of cached indices, e.g., if there were
// 256 indices, a CachedIndex_=uint8_t type would be large enough to represent
// each index (0 - 255) but not the number of indices.
template<typename Index_, typename CachedValue_, typename CachedIndex_ = Index_>
struct PrimarySlab { 
    CachedValue_* value = NULL;
    CachedIndex_* index = NULL;
    Index_ number = 0;
};

/*************************************
 **** Subset extraction utilities ****
 *************************************/

template<bool sparse_, typename Index_>
using SparsePrimaryRemapVector = typename std::conditional<sparse_, std::vector<unsigned char>, std::vector<Index_> >::type;

template<bool sparse_, typename Index_>
void populate_sparse_primary_remap_vector(const std::vector<Index_>& indices, SparsePrimaryRemapVector<sparse_, Index_>& remap, Index_& first_index, Index_& past_last_index) {
    if (indices.empty()) {
        first_index = 0;
        past_last_index = 0;
        return;
    }

    first_index = indices.front();
    past_last_index = indices.back() + 1; // increment is safe as Index_ should be large enough to store the dimension extent.
    tatami::resize_container_to_Index_size(remap, past_last_index - first_index);

    if constexpr(sparse_) {
        for (auto i : indices) {
            remap[i - first_index] = 1;
        }
    } else {
        // We start from +1 so that we can still use 0 as an 'absent' counter.
        // This should not overflow as Index_ should be large enough to hold
        // the dimension extent so it should be able to tolerate +1.
        Index_ counter = 1;
        for (auto i : indices) {
            remap[i - first_index] = counter;
            ++counter;
        }
    }
}

template<bool sparse_, typename In_, typename Index_, typename Output_, typename Map_>
Index_ scan_for_indices_in_sparse_primary_remap_vector(
    In_ indices_start, 
    In_ indices_end,
    Index_ first_index,
    Output_ output, 
    std::vector<Index_>& found,
    const std::vector<Map_>& remap,
    bool needs_value,
    bool needs_index) 
{
    Index_ counter = 0;
    found.clear();
    Index_ num_found = 0;

    for (auto x = indices_start; x != indices_end; ++x, ++counter) {
        auto present = remap[*x - first_index];
        if (present) {
            if (needs_index) {
                if constexpr(sparse_) {
                    *output = *x;
                } else {
                    // For dense extraction, we store the position on 'indices', to
                    // make life easier when filling up the output vector.
                    // Remember that we +1'd in 'populate_sparse_primary_remap_vector',
                    // so we have to undo it here.
                    *output = present - 1;
                }
                ++output;
            }
            if (needs_value) {
                found.push_back(counter);
            }
            ++num_found;
        }
    }

    return num_found;
}

/************************************
 **** LRU base extractor classes ****
 ************************************/

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruCoreBase {
public:
    PrimaryLruCoreBase(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_>, Index_ max_non_zeros, bool needs_value, bool needs_index) : 
        my_pointers(details.pointers),
        my_cache([&]() -> Index_ {
            return tatami_chunked::SlabCacheStats<Index_>(
                /* target_length = */ 1,
                /* non_target_length = */ max_non_zeros,
                /* target_num_chunks = */ details.primary_dim,
                /* cache_size_in_bytes = */ details.slab_cache_size,
                /* element_size = */ size_of_cached_element<CachedValue_, CachedIndex_>(needs_value, needs_index),
                /* require_minimum_cache = */ true
            ).max_slabs_in_cache;
        }()),
        my_needs_value(needs_value),
        my_needs_index(needs_index),
        my_max_non_zeros(max_non_zeros)
    {
        initialize(details, my_h5comp);

        auto pool_size = sanisizer::product<std::size_t>(max_non_zeros, my_cache.get_max_slabs());
        if (needs_value) {
            my_value_pool.resize(sanisizer::cast<decltype(my_value_pool.size())>(pool_size));
        }
        if (needs_index) {
            my_index_pool.resize(sanisizer::cast<decltype(my_index_pool.size())>(pool_size));
        }
    }

    ~PrimaryLruCoreBase() {
        destroy(my_h5comp);
    }

protected:
    std::unique_ptr<Components> my_h5comp;
    const std::vector<hsize_t>& my_pointers;
    tatami_chunked::LruSlabCache<Index_, PrimarySlab<Index_, CachedValue_, CachedIndex_> > my_cache;
    bool my_needs_value, my_needs_index;

private:
    std::vector<CachedValue_> my_value_pool;
    std::vector<CachedIndex_> my_index_pool;
    Index_ my_max_non_zeros;
    std::size_t my_offset = 0; // use size_t here as we're doing pointer arithmetic below.

public:
    PrimarySlab<Index_, CachedValue_, CachedIndex_> create() {
        PrimarySlab<Index_, CachedValue_, CachedIndex_> output;
        if (my_needs_value) {
            output.value = my_value_pool.data() + my_offset;
        }
        if (my_needs_index) {
            output.index = my_index_pool.data() + my_offset;
        }
        my_offset += my_max_non_zeros;
        return output;
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruFullCore : private PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryLruFullCore(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_> oracle, bool needs_value, bool needs_index) : 
        PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle),
            details.max_non_zeros,
            needs_value,
            needs_index
        )
    {}

public:
    const PrimarySlab<Index_, CachedValue_, CachedIndex_>& fetch_raw(Index_ i) {
        return this->my_cache.find(
            i, 
            /* create = */ [&]() -> PrimarySlab<Index_, CachedValue_, CachedIndex_> {
                return this->create();
            },
            /* populate = */ [&](Index_ i, PrimarySlab<Index_, CachedValue_, CachedIndex_>& current_cache) -> void {
                const auto& pointers = this->my_pointers; 
                hsize_t extraction_start = pointers[i];
                hsize_t extraction_len = pointers[i + 1] - pointers[i];
                current_cache.number = extraction_len;

                if (extraction_len == 0) {
                    return; 
                }
                if (!(this->my_needs_index || this->my_needs_value)) {
                    return;
                }

                serialize([&]() -> void {
                    auto& comp = *(this->my_h5comp);
                    comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    comp.memspace.setExtentSimple(1, &extraction_len);
                    comp.memspace.selectAll();
                    if (this->my_needs_index) {
                        comp.index_dataset.read(current_cache.index, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);
                    }
                    if (this->my_needs_value) {
                        comp.data_dataset.read(current_cache.value, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                    }
                });
            }
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruBlockCore : private PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryLruBlockCore(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_> oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_>(
            details,
            std::move(oracle), 
            std::min(details.max_non_zeros, block_length), // Tighten the bounds if we can, to fit more elements into the cache.
            needs_value, 
            needs_index
        ),
        my_secondary_dim(details.secondary_dim),
        my_block_start(block_start),
        my_block_past_end(block_start + block_length)
    {
        tatami::can_cast_Index_to_container_size<decltype(my_index_buffer)>(details.max_non_zeros); // Ensure that we can resize my_index_buffer safely.
        sanisizer::can_ptrdiff<decltype(my_index_buffer.begin())>(details.max_non_zeros); // Protect pointer differences against overflow when refining primary limits.
    }

private:
    Index_ my_secondary_dim;
    Index_ my_block_start, my_block_past_end;
    std::vector<CachedIndex_> my_index_buffer;

public:
    const PrimarySlab<Index_, CachedValue_, CachedIndex_>& fetch_raw(Index_ i) {
        return this->my_cache.find(
            i, 
            /* create = */ [&]() -> PrimarySlab<Index_, CachedValue_, CachedIndex_> {
                return this->create();
            },
            /* populate = */ [&](Index_ i, PrimarySlab<Index_, CachedValue_, CachedIndex_>& current_cache) -> void {
                const auto narrowed = narrow_primary_extraction_range(
                    this->my_pointers[i],
                    this->my_pointers[i + 1],
                    my_block_start,
                    my_block_past_end,
                    my_secondary_dim
                );
                const hsize_t extraction_start = narrowed.first;
                const hsize_t extraction_len = narrowed.second - extraction_start;
                if (extraction_len == 0) {
                    current_cache.number = 0;
                    return;
                }
                my_index_buffer.resize(extraction_len); // implicit cast is safe against overflows as we checked it in the constructor.

                serialize([&]() -> void {
                    auto& comp = *(this->my_h5comp);
                    comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    comp.memspace.setExtentSimple(1, &extraction_len);
                    comp.memspace.selectAll();
                    comp.index_dataset.read(my_index_buffer.data(), define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                    auto indices_start = my_index_buffer.begin();
                    auto indices_end = my_index_buffer.end();
                    refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_block_start, my_block_past_end);
                    current_cache.number = indices_end - indices_start; // pointer difference is safe against overflows as we checked it in the constructor.

                    if (current_cache.number) {
                        if (this->my_needs_index) {
                            std::copy(indices_start, indices_end, current_cache.index);
                            if constexpr(!sparse_) {
                                // For dense extraction, we subtract the block_start 
                                // to make life easier when filling up the output vector.
                                for (Index_ i = 0; i < current_cache.number; ++i) {
                                    current_cache.index[i] -= my_block_start;
                                }
                            }
                        }

                        if (this->my_needs_value) {
                            // Pointer difference is fine, we checked this in the constructor.
                            const hsize_t new_start = extraction_start + static_cast<hsize_t>(indices_start - my_index_buffer.begin()); 
                            const hsize_t new_len = current_cache.number;
                            comp.dataspace.selectHyperslab(H5S_SELECT_SET, &new_len, &new_start);
                            comp.memspace.setExtentSimple(1, &new_len);
                            comp.memspace.selectAll();
                            comp.data_dataset.read(current_cache.value, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                        }
                    }
                });
            }
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryLruIndexCore : private PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryLruIndexCore(const MatrixDetails<Index_>& details, tatami::MaybeOracle<false, Index_> oracle, const std::vector<Index_>& indices, bool needs_value, bool needs_index) : 
        PrimaryLruCoreBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            std::min(details.max_non_zeros, static_cast<Index_>(indices.size())), // Tighten the bounds to fit more elements into the cache.
            needs_value, 
            needs_index
        ),
        my_secondary_dim(details.secondary_dim)
    {
        populate_sparse_primary_remap_vector<sparse_>(indices, my_remap, my_first_index, my_past_last_index);

        // Ensure that we can resize my_index_buffer (and my_data_buffer) safely.
        tatami::can_cast_Index_to_container_size<decltype(my_index_buffer)>(details.max_non_zeros);
        if (this->my_needs_value) {
            tatami::can_cast_Index_to_container_size<decltype(my_data_buffer)>(details.max_non_zeros);
        }

        // Protect pointer differences against overflow when refining primary limits.
        sanisizer::can_ptrdiff<decltype(my_index_buffer.begin())>(details.max_non_zeros);
    }

private:
    Index_ my_secondary_dim;
    Index_ my_first_index, my_past_last_index;
    SparsePrimaryRemapVector<sparse_, Index_> my_remap;
    std::vector<CachedIndex_> my_index_buffer;
    std::vector<CachedValue_> my_data_buffer;
    std::vector<Index_> my_found;

public:
    const PrimarySlab<Index_, CachedValue_, CachedIndex_>& fetch_raw(Index_ i) {
        return this->my_cache.find(
            i, 
            /* create = */ [&]() -> PrimarySlab<Index_, CachedValue_, CachedIndex_> {
                return this->create();
            },
            /* populate = */ [&](Index_ i, PrimarySlab<Index_, CachedValue_, CachedIndex_>& current_cache) -> void {
                const auto narrowed = narrow_primary_extraction_range(
                    this->my_pointers[i],
                    this->my_pointers[i + 1],
                    my_first_index,
                    my_past_last_index,
                    my_secondary_dim
                );
                const hsize_t extraction_start = narrowed.first;
                const hsize_t extraction_len = narrowed.second - extraction_start;
                if (extraction_len == 0) {
                    current_cache.number = 0;
                    return;
                }
                my_index_buffer.resize(extraction_len); // implicit cast is safe as we checked it in the constructor.

                serialize([&]() -> void {
                    auto& comp = *(this->my_h5comp);
                    comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                    comp.memspace.setExtentSimple(1, &extraction_len);
                    comp.memspace.selectAll();
                    comp.index_dataset.read(my_index_buffer.data(), define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                    auto indices_start = my_index_buffer.begin();
                    auto indices_end = my_index_buffer.end();
                    refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_first_index, my_past_last_index);

                    const Index_ num_found = scan_for_indices_in_sparse_primary_remap_vector<sparse_>(
                        indices_start,
                        indices_end,
                        my_first_index,
                        current_cache.index,
                        my_found,
                        my_remap,
                        this->my_needs_value,
                        this->my_needs_index
                    );

                    if (this->my_needs_value && num_found > 0) {
                        const auto first_found = my_found.front();

                        // The idea is to extract the entire range spanned by the indices and then only store what we need in the cache.
                        // This avoids using hyperslab unions that are incredibly slow as of time of writing (HDF5 1.14.6 and earlier).
                        // Note that pointer differences are safe for the cast below, we already checked in the constructor.
                        const hsize_t range_start = extraction_start + static_cast<hsize_t>(indices_start - my_index_buffer.begin()) + first_found;
                        const hsize_t range_len = my_found.back() - first_found + 1;
                        my_data_buffer.resize(range_len); // implicit cast is safe as we checked it in the constructor.

                        comp.dataspace.selectHyperslab(H5S_SELECT_SET, &range_len, &range_start);
                        comp.memspace.setExtentSimple(1, &range_len);
                        comp.memspace.selectAll();
                        comp.data_dataset.read(my_data_buffer.data(), define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);

                        for (Index_ f = 0; f < num_found; ++f) {
                            current_cache.value[f] = my_data_buffer[my_found[f] - first_found];
                        }
                    }

                    current_cache.number = num_found;
                });
            }
        );
    }
};

/******************************************
 **** Oraclular base extractor classes ****
 ******************************************/

template<typename CachedValue_, typename CachedIndex_, typename Index_>
std::size_t choose_cache_size_for_primary_oracular(const MatrixDetails<Index_>& details, bool needs_cached_value, bool needs_cached_index) {
    std::size_t elsize = size_of_cached_element<CachedValue_, CachedIndex_>(needs_cached_value, needs_cached_index);
    if (elsize == 0) {
        // Just return the largest possible value. The exact value doesn't matter here as we won't actually be storing anything.
        // Importantly, we won't be resizing the vectors either as both needs_cached_value and needs_cached_index are false.
        return -1;
    }

    std::size_t proposed = details.slab_cache_size / elsize;
    if (sanisizer::is_greater_than(proposed, details.pointers.back())) {
        return details.pointers.back(); // no point in exceeding the maximum number of non-zero elements, as we've already cached everything. 
    }

    // Take the max() to make sure we can hold the largest dimension element.
    // The cast is safe here, as we know that dimension extents can fit into std::size_t.
    return std::max(static_cast<std::size_t>(details.max_non_zeros), proposed);
}

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularCoreBase {
public:
    PrimaryOracularCoreBase(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        bool needs_cached_value, 
        bool needs_cached_index) : 
        my_pointers(details.pointers),
        my_cache(std::move(oracle), choose_cache_size_for_primary_oracular<CachedValue_, CachedIndex_>(details, needs_cached_value, needs_cached_index))
    {
        initialize(details, my_h5comp);

        if (needs_cached_value) {
            my_full_value_buffer.resize(sanisizer::cast<decltype(my_full_value_buffer.size())>(my_cache.get_max_size()));
        }
        if (needs_cached_index) {
            my_full_index_buffer.resize(sanisizer::cast<decltype(my_full_index_buffer.size())>(my_cache.get_max_size()));
        }
    }

    ~PrimaryOracularCoreBase() {
        destroy(my_h5comp);
    }

protected:
    std::unique_ptr<Components> my_h5comp;
    const std::vector<hsize_t>& my_pointers;

    struct SlabPrecursor {
        std::size_t mem_offset;
        std::size_t length;
    };

protected:
    // Here, we use a giant contiguous buffer to optimize for near-consecutive
    // iteration. This allows the HDF5 library to pull out long strips of data from
    // the file.  It also allows us to maximize the use of the cache_size_limit by
    // accounting for differences in the non-zeros for each element, rather than
    // conservatively assuming they're all at max (as in the LRU case). The
    // downside is that we need to do some copying within the cache to make space
    // for new reads, but that works out to be no more than one extra copy per
    // fetch() call, which is tolerable. I suppose we could do better by
    // defragmenting within this buffer but that's probably overkill.
    std::vector<CachedValue_> my_full_value_buffer;
    std::vector<CachedIndex_> my_full_index_buffer;

private:
    tatami_chunked::OracularVariableSlabCache<Index_, Index_, SlabPrecursor, std::size_t> my_cache;

protected:
    template<class Function_>
    static void sort_by_field(std::vector<std::pair<Index_, std::size_t> >& indices, Function_ field) {
        auto comp = [&field](const std::pair<Index_, std::size_t>& l, const std::pair<Index_, std::size_t>& r) -> bool {
            return field(l) < field(r);
        };
        if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
            std::sort(indices.begin(), indices.end(), comp);
        }
    }

public:
    // Note that needs_cached_index in the constructor may not always equal
    // needs_index. When doing block/indexed extraction, we need the index to
    // determine the block/index (and thus it needs to be loaded into the cache
    // at some point, and thus we need to consider it in the cache size
    // calculations) even if we don't want to report those indices. In those
    // cases, needs_cached_index=true but we might have needs_index=false,
    // hence the separate arguments here versus the constructor. We just do the
    // same for needs_(cached_)value just for consistency.
    template<class GuessSize_, class Extract_>
    PrimarySlab<Index_, CachedValue_, CachedIndex_> next(GuessSize_ guess_size, Extract_ extract, bool needs_value, bool needs_index) {
        auto out = my_cache.next(
            /* identify = */ [](Index_ i) -> std::pair<Index_, Index_> { 
                return std::pair<Index_, Index_>(i, 0); 
            },
            /* estimated_size = */ std::move(guess_size),
            /* actual_size = */ [&](Index_, const SlabPrecursor& preslab) -> std::size_t {
                return preslab.length;
            },
            /* create = */ [&]() -> SlabPrecursor {
                return SlabPrecursor();
            },
            /* populate = */ [&](
                std::vector<std::pair<Index_, std::size_t> >& to_populate,
                std::vector<std::pair<Index_, std::size_t> >& to_reuse,
                std::vector<SlabPrecursor>& all_preslabs
            ) -> void {
                std::size_t dest_offset = 0;

                if (to_reuse.size()) {
                    // Shuffling all re-used elements to the start of the buffer,
                    // so that we can perform a contiguous extraction of the needed
                    // elements in the rest of the buffer. This needs some sorting
                    // to ensure that we're not clobbering one re-used element's
                    // contents when shifting another element to the start.
                    sort_by_field(to_reuse, [&](const std::pair<Index_, std::size_t>& p) -> std::size_t { return all_preslabs[p.second].mem_offset; });

                    for (const auto& p : to_reuse) {
                        auto& preslab = all_preslabs[p.second];
                        if (needs_index) {
                            auto isrc = my_full_index_buffer.begin() + preslab.mem_offset;
                            std::copy_n(isrc, preslab.length, my_full_index_buffer.begin() + dest_offset);
                        }
                        if (needs_value) {
                            auto vsrc = my_full_value_buffer.begin() + preslab.mem_offset;
                            std::copy_n(vsrc, preslab.length, my_full_value_buffer.begin() + dest_offset); 
                        }
                        preslab.mem_offset = dest_offset;
                        dest_offset += preslab.length;
                    }
                }

                extract(dest_offset, to_populate, all_preslabs, my_full_value_buffer, my_full_index_buffer);
            }
        );

        PrimarySlab<Index_, CachedValue_, CachedIndex_> output;
        output.number = out.first->length;
        if (needs_value) {
            output.value = my_full_value_buffer.data() + out.first->mem_offset;
        }
        if (needs_index) {
            output.index = my_full_index_buffer.data() + out.first->mem_offset;
        }
        return output;
    }
};

template<typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularFullCore : private PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracularFullCore(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        bool needs_value,
        bool needs_index
    ) : 
        PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>(details, std::move(oracle), needs_value, needs_index),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {}

private:
    bool my_needs_value, my_needs_index;

public:
    PrimarySlab<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::SlabPrecursor SlabPrecursor;
        return this->next(
            [&](Index_ i) -> std::size_t {
                // cast on return is safe as we know this value is <= max_non_zeros <= secondary_dim, which fits in a size_t.
                return this->my_pointers[i + 1] - this->my_pointers[i];
            },
            [&](
                const std::size_t dest_offset,
                std::vector<std::pair<Index_, std::size_t> >& to_populate, 
                std::vector<SlabPrecursor>& all_preslabs, 
                std::vector<CachedValue_>& full_value_buffer, 
                std::vector<CachedIndex_>& full_index_buffer
            ) -> void {
                // Sorting so that we get consecutive accesses in the hyperslab construction.
                // This should improve re-use of partially read HDF5 chunks inside DataSet::read().
                const auto& pointers = this->my_pointers;
                PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::sort_by_field(
                    to_populate,
                    [&](const std::pair<Index_, std::size_t>& p) -> hsize_t { return pointers[p.first]; }
                );

                auto num_needed = to_populate.size();
                decltype(num_needed) sofar = 0;
                std::size_t used_offset = dest_offset;

                // Finds contiguous slabs of indices to consolidate extraction into a single HDF5 call for efficiency.
                while (sofar < num_needed) {
                    const auto& pfirst = to_populate[sofar];
                    const hsize_t run_offset = pointers[pfirst.first];
                    hsize_t run_len = pointers[pfirst.first + 1] - run_offset;

                    auto& first_preslab = all_preslabs[pfirst.second];
                    first_preslab.mem_offset = used_offset;
                    first_preslab.length = run_len;
                    ++sofar;

                    Index_ previous = pfirst.first;
                    for (; sofar < num_needed; ++sofar) {
                        const auto& pnext = to_populate[sofar];
                        if (previous + 1 < pnext.first) {
                            break;
                        }

                        auto& next_preslab = all_preslabs[pnext.second];
                        next_preslab.mem_offset = used_offset + run_len;
                        hsize_t next_len = pointers[pnext.first + 1] - pointers[pnext.first];
                        next_preslab.length = next_len; // cast is safe, as we know that this is <= max_non_zeros <= secondary_dim, which fits in a size_t.
                        run_len += next_len;
                        previous = pnext.first;
                    }

                    serialize([&]() -> void {
                        auto& comp = *(this->my_h5comp);
                        comp.memspace.setExtentSimple(1, &run_len);
                        comp.memspace.selectAll();
                        comp.dataspace.selectHyperslab(H5S_SELECT_SET, &run_len, &run_offset);

                        if (my_needs_index) {
                            comp.index_dataset.read(full_index_buffer.data() + used_offset, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);
                        }
                        if (my_needs_value) {
                            comp.data_dataset.read(full_value_buffer.data() + used_offset, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                        }
                    });

                    used_offset += run_len;
                }
            },
            my_needs_value,
            my_needs_index
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularBlockCore : private PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracularBlockCore(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        Index_ block_start,
        Index_ block_length,
        bool needs_value,
        bool needs_index
    ) : 
        // Don't try to tighten the max_non_zeros like in the LRU case; we need
        // to keep enough space to ensure that every primary dimension element
        // can be extracted in its entirety (as we don't have a separate
        // index_buffer class to perform the initial HDF5 extraction).
        PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            needs_value, 
            true // We always need indices to compute the block boundaries.
        ), 
        my_secondary_dim(details.secondary_dim),
        my_block_start(block_start),
        my_block_past_end(block_start + block_length),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {
        // Protect pointer differences against overflow when refining primary limits.
        sanisizer::can_ptrdiff<decltype(this->my_full_index_buffer.begin())>(my_secondary_dim);
    }

private:
    Index_ my_secondary_dim;
    Index_ my_block_start, my_block_past_end;
    bool my_needs_value, my_needs_index;

public:
    PrimarySlab<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::SlabPrecursor SlabPrecursor;
        return this->next(
            [&](Index_ i) -> std::size_t {
                const auto narrowed = narrow_primary_extraction_range(
                    this->my_pointers[i],
                    this->my_pointers[i + 1],
                    my_block_start,
                    my_block_past_end,
                    my_secondary_dim
                );
                // cast on return is safe as we know this value is <= max_non_zeros <= secondary_dim, which fits in a size_t.
                return narrowed.second - narrowed.first;
            },
            [&](
                const std::size_t dest_offset,
                std::vector<std::pair<Index_, std::size_t> >& to_populate,
                std::vector<SlabPrecursor>& all_preslabs, 
                std::vector<CachedValue_>& full_value_buffer, 
                std::vector<CachedIndex_>& full_index_buffer
            ) -> void {
                serialize([&]() -> void {
                    std::size_t used_offset = dest_offset;
                    for (auto& pop : to_populate) {
                        auto& current_preslab = all_preslabs[pop.second];
                        current_preslab.mem_offset = used_offset;

                        const auto narrowed = narrow_primary_extraction_range(
                            this->my_pointers[pop.first],
                            this->my_pointers[pop.first + 1],
                            my_block_start,
                            my_block_past_end,
                            my_secondary_dim
                        );
                        const hsize_t extraction_start = narrowed.first;
                        const hsize_t extraction_len = narrowed.second - extraction_start;
                        if (extraction_len == 0) {
                            current_preslab.length = 0;
                            continue;
                        }

                        auto& comp = *(this->my_h5comp);
                        comp.memspace.setExtentSimple(1, &extraction_len);
                        comp.memspace.selectAll();
                        comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                        comp.index_dataset.read(full_index_buffer.data() + used_offset, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                        const auto original_start = full_index_buffer.begin() + used_offset;
                        auto indices_start = original_start;
                        auto indices_end = indices_start + extraction_len;
                        refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_block_start, my_block_past_end);

                        // We just update the pointers to the cache buffer instead of moving values around.
                        // Also, pointer subtraction is safe as we checked this in the constructor.
                        const auto refined_shift = indices_start - original_start;
                        current_preslab.mem_offset += refined_shift;
                        current_preslab.length = indices_end - indices_start;

                        // For dense extraction, we subtract block start so that the resulting indices can be used to directly index the output buffer.
                        if constexpr(!sparse_) {
                            if (my_needs_index) {
                                for (auto it = indices_start; it != indices_end; ++it) {
                                    *it -= my_block_start;
                                }
                            }
                        }

                        if (my_needs_value) {
                            const hsize_t refined_extraction_len = current_preslab.length;
                            const hsize_t refined_extraction_start = refined_shift + extraction_start;
                            comp.memspace.setExtentSimple(1, &refined_extraction_len);
                            comp.memspace.selectAll();
                            comp.dataspace.selectHyperslab(H5S_SELECT_SET, &refined_extraction_len, &refined_extraction_start);
                            comp.data_dataset.read(full_value_buffer.data() + current_preslab.mem_offset, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);
                        }

                        used_offset += extraction_len;
                    } 
                });
            },
            my_needs_value,
            my_needs_index
        );
    }
};

template<bool sparse_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryOracularIndexCore : private PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_> {
public:
    PrimaryOracularIndexCore(
        const MatrixDetails<Index_>& details,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        const std::vector<Index_>& indices,
        const bool needs_value,
        const bool needs_index
    ) : 
        // Don't try to tighten the max_non_zeros like in the LRU case; we need
        // to keep enough space to ensure that every primary dimension element
        // can be extracted in its entirety (as we don't have a separate
        // index_buffer object to perform the initial HDF5 extraction).
        PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>(
            details, 
            std::move(oracle), 
            needs_value, 
            true // We always need indices to figure out what to keep.
        ), 
        my_secondary_dim(details.secondary_dim),
        my_needs_value(needs_value),
        my_needs_index(needs_index)
    {
        populate_sparse_primary_remap_vector<sparse_>(indices, my_remap, my_first_index, my_past_last_index);

        // Protect pointer differences against overflow when refining primary limits.
        sanisizer::can_ptrdiff<decltype(this->my_full_index_buffer.begin())>(my_secondary_dim);
    }

private:
    Index_ my_secondary_dim;
    Index_ my_first_index, my_past_last_index;
    SparsePrimaryRemapVector<sparse_, Index_> my_remap;
    std::vector<Index_> my_found;
    bool my_needs_value, my_needs_index;

public:
    PrimarySlab<Index_, CachedValue_, CachedIndex_> fetch_raw(Index_ /* ignored, for consistency only.*/) {
        typedef typename PrimaryOracularCoreBase<Index_, CachedValue_, CachedIndex_>::SlabPrecursor SlabPrecursor;
        return this->next(
            [&](Index_ i) -> std::size_t {
                const auto narrowed = narrow_primary_extraction_range(
                    this->my_pointers[i],
                    this->my_pointers[i + 1],
                    my_first_index,
                    my_past_last_index,
                    my_secondary_dim
                );
                // cast on return is safe as we know this value is <= max_non_zeros <= secondary_dim, which fits in a size_t.
                return narrowed.second - narrowed.first;
            },
            [&](
                std::size_t dest_offset,
                std::vector<std::pair<Index_, std::size_t> >& to_populate,
                std::vector<SlabPrecursor>& all_preslabs, 
                std::vector<CachedValue_>& full_value_buffer, 
                std::vector<CachedIndex_>& full_index_buffer
            ) -> void {
                serialize([&]() -> void {
                    std::size_t used_offset = dest_offset;
                    for (auto& pop : to_populate) {
                        auto& current_preslab = all_preslabs[pop.second];
                        current_preslab.mem_offset = used_offset;

                        const auto narrowed = narrow_primary_extraction_range(
                            this->my_pointers[pop.first],
                            this->my_pointers[pop.first + 1],
                            my_first_index,
                            my_past_last_index,
                            my_secondary_dim
                        );
                        const hsize_t extraction_start = narrowed.first;
                        const hsize_t extraction_len = narrowed.second - extraction_start;
                        if (extraction_len == 0) {
                            current_preslab.length = 0;
                            continue;
                        }

                        auto& comp = *(this->my_h5comp);
                        comp.memspace.setExtentSimple(1, &extraction_len);
                        comp.memspace.selectAll();
                        comp.dataspace.selectHyperslab(H5S_SELECT_SET, &extraction_len, &extraction_start);
                        comp.index_dataset.read(full_index_buffer.data() + used_offset, define_mem_type<CachedIndex_>(), comp.memspace, comp.dataspace);

                        const auto original_start = full_index_buffer.begin() + used_offset;
                        auto indices_start = original_start;
                        auto indices_end = indices_start + extraction_len;
                        refine_primary_limits(indices_start, indices_end, my_secondary_dim, my_first_index, my_past_last_index);

                        const auto num_found = scan_for_indices_in_sparse_primary_remap_vector<sparse_>(
                            indices_start,
                            indices_end,
                            my_first_index,
                            full_index_buffer.begin() + current_preslab.mem_offset,
                            my_found,
                            my_remap,
                            my_needs_value,
                            my_needs_index
                        );
                        current_preslab.length = num_found;

                        if (my_needs_value) {
                            // Pointer difference is safe as we checked this in the constructor.
                            const hsize_t refined_shift = static_cast<std::size_t>(indices_start - original_start);
                            const hsize_t refined_extraction_start = extraction_start + refined_shift;
                            const hsize_t refined_extraction_len = (indices_end - indices_start); 

                            comp.memspace.setExtentSimple(1, &refined_extraction_len);
                            comp.memspace.selectAll();
                            comp.dataspace.selectHyperslab(H5S_SELECT_SET, &refined_extraction_len, &refined_extraction_start);
                            comp.data_dataset.read(full_value_buffer.data() + used_offset, define_mem_type<CachedValue_>(), comp.memspace, comp.dataspace);

                            for (Index_ f = 0; f < num_found; ++f) {
                                full_value_buffer[used_offset + f] = full_value_buffer[used_offset + my_found[f]];
                            }
                        }

                        used_offset += num_found;
                    }
                });
            },
            my_needs_value,
            my_needs_index
        );
    }
};

/********************************
 **** Full extractor classes ****
 ********************************/

template<bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryFullCore = typename std::conditional<
    oracle_,
    PrimaryOracularFullCore<Index_, CachedValue_, CachedIndex_>,
    PrimaryLruFullCore<Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullSparse final : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryFullSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), needs_value, needs_index) {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_sparse(slab, value_buffer, index_buffer);
    }

private:
    ConditionalPrimaryFullCore<oracle_, Index_, CachedValue_, CachedIndex_> my_core;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryFullDense final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryFullDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle) : 
        my_core(details, std::move(oracle), true, true),
        my_secondary_dim(details.secondary_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_dense(slab, buffer, my_secondary_dim);
    }

private:
    ConditionalPrimaryFullCore<oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_secondary_dim;
};

/*********************************
 **** Block extractor classes ****
 *********************************/

template<bool sparse_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryBlockCore = typename std::conditional<
    oracle_,
    PrimaryOracularBlockCore<sparse_, Index_, CachedValue_, CachedIndex_>,
    PrimaryLruBlockCore<sparse_, Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryBlockSparse final : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryBlockSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), block_start, block_length, needs_value, needs_index) {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_sparse(slab, value_buffer, index_buffer);
    }

private:
    ConditionalPrimaryBlockCore<true, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryBlockDense final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryBlockDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length) :
        my_core(details, std::move(oracle), block_start, block_length, true, true), my_extract_length(block_length) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_dense(slab, buffer, my_extract_length);
    }

private:
    ConditionalPrimaryBlockCore<false, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_extract_length;
};

/*********************************
 **** Index extractor classes ****
 *********************************/

template<bool sparse_, bool oracle_, typename Index_, typename CachedValue_, typename CachedIndex_>
using ConditionalPrimaryIndexCore = typename std::conditional<
    oracle_,
    PrimaryOracularIndexCore<sparse_, Index_, CachedValue_, CachedIndex_>,
    PrimaryLruIndexCore<sparse_, Index_, CachedValue_, CachedIndex_>
>::type;

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryIndexSparse final : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryIndexSparse(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, bool needs_value, bool needs_index) : 
        my_core(details, std::move(oracle), *indices_ptr, needs_value, needs_index) {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_sparse(slab, value_buffer, index_buffer);
    }

private:
    ConditionalPrimaryIndexCore<true, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
};

template<bool oracle_, typename Value_, typename Index_, typename CachedValue_, typename CachedIndex_>
class PrimaryIndexDense final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    PrimaryIndexDense(const MatrixDetails<Index_>& details, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr) :
        my_core(details, std::move(oracle), *indices_ptr, true, true), my_extract_length(indices_ptr->size()) {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto slab = my_core.fetch_raw(i);
        return slab_to_dense(slab, buffer, my_extract_length);
    }

private:
    ConditionalPrimaryIndexCore<false, oracle_, Index_, CachedValue_, CachedIndex_> my_core;
    Index_ my_extract_length;
};

}

}

#endif
