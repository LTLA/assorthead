#ifndef TATAMI_HDF5_SPARSE_UTILS_HPP
#define TATAMI_HDF5_SPARSE_UTILS_HPP

#include "serialize.hpp"

#include <algorithm>
#include <utility>
#include <cstddef>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"

namespace tatami_hdf5 {

namespace CompressedSparseMatrix_internal {

template<typename CachedValue_, typename CachedIndex_>
std::size_t size_of_cached_element(bool needs_cached_value, bool needs_cached_index) {
    return (needs_cached_index ? sizeof(CachedIndex_) : 0) + (needs_cached_value ? sizeof(CachedValue_) : 0);
}

struct Components {
    H5::H5File file;
    H5::DataSet data_dataset;
    H5::DataSet index_dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

struct ChunkCacheSizes {
    ChunkCacheSizes() = default;
    ChunkCacheSizes(hsize_t value, hsize_t index) : value(value), index(index) {}
    hsize_t value = 0;
    hsize_t index = 0;
};

// Consider the case where we're iterating across primary dimension elements and extracting its contents from file.
// At each iteration, we will have a partially-read chunk that spans the ending values of the latest primary dimension element.
// (Unless we're going backwards, in which case the partially-read chunk would span the starting values.)
// We want to cache this chunk so that its contents can be fully read upon scanning the next primary dimension element.
//
// In practice, we want the cache to be large enough to hold three chunks simultaneously;
// one for the partially read chunk at the start of a primary dimension element, one for the partially read chunk at the end,
// and another just in case HDF5 needs to cache the middle chunks before copying them to the user buffer.
// This arrangement ensures that we can hold both partially read chunks while storing and evicting the fully read chunks,
// no matter what order the HDF5 library uses to read chunks to satisfy our request.
//
// In any case, we also ensure that the HDF5 chunk cache is at least 1 MB (the default).
// This allows us to be at least as good as the default in cases where reads are non-contiguous and we've got partially read chunks everywhere.
inline hsize_t compute_chunk_cache_size(hsize_t nonzeros, hsize_t chunk_length, std::size_t element_size) {
    if (chunk_length == 0) {
        return 0;
    }
    hsize_t num_chunks = std::min(nonzeros / chunk_length + (nonzeros % chunk_length > 0), static_cast<hsize_t>(3));
    hsize_t cache_size = sanisizer::product<hsize_t>(num_chunks, chunk_length);
    return std::max(sanisizer::product<hsize_t>(cache_size, element_size), sanisizer::cap<hsize_t>(1000000));
}

// In all cases, we know that max_non_zeros can be safely casted between hsize_t and Index_,
// because the value is derived from differences between hsize_t pointers.
// We also know that it can be safely cast to std::size_t, as max_non_zeros is no greater than the dimension extents,
// and we know that the dimension extents must be representable as a std::size_t as per the tatami contract.

template<typename Index_>
struct MatrixDetails {
    MatrixDetails(
        const std::string& file_name, 
        const std::string& value_name, 
        const std::string& index_name, 
        Index_ primary_dim, 
        Index_ secondary_dim, 
        const std::vector<hsize_t>& pointers, 
        std::size_t slab_cache_size,
        Index_ max_non_zeros,
        ChunkCacheSizes chunk_cache_sizes
    ) :
        file_name(file_name), 
        value_name(value_name), 
        index_name(index_name), 
        primary_dim(primary_dim), 
        secondary_dim(secondary_dim), 
        pointers(pointers), 
        slab_cache_size(slab_cache_size),
        max_non_zeros(max_non_zeros),
        chunk_cache_sizes(std::move(chunk_cache_sizes))
    {}

    const std::string& file_name;
    const std::string& value_name;
    const std::string& index_name;

    Index_ primary_dim;
    Index_ secondary_dim;
    const std::vector<hsize_t>& pointers;

    std::size_t slab_cache_size; // size for our own cache of slabs.
    Index_ max_non_zeros;
    ChunkCacheSizes chunk_cache_sizes; // size for HDF5's cache of uncompressed chunks.
};

// All HDF5-related members are stored in a separate pointer so we can serialize construction and destruction.
template<typename Index_>
void initialize(const MatrixDetails<Index_>& details, std::unique_ptr<Components>& h5comp) {
    serialize([&]() -> void {
        h5comp.reset(new Components);

        auto create_dapl = [&](hsize_t cache_size) -> H5::DSetAccPropList {
            // passing an ID is the only way to get the constructor to make a copy, not a reference (who knows why???).
            H5::DSetAccPropList dapl(H5::DSetAccPropList::DEFAULT.getId());
            dapl.setChunkCache(H5D_CHUNK_CACHE_NSLOTS_DEFAULT, cache_size, H5D_CHUNK_CACHE_W0_DEFAULT);
            return dapl;
        };

        h5comp->file.openFile(details.file_name, H5F_ACC_RDONLY);
        h5comp->data_dataset = h5comp->file.openDataSet(details.value_name, create_dapl(details.chunk_cache_sizes.value));
        h5comp->index_dataset = h5comp->file.openDataSet(details.index_name, create_dapl(details.chunk_cache_sizes.index));
        h5comp->dataspace = h5comp->data_dataset.getSpace();
    });
}

inline void destroy(std::unique_ptr<Components>& h5comp) {
    serialize([&]() -> void {
        h5comp.reset();
    });
}

// If we're only interested in the block [X, Y), we don't need to extract the entire set of indices for a primary element.
// We only need to extract the first Y elements, as all subsequent elements must have indices greater than or equal to Y.
// Similarly, we only need to extract the last 'dim - X' elements, as all preceding elements must have indices less than X.
// This allows us to narrow the range of indices to be extracted.
template<typename Pointer_, typename Index_>
std::pair<Pointer_, Pointer_> narrow_primary_extraction_range(
    const Pointer_ ptr_start,
    const Pointer_ ptr_end,
    const Index_ block_start,
    const Index_ block_end,
    const Index_ secondary_dim
) {
    const Pointer_ num_nonzeros = ptr_end - ptr_start;
    const Pointer_ new_ptr_end = ptr_start + sanisizer::min(block_end, num_nonzeros);
    const Pointer_ new_ptr_start = ptr_end - sanisizer::min(secondary_dim - block_start, num_nonzeros);

    // It is guaranteed that new_ptr_start <= new_ptr_end.
    // This is because the difference boils down to:
    //
    // -num_nonzeros + num_nonzeros + num_nonzeros, OR
    // -num_nonzeros + block_end + num_nonzeros, OR
    // -num_nonzeros + num_nonzeros + secondary_dim - block_start, OR
    // -num_nonzeros + block_end + secondary_dim - block_start
    // 
    // All of which are guaranteed to be non-negative.
    return std::make_pair(new_ptr_start, new_ptr_end);
}

template<class IndexIt_, typename Index_>
void refine_primary_limits(IndexIt_& indices_start, IndexIt_& indices_end, Index_ extent, Index_ smallest, Index_ largest_plus_one) {
    if (smallest) {
        // Using custom comparator to ensure that we cast to Index_ for signedness-safe comparisons.
        indices_start = std::lower_bound(indices_start, indices_end, smallest, [](Index_ a, Index_ b) -> bool { return a < b; });
    }

    if (largest_plus_one != extent) {
        indices_end = std::lower_bound(indices_start, indices_end, largest_plus_one, [](Index_ a, Index_ b) -> bool { return a < b; });
    }
}

template<typename Slab_, typename Value_, typename Index_>
tatami::SparseRange<Value_, Index_> slab_to_sparse(const Slab_& slab, Value_* value_buffer, Index_* index_buffer) {
    tatami::SparseRange<Value_, Index_> output(slab.number);
    if (slab.value) {
        std::copy_n(slab.value, slab.number, value_buffer);
        output.value = value_buffer;
    }
    if (slab.index) {
        std::copy_n(slab.index, slab.number, index_buffer);
        output.index = index_buffer;
    }
    return output;
}

template<typename Slab_, typename Value_, typename Index_>
Value_* slab_to_dense(const Slab_& slab, Value_* buffer, Index_ extract_length) {
    std::fill_n(buffer, extract_length, 0);
    for (Index_ i = 0; i < slab.number; ++i) {
        buffer[slab.index[i]] = slab.value[i];
    }
    return buffer;
}

}

}

#endif
