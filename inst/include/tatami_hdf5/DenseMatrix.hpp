#ifndef TATAMI_HDF5_DENSE_MATRIX_HPP
#define TATAMI_HDF5_DENSE_MATRIX_HPP

#include "serialize.hpp"
#include "utils.hpp"

#include <string>
#include <type_traits>
#include <cmath>
#include <vector>

#include "H5Cpp.h"
#include "tatami_chunked/tatami_chunked.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file DenseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed dense matrix.
 */

namespace tatami_hdf5 {

/**
 * @brief Options for `DenseMatrix` extraction.
 */
struct DenseMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all chunks required to read a row/column in `tatami::MyopicDenseExtractor::fetch()` and related methods.
     * This allows us to re-use the cached chunks when adjacent rows/columns are requested, rather than re-reading them from disk.
     *
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored if `require_minimum_cache` is `true`.
     */
    std::size_t maximum_cache_size = sanisizer::cap<std::size_t>(100000000);

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that all chunks overlapping one row (or a slice/subset thereof) can be retained in memory,
     * so that the same chunks are not repeatedly re-read from disk when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

/**
 * @cond
 */
namespace DenseMatrix_internal {

// All HDF5-related members.
struct Components {
    H5::H5File file;
    H5::DataSet dataset;
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
};

// In all cases, we know that the dimension extents can be safely casted between hsize_t and Index_,
// because we checked for a safe cast (via the ChunkDimensionStats constructor) in the DenseMatrix constructor. 
// This is in addition to knowing that the extents can be safely casted to size_t as per the tatami contract.

template<typename Index_, typename OutputValue_>
void extract_block(bool h5_row_is_target, Index_ cache_start, Index_ cache_length, Index_ block_start, Index_ block_length, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    int target_dim = 1 - h5_row_is_target;
    offset[target_dim] = cache_start;
    count[target_dim] = cache_length;

    int non_target_dim = h5_row_is_target;
    offset[non_target_dim] = block_start;
    count[non_target_dim] = block_length;
    comp.dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // HDF5 is a lot faster when the memspace and dataspace match in dimensionality.
    // Presumably there is some shuffling that happens inside when dimensions don't match.
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

template<typename Index_, typename OutputValue_>
void extract_indices(bool h5_row_is_target, Index_ cache_start, Index_ cache_length, const std::vector<Index_>& indices, OutputValue_* buffer, Components& comp) {
    hsize_t offset[2];
    hsize_t count[2];

    int target_dim = 1 - h5_row_is_target;
    offset[target_dim] = cache_start;
    count[target_dim] = cache_length;

    int non_target_dim = h5_row_is_target;

    // Take slices across the current chunk for each index. This should be okay if consecutive,
    // but hopefully they've fixed the problem with non-consecutive slices in:
    // https://forum.hdfgroup.org/t/union-of-non-consecutive-hyperslabs-is-very-slow/5062
    comp.dataspace.selectNone();
    tatami::process_consecutive_indices<Index_>(
        indices.data(),
        indices.size(),
        [&](Index_ start, Index_ length) -> void {
            offset[non_target_dim] = start;
            count[non_target_dim] = length;
            comp.dataspace.selectHyperslab(H5S_SELECT_OR, count, offset);
        }
    );

    // Again, matching the dimensionality.
    count[non_target_dim] = indices.size();
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();

    comp.dataset.read(buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);
}

/********************
 *** Core classes ***
 ********************/

// We store the Components in a pointer so that we can serialize their
// construction and destruction.

inline void initialize(const std::string& file_name, const std::string& dataset_name, std::unique_ptr<Components>& h5comp) {
    serialize([&]() -> void {
        h5comp.reset(new Components);

        // Turn off HDF5's caching, as we'll be handling that.
        // This allows us to parallelize extractions without locking when the data has already been loaded into memory.
        // If we just used HDF5's cache, we would have to lock on every extraction, given the lack of thread safety.
        H5::FileAccPropList fapl(H5::FileAccPropList::DEFAULT.getId());
        fapl.setCache(0, 0, 0, 0);

        h5comp->file.openFile(file_name, H5F_ACC_RDONLY, fapl);
        h5comp->dataset = h5comp->file.openDataSet(dataset_name);
        h5comp->dataspace = h5comp->dataset.getSpace();
    });
}

inline void destroy(std::unique_ptr<Components>& h5comp) {
    serialize([&]() -> void {
        h5comp.reset();
    });
}

template<bool oracle_, bool by_h5_row_, typename Index_>
class SoloCore {
public:
    SoloCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        [[maybe_unused]] tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, // only listed here for compatibility with the other constructors.
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        [[maybe_unused]] const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_oracle(std::move(oracle))
    {
        initialize(file_name, dataset_name, my_h5comp);
    }

    ~SoloCore() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, tatami::PredictionIndex, bool>::type my_counter = 0;

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&]() -> void {
            extract_block(by_h5_row_, i, static_cast<Index_>(1), block_start, block_length, buffer, *my_h5comp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&]() -> void {
            extract_indices(by_h5_row_, i, static_cast<Index_>(1), indices, buffer, *my_h5comp);
        });
        return buffer;
    }
};

template<bool by_h5_row_, typename Index_, typename CachedValue_>
class MyopicCore {
public:
    MyopicCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_>, // for consistency with the oracular version.
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_dim_stats(std::move(target_dim_stats)),
        my_factory(slab_stats), 
        my_cache(slab_stats.max_slabs_in_cache)
    {
        initialize(file_name, dataset_name, my_h5comp);
        if constexpr(!by_h5_row_) {
            my_transposition_buffer.resize(sanisizer::cast<decltype(my_transposition_buffer.size())>(slab_stats.slab_size_in_elements));
        }
    }

    ~MyopicCore() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;

    tatami_chunked::DenseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

    typename std::conditional<by_h5_row_, bool, std::vector<CachedValue_> >::type my_transposition_buffer;

private:
    template<typename Value_, class Extract_>
    void fetch_raw(Index_ i, Value_* buffer, Index_ non_target_length, Extract_ extract) {
        Index_ chunk = i / my_dim_stats.chunk_length;
        Index_ index = i % my_dim_stats.chunk_length;

        const auto& info = my_cache.find(
            chunk, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](Index_ id, Slab& contents) -> void {
                auto curdim = tatami_chunked::get_chunk_length(my_dim_stats, id);

                if constexpr(by_h5_row_) {
                    serialize([&]() -> void {
                        extract(id * my_dim_stats.chunk_length, curdim, contents.data);
                    });

                } else {
                    // Transposing the data for easier retrieval, but only once the lock is released.
                    auto tptr = my_transposition_buffer.data();
                    serialize([&]() -> void {
                        extract(id * my_dim_stats.chunk_length, curdim, tptr);
                    });
                    tatami::transpose(tptr, non_target_length, curdim, contents.data);
                }
            }
        );

        auto ptr = info.data + sanisizer::product_unsafe<std::size_t>(non_target_length, index);
        std::copy_n(ptr, non_target_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(
            i, 
            buffer, 
            block_length, 
            [&](Index_ start, Index_ length, CachedValue_* buf) -> void {
                extract_block(by_h5_row_, start, length, block_start, block_length, buf, *my_h5comp);
            }
        );
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(
            i,
            buffer,
            indices.size(), 
            [&](Index_ start, Index_ length, CachedValue_* buf) -> void {
                extract_indices(by_h5_row_, start, length, indices, buf, *my_h5comp);
            }
        );
        return buffer;
    }
};

// This class performs oracular dense extraction when each target dimension element is a row in the HDF5 matrix.
// No transposition is required and we can achieve some optimizations with the HDF5 library call.
template<typename Index_, typename CachedValue_>
class OracularCoreNormal {
public:
    OracularCoreNormal(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_dim_stats(std::move(target_dim_stats)),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache),
        my_slab_size(slab_stats.slab_size_in_elements),
        my_memory_pool(sanisizer::product<decltype(my_memory_pool.size())>(slab_stats.max_slabs_in_cache, my_slab_size))
    {
        initialize(file_name, dataset_name, my_h5comp);
    }

    ~OracularCoreNormal() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;

    struct Slab {
        std::size_t offset;
    };

    tatami_chunked::OracularSlabCache<Index_, Index_, Slab, true> my_cache;
    std::size_t my_slab_size;
    std::vector<CachedValue_> my_memory_pool;
    std::size_t my_offset = 0;

private:
    template<class Function_>
    static void sort_by_field(std::vector<std::pair<Index_, Slab*> >& indices, Function_ field) {
        auto comp = [&field](const std::pair<Index_, Slab*>& l, const std::pair<Index_, Slab*>& r) -> bool {
            return field(l) < field(r);
        };
        if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
            std::sort(indices.begin(), indices.end(), comp);
        }
    }

    template<typename Value_, class Unionize_>
    void fetch_raw([[maybe_unused]] Index_ i, Value_* buffer, Index_ non_target_length, Unionize_ unionize) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_dim_stats.chunk_length, current % my_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                Slab output;
                output.offset = my_offset;
                my_offset += my_slab_size;
                return output;
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& chunks, std::vector<std::pair<Index_, Slab*> >& to_reuse) -> void {
                // Defragmenting the existing chunks. We sort by offset to make 
                // sure that we're not clobbering in-use slabs during the copy().
                sort_by_field(to_reuse, [](const std::pair<Index_, Slab*>& x) -> std::size_t { return x.second->offset; });

                auto dest = my_memory_pool.data();
                std::size_t running_offset = 0;
                for (auto& x : to_reuse) {
                    auto& cur_offset = x.second->offset;
                    if (cur_offset != running_offset) {
                        std::copy_n(dest + cur_offset, my_slab_size, dest + running_offset);
                        cur_offset = running_offset;
                    }
                    running_offset += my_slab_size;
                }

                // Collapsing runs of consecutive hyperslabs into a single hyperslab;
                // otherwise, taking hyperslab unions. This allows a single HDF5 call
                // to populate the contiguous memory pool that we made available after
                // defragmentation; then we just update the slab pointers to refer
                // to the slices of memory corresponding to each slab.
                sort_by_field(chunks, [](const std::pair<Index_, Slab*>& x) -> Index_ { return x.first; });

                serialize([&]() -> void {
                    auto& components = *my_h5comp;
                    auto& dspace = my_h5comp->dataspace;
                    dspace.selectNone();

                    // Remember, the slab size is equal to the product of the chunk length and the 
                    // non-target length, so shifting the memory pool offsets by 'slab_size' will 
                    // correspond to a shift of 'chunk_length' on the target dimension. The only
                    // exception is that of the last chunk, but at that point it doesn't matter as 
                    // there's no data following the last chunk.
                    Index_ run_chunk_id = chunks.front().first;
                    Index_ chunk_length = tatami_chunked::get_chunk_length(my_dim_stats, run_chunk_id);
                    Index_ run_length = chunk_length;
                    Index_ total_length = chunk_length;
                    chunks.front().second->offset = running_offset;
                    auto start_offset = running_offset;
                    running_offset += my_slab_size;

                    for (decltype(chunks.size()) ci = 1, cend = chunks.size(); ci < cend; ++ci) {
                        auto& current_chunk = chunks[ci];
                        Index_ current_chunk_id = current_chunk.first;

                        if (current_chunk_id - run_chunk_id > 1) { // save the existing run of chunks as one hyperslab, and start a new run.
                            unionize(dspace, run_chunk_id * my_dim_stats.chunk_length, run_length);
                            run_chunk_id = current_chunk_id;
                            run_length = 0;
                        }

                        Index_ current_length = tatami_chunked::get_chunk_length(my_dim_stats, current_chunk_id);
                        run_length += current_length;
                        total_length += current_length;
                        current_chunk.second->offset = running_offset;
                        running_offset += my_slab_size;
                    }

                    unionize(dspace, run_chunk_id * my_dim_stats.chunk_length, run_length);

                    hsize_t count[2];
                    count[0] = total_length;
                    count[1] = non_target_length;
                    components.memspace.setExtentSimple(2, count);
                    components.memspace.selectAll();
                    components.dataset.read(dest + start_offset, define_mem_type<CachedValue_>(), components.memspace, dspace);
                });
            }
        );

        auto ptr = my_memory_pool.data() + info.first->offset + sanisizer::product_unsafe<std::size_t>(non_target_length, info.second);
        std::copy_n(ptr, non_target_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(
            i,
            buffer,
            block_length,
            [&](H5::DataSpace& dspace, Index_ run_start, Index_ run_length) -> void {
                hsize_t offset[2];
                hsize_t count[2];
                offset[0] = run_start;
                offset[1] = block_start;
                count[0] = run_length;
                count[1] = block_length;
                dspace.selectHyperslab(H5S_SELECT_OR, count, offset);
            }
        );
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(
            i,
            buffer,
            indices.size(),
            [&](H5::DataSpace& dspace, Index_ run_start, Index_ run_length) -> void {
                hsize_t offset[2];
                hsize_t count[2];
                offset[0] = run_start;
                count[0] = run_length;

                // See comments in extract_indices().
                tatami::process_consecutive_indices<Index_>(
                    indices.data(),
                    indices.size(),
                    [&](Index_ start, Index_ length) -> void {
                        offset[1] = start;
                        count[1] = length;
                        dspace.selectHyperslab(H5S_SELECT_OR, count, offset);
                    }
                );
            }
        );
        return buffer;
    }
};

// This class performs oracular dense extraction when each target dimension element is NOT a row in the HDF5 matrix.
// This requires an additional transposition step for each slab after its extraction from the HDF5 library.
template<typename Index_, typename CachedValue_>
class OracularCoreTransposed {
public:
    OracularCoreTransposed(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_dim_stats(std::move(target_dim_stats)),
        my_factory(slab_stats), 
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache),
        my_transposition_buffer(sanisizer::cast<decltype(my_transposition_buffer.size())>(slab_stats.slab_size_in_elements)),
        my_transposition_buffer_ptr(my_transposition_buffer.data())
    {
        initialize(file_name, dataset_name, my_h5comp);
        my_cache_transpose_info.reserve(slab_stats.max_slabs_in_cache);
    }

    ~OracularCoreTransposed() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;

    tatami_chunked::DenseSlabFactory<CachedValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;
    tatami_chunked::OracularSlabCache<Index_, Index_, Slab> my_cache;

    std::vector<CachedValue_> my_transposition_buffer;
    CachedValue_* my_transposition_buffer_ptr;
    std::vector<std::pair<Slab*, Index_> > my_cache_transpose_info;

private:
    template<typename Value_, class Extract_>
    void fetch_raw([[maybe_unused]] Index_ i, Value_* buffer, Index_ non_target_length, Extract_ extract) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_dim_stats.chunk_length, current % my_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                return my_factory.create();
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                my_cache_transpose_info.clear();

                serialize([&]() -> void {
                    for (const auto& c : chunks) {
                        auto curdim = tatami_chunked::get_chunk_length(my_dim_stats, c.first);
                        extract(c.first * my_dim_stats.chunk_length, curdim, c.second->data);
                        my_cache_transpose_info.emplace_back(c.second, curdim);
                    }
                });

                // Applying transpositions to each cached buffers for easier
                // retrieval. Done outside the serial section to unblock other threads.
                if (non_target_length != 1) {
                    for (const auto& c : my_cache_transpose_info) {
                        if (c.second != 1) {
                            tatami::transpose(c.first->data, non_target_length, c.second, my_transposition_buffer_ptr);

                            // We actually swap the pointers here, so the slab
                            // pointers might not point to the factory pool after this!
                            // Shouldn't matter as long as neither of them leave this class.
                            std::swap(c.first->data, my_transposition_buffer_ptr);
                        }
                    }
                }
            }
        );

        auto ptr = info.first->data + sanisizer::product_unsafe<std::size_t>(non_target_length, info.second);
        std::copy_n(ptr, non_target_length, buffer);
    }

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        fetch_raw(
            i,
            buffer,
            block_length,
            [&](Index_ start, Index_ length, CachedValue_* buf) -> void {
                extract_block(false, start, length, block_start, block_length, buf, *my_h5comp);
            }
        );
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        fetch_raw(
            i,
            buffer,
            indices.size(),
            [&](Index_ start, Index_ length, CachedValue_* buf) -> void {
                extract_indices(false, start, length, indices, buf, *my_h5comp);
            }
        );
        return buffer;
    }
};

// COMMENT: technically, for all oracular extractors, we could pretend that the chunk length on the target dimension is 1.
// This would allow us to only extract the desired indices on each call, allowing for more efficient use of the cache.
// (In the transposed case, we would also reduce the amount of transposition that we need to perform.)
// The problem is that we would get harshly penalized for any chunk reuse outside of the current prediction cycle,
// where the chunk would need to be read from disk again if the exact elements weren't already in the cache.
// This access pattern might not be uncommon after applying a DelayedSubset with shuffled rows/columns. 

template<bool solo_, bool oracle_, bool by_h5_row_, typename Index_, typename CachedValue_>
using DenseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, by_h5_row_, Index_>,
      typename std::conditional<!oracle_,
          MyopicCore<by_h5_row_, Index_, CachedValue_>,
          typename std::conditional<by_h5_row_,
              OracularCoreNormal<Index_, CachedValue_>,
              OracularCoreTransposed<Index_, CachedValue_>
          >::type
      >::type
>::type;

/***************************
 *** Concrete subclasses ***
 ***************************/

template<bool solo_, bool oracle_, bool by_h5_row_, typename Value_, typename Index_, typename CachedValue_>
class Full final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Full(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_core(
            file_name, 
            dataset_name, 
            std::move(target_dim_stats),
            std::move(oracle),
            slab_stats
        ),
        my_non_target_dim(non_target_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, 0, my_non_target_dim, buffer);
    }

private:
    DenseCore<solo_, oracle_, by_h5_row_, Index_, CachedValue_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, bool by_h5_row_, typename Value_, typename Index_, typename CachedValue_> 
class Block final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Block(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_core( 
            file_name, 
            dataset_name, 
            std::move(target_dim_stats),
            std::move(oracle),
            slab_stats
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, my_block_start, my_block_length, buffer);
    }

private:
    DenseCore<solo_, oracle_, by_h5_row_, Index_, CachedValue_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, bool by_h5_row_, typename Value_, typename Index_, typename CachedValue_>
class Index final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Index(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats) :
        my_core(
            file_name,
            dataset_name, 
            std::move(target_dim_stats),
            std::move(oracle),
            slab_stats
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_indices(i, *my_indices_ptr, buffer);
    }

private:
    DenseCore<solo_, oracle_, by_h5_row_, Index_, CachedValue_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
};

}
/**
 * @endcond
 */

/**
 * @brief Dense matrix backed by a DataSet in a HDF5 file.
 *
 * This class retrieves data from the HDF5 file on demand rather than loading it all in at the start.
 * This allows us to handle very large datasets in limited memory at the cost of some speed.
 *
 * We manually handle the chunk caching to speed up access for consecutive rows and columns.
 * The policy is to minimize the number of calls to the HDF5 library by requesting large contiguous slices where possible,
 * where each slice typically consists of multiple rows/columns that belong in the same HDF5 chunk.
 * The size of the slice is determined by the `options` in the constructor.
 *
 * Callers should follow the `prefer_rows()` suggestion when extracting data,
 * as this tries to minimize the number of chunks that need to be read per access request.
 * If they do not, the access pattern on disk may be slightly to highly suboptimal, depending on the chunk dimensions.
 *
 * As the HDF5 library is not generally thread-safe, the HDF5-related operations should only be run in a single thread.
 * This is normally handled automatically but developers can check out `serialize()` to customize the locking scheme.
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store the values (based on their HDF5 type or other knowledge).
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_>
class DenseMatrix final : public tatami::Matrix<Value_, Index_> {
    std::string my_file_name, my_dataset_name;
    bool my_transpose;

    std::size_t my_cache_size_in_elements;
    bool my_require_minimum_cache;

    tatami_chunked::ChunkDimensionStats<Index_> my_firstdim_stats, my_seconddim_stats;
    bool my_prefer_firstdim;

public:
    /**
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     * @param transpose Whether the dataset is transposed in its storage order, i.e., rows in the HDF5 dataset correspond to columns of this matrix.
     * This may be true for HDF5 files generated by frameworks that use column-major matrices,
     * where preserving the data layout between memory and disk is more efficient (see, e.g., the **rhdf5** Bioconductor package).
     * @param options Further options for data extraction.
     */
    DenseMatrix(std::string file, std::string name, bool transpose, const DenseMatrixOptions& options) :
        my_file_name(std::move(file)), 
        my_dataset_name(std::move(name)),
        my_transpose(transpose),
        my_cache_size_in_elements(options.maximum_cache_size / sizeof(CachedValue_)),
        my_require_minimum_cache(options.require_minimum_cache)
    {
        serialize([&]() -> void {
            H5::H5File fhandle(my_file_name, H5F_ACC_RDONLY);
            auto dhandle = open_and_check_dataset<false>(fhandle, my_dataset_name);
            auto dims = get_array_dimensions<2>(dhandle, my_dataset_name);

            hsize_t chunk_dims[2];
            auto dparms = dhandle.getCreatePlist();
            if (dparms.getLayout() != H5D_CHUNKED) {
                // If contiguous, each firstdim is treated as a chunk.
                chunk_dims[0] = 1;
                chunk_dims[1] = dims[1];
            } else {
                dparms.getChunk(2, chunk_dims);
            }

            my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(dims[0], chunk_dims[0]);
            my_seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(dims[1], chunk_dims[1]);
        });

        // Favoring extraction on the dimension that involves pulling out fewer
        // chunks per dimension element. Remember, 'firstdim_stats.num_chunks'
        // represents the number of chunks along the first dimension, and thus
        // is the number of chunks that need to be loaded to pull out an
        // element of the **second** dimension; and vice versa.
        my_prefer_firstdim = (my_firstdim_stats.num_chunks > my_seconddim_stats.num_chunks);
    }

    /**
     * Overload that uses the default `DenseMatrixOptions`.
     * @param file Path to the file.
     * @param name Path to the dataset inside the file.
     * @param transpose Whether the dataset is transposed in its storage order.
     */
    DenseMatrix(std::string file, std::string name, bool transpose) : 
        DenseMatrix(std::move(file), std::move(name), transpose, DenseMatrixOptions()) {}

private:
    bool prefer_rows_internal() const {
        if (my_transpose) {
            return !my_prefer_firstdim;
        } else {
            return my_prefer_firstdim;
        }
    }

    Index_ nrow_internal() const {
        if (my_transpose) {
            return my_seconddim_stats.dimension_extent;
        } else {
            return my_firstdim_stats.dimension_extent;
        }
    }

    Index_ ncol_internal() const {
        if (my_transpose) {
            return my_firstdim_stats.dimension_extent;
        } else {
            return my_seconddim_stats.dimension_extent;
        }
    }

public:
    Index_ nrow() const {
        return nrow_internal();
    }

    Index_ ncol() const {
        return ncol_internal();
    }

    /**
     * @return Boolean indicating whether to prefer row extraction.
     *
     * We favor extraction on the first dimension (rows by default, columns when `transpose = true`) as this matches the HDF5 storage order.
     * However, for some chunking scheme and `cache_limit`, this might require repeated reads from file;
     * in such cases, we switch to extraction on the second dimension.
     */
    bool prefer_rows() const {
        return prefer_rows_internal();
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(prefer_rows_internal());
    }

    bool uses_oracle(bool) const {
        return true;
    }

    bool is_sparse() const {
        return false;
    }

    double is_sparse_proportion() const { 
        return 0;
    }

    using tatami::Matrix<Value_, Index_>::dense;

    using tatami::Matrix<Value_, Index_>::sparse;

private:
    template<bool oracle_, template<bool, bool, bool, typename, typename, typename> class Extractor_, typename ... Args_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(bool row, Index_ non_target_length, tatami::MaybeOracle<oracle_, Index_> oracle, Args_&& ... args) const {
        bool by_h5_row = (row != my_transpose);
        const auto& dim_stats = (by_h5_row ? my_firstdim_stats : my_seconddim_stats);

        tatami_chunked::SlabCacheStats<Index_> slab_stats(dim_stats.chunk_length, non_target_length, dim_stats.num_chunks, my_cache_size_in_elements, my_require_minimum_cache);
        if (slab_stats.max_slabs_in_cache > 0) {
            if (by_h5_row) {
                return std::make_unique<Extractor_<false, oracle_, true, Value_, Index_, CachedValue_> >(
                    my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats
                );
            } else {
                return std::make_unique<Extractor_<false, oracle_, false, Value_, Index_, CachedValue_> >(
                    my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats
                );
            }

        } else {
            if (by_h5_row) {
                return std::make_unique<Extractor_<true, oracle_, true, Value_, Index_, CachedValue_> >(
                    my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats
                );
            } else {
                return std::make_unique<Extractor_<true, oracle_, false, Value_, Index_, CachedValue_> >(
                    my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats
                );
            }
        }
    }

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options&) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, DenseMatrix_internal::Full>(row, full_non_target, false, full_non_target);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options&) const {
        return populate<false, DenseMatrix_internal::Block>(row, block_length, false, block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options&) const {
        auto nidx = indices_ptr->size();
        return populate<false, DenseMatrix_internal::Index>(row, nidx, false, std::move(indices_ptr));
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(dense(row, opt), full_non_target, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<false, Value_, Index_> >(dense(row, block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto ptr = dense(row, indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<false, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        const tatami::Options&) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<true, DenseMatrix_internal::Full>(row, full_non_target, std::move(oracle), full_non_target);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options&) 
    const {
        return populate<true, DenseMatrix_internal::Block>(row, block_length, std::move(oracle), block_start, block_length);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options&) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, DenseMatrix_internal::Index>(row, nidx, std::move(oracle), std::move(indices_ptr));
    }

    /***********************
     *** Oracular sparse ***
     ***********************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        const tatami::Options& opt) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), opt), full_non_target, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto ptr = dense(row, std::move(oracle), indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }
};

}

#endif
