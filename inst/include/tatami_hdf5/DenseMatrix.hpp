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

template<typename Index_, typename TmpValue_, typename OutputValue_>
void extract_indices(
    bool h5_row_is_target,
    Index_ chunk_start,
    Index_ chunk_length,
    const std::vector<Index_>& indices,
    TmpValue_* tmp_buffer,
    OutputValue_* final_buffer,
    Components& comp
) {
    const Index_ num_indices = indices.size();
    if (num_indices == 0) {
        return;
    }

    hsize_t offset[2];
    hsize_t count[2];

    int target_dim = 1 - h5_row_is_target;
    offset[target_dim] = chunk_start;
    count[target_dim] = chunk_length;

    // The strategy here is to just extract the entire range into our tmp_buffer and then cherry-pick the indices we want into the final_buffer.
    // This is because the hyperslab unions are still brutally slow in 1.14.6, so it's not worth creating a non-contiguous union.
    int non_target_dim = h5_row_is_target;
    const auto first_index = indices.front();
    offset[non_target_dim] = first_index;
    const auto non_target_range = indices.back() - first_index + 1;
    count[non_target_dim] = non_target_range;

    comp.dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
    comp.memspace.setExtentSimple(2, count);
    comp.memspace.selectAll();
    comp.dataset.read(tmp_buffer, define_mem_type<OutputValue_>(), comp.memspace, comp.dataspace);

    if (h5_row_is_target) {
        for (Index_ t = 0; t < chunk_length; ++t) {
            for (Index_ i = 0; i < num_indices; ++i) {
                const auto in_offset = sanisizer::nd_offset<std::size_t>(indices[i] - first_index, non_target_range, t);
                const auto out_offset = sanisizer::nd_offset<std::size_t>(i, num_indices, t);
                final_buffer[out_offset] = tmp_buffer[in_offset];
            }
        }
    } else {
        for (Index_ i = 0; i < num_indices; ++i) {
            const auto in_offset = sanisizer::product_unsafe<std::size_t>(chunk_length, indices[i] - first_index);
            const auto out_offset = sanisizer::product_unsafe<std::size_t>(chunk_length, i);
            std::copy_n(tmp_buffer + in_offset, chunk_length, final_buffer + out_offset);
        }
    }
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

template<bool oracle_, typename Index_, typename CachedValue_>
class SoloCore {
public:
    SoloCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        [[maybe_unused]] tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats, // only listed here for compatibility with the other constructors.
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        [[maybe_unused]] const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        const bool h5_row_is_target,
        const Index_ non_target_range_for_indexed
    ) :
        my_oracle(std::move(oracle)),
        my_h5_row_is_target(h5_row_is_target),
        my_non_target_buffer_for_indexed(sanisizer::cast<decltype(my_non_target_buffer_for_indexed.size())>(non_target_range_for_indexed))
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
    bool my_h5_row_is_target;
    std::vector<CachedValue_> my_non_target_buffer_for_indexed;

public:
    template<typename Value_>
    const Value_* fetch_block(Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&]() -> void {
            extract_block(my_h5_row_is_target, i, static_cast<Index_>(1), block_start, block_length, buffer, *my_h5comp);
        });
        return buffer;
    }

    template<typename Value_>
    const Value_* fetch_indices(Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        serialize([&]() -> void {
            extract_indices(my_h5_row_is_target, i, static_cast<Index_>(1), indices, my_non_target_buffer_for_indexed.data(), buffer, *my_h5comp);
        });
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
class MyopicCore {
public:
    MyopicCore(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_>, // for consistency with the oracular version.
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        const bool h5_row_is_target,
        const Index_ non_target_range_for_indexed
    ) :
        my_dim_stats(std::move(target_dim_stats)),
        my_factory(slab_stats), 
        my_cache(slab_stats.max_slabs_in_cache),
        my_h5_row_is_target(h5_row_is_target),
        my_non_target_buffer_for_indexed(sanisizer::product<decltype(my_non_target_buffer_for_indexed.size())>(non_target_range_for_indexed, my_dim_stats.chunk_length))
    {
        initialize(file_name, dataset_name, my_h5comp);
        if (!my_h5_row_is_target) {
            sanisizer::resize(my_transposition_buffer, slab_stats.slab_size_in_elements);
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

    bool my_h5_row_is_target;
    std::vector<CachedValue_> my_transposition_buffer;

    std::vector<CachedValue_> my_non_target_buffer_for_indexed;

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

                if (my_h5_row_is_target) {
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
                extract_block(my_h5_row_is_target, start, length, block_start, block_length, buf, *my_h5comp);
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
                extract_indices(my_h5_row_is_target, start, length, indices, my_non_target_buffer_for_indexed.data(), buf, *my_h5comp);
            }
        );
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
class OracularBlockNormal {
public:
    OracularBlockNormal(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        [[maybe_unused]] const bool h5_row_is_target, // for consistency with the solo/myopic version
        [[maybe_unused]] const Index_ non_target_range_for_indexed
    ) :
        my_dim_stats(std::move(target_dim_stats)),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache),
        my_slab_size(slab_stats.slab_size_in_elements),
        my_memory_pool(sanisizer::product<decltype(my_memory_pool.size())>(slab_stats.max_slabs_in_cache, my_slab_size))
    {
        initialize(file_name, dataset_name, my_h5comp);
    }

    ~OracularBlockNormal() {
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

public:
    template<typename Value_>
    const Value_* fetch_block([[maybe_unused]] Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
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

                const auto dest = my_memory_pool.data();
                std::size_t reused_offset = 0;
                for (auto& x : to_reuse) {
                    auto& cur_offset = x.second->offset;
                    if (cur_offset != reused_offset) {
                        std::copy_n(dest + cur_offset, my_slab_size, dest + reused_offset);
                        cur_offset = reused_offset;
                    }
                    reused_offset += my_slab_size;
                }

                // If we don't have to transpose, we can extract data directly into the cache buffer.
                // To do so, we try to form as many contiguous hyperslabs as possible, reducing the number of calls into the HDF5 library.
                // Then we update the slab pointers to refer to the relevant slices of memory in the cache.
                // We don't use hyperslab unions because they're slow as shit for non-contiguous hyperslabs.
                sort_by_field(chunks, [](const std::pair<Index_, Slab*>& x) -> Index_ { return x.first; });

                Index_ run_chunk_id = chunks.front().first;
                Index_ run_length = tatami_chunked::get_chunk_length(my_dim_stats, run_chunk_id);
                auto run_offset = reused_offset;
                chunks.front().second->offset = run_offset;
                auto total_used_offset = run_offset + my_slab_size;
                Index_ last_chunk_id = run_chunk_id;

                hsize_t count[2];
                hsize_t offset[2];
                count[1] = block_length;
                offset[1] = block_start;

                serialize([&]() -> void {
                    for (decltype(chunks.size()) ci = 1, cend = chunks.size(); ci < cend; ++ci) {
                        const auto& current_chunk = chunks[ci];
                        const auto current_chunk_id = current_chunk.first;

                        if (current_chunk_id - last_chunk_id > 1) { // save the existing run of chunks as one hyperslab, and start a new run.
                            count[0] = run_length;
                            offset[0] = run_chunk_id * my_dim_stats.chunk_length;
                            my_h5comp->memspace.setExtentSimple(2, count);
                            my_h5comp->memspace.selectAll();
                            my_h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
                            my_h5comp->dataset.read(dest + run_offset, define_mem_type<CachedValue_>(), my_h5comp->memspace, my_h5comp->dataspace);

                            run_chunk_id = current_chunk_id;
                            run_length = 0;
                            run_offset = total_used_offset;
                        }

                        run_length += tatami_chunked::get_chunk_length(my_dim_stats, current_chunk_id);
                        current_chunk.second->offset = total_used_offset;
                        total_used_offset += my_slab_size;
                        last_chunk_id = current_chunk.first;
                    }

                    count[0] = run_length;
                    offset[0] = run_chunk_id * my_dim_stats.chunk_length;
                    my_h5comp->memspace.setExtentSimple(2, count);
                    my_h5comp->memspace.selectAll();
                    my_h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
                    my_h5comp->dataset.read(dest + run_offset, define_mem_type<CachedValue_>(), my_h5comp->memspace, my_h5comp->dataspace);
                });
            }
        );

        auto ptr = my_memory_pool.data() + info.first->offset + sanisizer::product_unsafe<std::size_t>(block_length, info.second);
        std::copy_n(ptr, block_length, buffer);
        return buffer;
    }
};

template<typename Index_, class Slab_, typename CachedValue_>
CachedValue_* transpose_chunks(
    std::vector<std::pair<Index_, Slab_*> >& chunks, 
    const tatami_chunked::ChunkDimensionStats<Index_>& dim_stats,
    const Index_ non_target_length,
    CachedValue_* transposition_buffer_ptr
) {
    if (non_target_length > 1) {
        for (const auto& current_chunk : chunks) {
            const Index_ chunk_length = tatami_chunked::get_chunk_length(dim_stats, current_chunk.first);
            if (chunk_length <= 1) {
                continue;
            }

            // We actually swap the pointers here, so the slab pointers might not point to the factory pool after this!
            // Shouldn't matter as long as neither of them are exposed to the user.
            auto& dest = current_chunk.second->data;
            tatami::transpose(dest, non_target_length, chunk_length, transposition_buffer_ptr);
            std::swap(dest, transposition_buffer_ptr);
        }
    }

    return transposition_buffer_ptr;
}

template<typename Index_, typename CachedValue_>
class OracularBlockTransposed  {
public:
    OracularBlockTransposed(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        [[maybe_unused]] const bool h5_row_is_target, // for consistency with the solo/myopic versions
        [[maybe_unused]] const Index_ non_target_range_for_indexed
    ) :
        my_dim_stats(std::move(target_dim_stats)),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache),
        my_slab_size(slab_stats.slab_size_in_elements),
        my_memory_pool(sanisizer::product<decltype(my_memory_pool.size())>(slab_stats.max_slabs_in_cache, my_slab_size))
    {
        initialize(file_name, dataset_name, my_h5comp);
        sanisizer::resize(my_transposition_buffer, my_slab_size);
        my_transposition_buffer_ptr = my_transposition_buffer.data();
    }

    ~OracularBlockTransposed() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;

    struct Slab {
        CachedValue_* data;
    };

    tatami_chunked::OracularSlabCache<Index_, Index_, Slab, false> my_cache;
    std::size_t my_slab_size;
    std::vector<CachedValue_> my_memory_pool;
    std::size_t my_offset = 0;

    bool my_h5_row_is_target;
    std::vector<CachedValue_> my_transposition_buffer;
    CachedValue_* my_transposition_buffer_ptr;

public:
    template<typename Value_>
    const Value_* fetch_block([[maybe_unused]] Index_ i, Index_ block_start, Index_ block_length, Value_* buffer) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_dim_stats.chunk_length, current % my_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                Slab output;
                output.data = my_memory_pool.data() + my_offset;
                my_offset += my_slab_size;
                return output;
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                // If we have to transpose, we extract slab-by-slab and transpose each one as it comes in.
                // It's too hard to try to do an in-place transposition after reading everything into the cache.
                hsize_t count[2];
                hsize_t offset[2];
                count[0] = block_length;
                offset[0] = block_start;

                serialize([&]() -> void {
                    for (const auto& current_chunk : chunks) {
                        const auto chunk_length = tatami_chunked::get_chunk_length(my_dim_stats, current_chunk.first);
                        count[1] = chunk_length;
                        offset[1] = current_chunk.first * my_dim_stats.chunk_length;
                        my_h5comp->memspace.setExtentSimple(2, count);
                        my_h5comp->memspace.selectAll();
                        my_h5comp->dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
                        my_h5comp->dataset.read(current_chunk.second->data, define_mem_type<CachedValue_>(), my_h5comp->memspace, my_h5comp->dataspace);
                    }
                });

                // Transposition is done outside the serial section to unblock other threads.
                my_transposition_buffer_ptr = transpose_chunks(chunks, my_dim_stats, block_length, my_transposition_buffer_ptr);
            }
        );

        auto ptr = info.first->data + sanisizer::product_unsafe<std::size_t>(block_length, info.second);
        std::copy_n(ptr, block_length, buffer);
        return buffer;
    }
};

template<typename Index_, typename CachedValue_>
class OracularIndexed {
public:
    OracularIndexed(
        const std::string& file_name,
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        const bool h5_row_is_target,
        const Index_ non_target_range_for_indexed
    ) :
        my_dim_stats(std::move(target_dim_stats)),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache),
        my_slab_size(slab_stats.slab_size_in_elements),
        my_memory_pool(sanisizer::product<decltype(my_memory_pool.size())>(slab_stats.max_slabs_in_cache, my_slab_size)),
        my_h5_row_is_target(h5_row_is_target),
        my_non_target_buffer_for_indexed(sanisizer::product<decltype(my_non_target_buffer_for_indexed.size())>(non_target_range_for_indexed, my_dim_stats.chunk_length))
    {
        initialize(file_name, dataset_name, my_h5comp);
        if (!my_h5_row_is_target) {
            sanisizer::resize(my_transposition_buffer, my_slab_size);
            my_transposition_buffer_ptr = my_transposition_buffer.data();
        }
    }

    ~OracularIndexed() {
        destroy(my_h5comp);
    }

private:
    std::unique_ptr<Components> my_h5comp;
    tatami_chunked::ChunkDimensionStats<Index_> my_dim_stats;

    struct Slab {
        CachedValue_* data;
    };

    tatami_chunked::OracularSlabCache<Index_, Index_, Slab, false> my_cache;
    std::size_t my_slab_size;
    std::vector<CachedValue_> my_memory_pool;
    std::size_t my_offset = 0;

    bool my_h5_row_is_target;
    std::vector<CachedValue_> my_transposition_buffer;
    CachedValue_* my_transposition_buffer_ptr;

    std::vector<CachedValue_> my_non_target_buffer_for_indexed;

public:
    template<typename Value_>
    const Value_* fetch_indices([[maybe_unused]] Index_ i, const std::vector<Index_>& indices, Value_* buffer) {
        const Index_ num_indices = indices.size();
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_dim_stats.chunk_length, current % my_dim_stats.chunk_length);
            }, 
            /* create = */ [&]() -> Slab {
                Slab output;
                output.data = my_memory_pool.data() + my_offset;
                my_offset += my_slab_size;
                return output;
            },
            /* populate = */ [&](std::vector<std::pair<Index_, Slab*> >& chunks) -> void {
                serialize([&]() -> void {
                    for (const auto& current_chunk : chunks) {
                        extract_indices(
                            my_h5_row_is_target,
                            current_chunk.first * my_dim_stats.chunk_length,
                            tatami_chunked::get_chunk_length(my_dim_stats, current_chunk.first),
                            indices,
                            my_non_target_buffer_for_indexed.data(),
                            current_chunk.second->data,
                            *my_h5comp
                        );
                    }
                });

                // Transposition is done outside the serial section to unblock other threads.
                if (!my_h5_row_is_target) {
                    my_transposition_buffer_ptr = transpose_chunks(chunks, my_dim_stats, num_indices, my_transposition_buffer_ptr);
                }
            }
        );

        auto ptr = info.first->data + sanisizer::product_unsafe<std::size_t>(num_indices, info.second);
        std::copy_n(ptr, num_indices, buffer);
        return buffer;
    }
};

// COMMENT: technically, for all oracular extractors, we could pretend that the chunk length on the target dimension is 1.
// This would allow us to only extract the desired indices on each call, allowing for more efficient use of the cache.
//
// The problem is that we would get harshly penalized for any chunk reuse outside of the current prediction cycle,
// where the chunk would need to be read from disk again if the exact elements weren't already in the cache.
// Consider a chunk length of 2 where we want to access the following indices {0, 2, 4, 5, 2, 3} and our cache is large enough to hold 4 elements.
//
// - With our current approach, the first prediction cycle would load the first and second chunks, i.e., {0, 1, 2, 3}.
//   The next prediction cycle would load the third chunk and re-use the second chunk to get {2, 3, 4, 5}.
// - With a chunk length of 1, the first prediction cycle would load all three chunks but only keep {0, 2, 4, 5}.
//   The next prediction cycle would need to reload the second chunk to get {3}.
//
// This access pattern might not be uncommon after applying a DelayedSubset with shuffled rows/columns. 

/***************************
 *** Concrete subclasses ***
 ***************************/

enum class OracularMode : char { NOT_APPLICABLE, BLOCK_NORMAL, BLOCK_TRANSPOSED, INDEXED };

template<bool solo_, bool oracle_, OracularMode omode_, typename Index_, typename CachedValue_>
using DenseCore = typename std::conditional<solo_, 
      SoloCore<oracle_, Index_, CachedValue_>,
      typename std::conditional<!oracle_,
          MyopicCore<Index_, CachedValue_>,
          typename std::conditional<omode_ == OracularMode::BLOCK_NORMAL,
              OracularBlockNormal<Index_, CachedValue_>,
              typename std::conditional<omode_ == OracularMode::BLOCK_TRANSPOSED,
                  OracularBlockTransposed<Index_, CachedValue_>,
                  OracularIndexed<Index_, CachedValue_>
              >::type
          >::type
      >::type
>::type;

template<bool solo_, bool oracle_, OracularMode omode_, typename Value_, typename Index_, typename CachedValue_>
class Full final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Full(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim,
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        bool h5_row_is_target
    ) :
        my_core(
            file_name, 
            dataset_name, 
            std::move(target_dim_stats),
            std::move(oracle),
            slab_stats,
            h5_row_is_target,
            0
        ),
        my_non_target_dim(non_target_dim)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, 0, my_non_target_dim, buffer);
    }

private:
    DenseCore<solo_, oracle_, omode_, Index_, CachedValue_> my_core;
    Index_ my_non_target_dim;
};

template<bool solo_, bool oracle_, OracularMode omode_, typename Value_, typename Index_, typename CachedValue_> 
class Block final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Block(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        const bool h5_row_is_target
    ) :
        my_core( 
            file_name, 
            dataset_name, 
            std::move(target_dim_stats),
            std::move(oracle),
            slab_stats,
            h5_row_is_target,
            0
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_block(i, my_block_start, my_block_length, buffer);
    }

private:
    DenseCore<solo_, oracle_, omode_, Index_, CachedValue_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool solo_, bool oracle_, OracularMode omode_, typename Value_, typename Index_, typename CachedValue_>
class Index final : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    Index(
        const std::string& file_name, 
        const std::string& dataset_name, 
        tatami_chunked::ChunkDimensionStats<Index_> target_dim_stats,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami_chunked::SlabCacheStats<Index_>& slab_stats,
        const bool h5_row_is_target
    ) :
        my_core(
            file_name,
            dataset_name, 
            std::move(target_dim_stats),
            std::move(oracle),
            slab_stats,
            h5_row_is_target,
            indices_ptr->empty() ? 0 : (indices_ptr->back() - indices_ptr->front() + 1)
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        return my_core.fetch_indices(i, *my_indices_ptr, buffer);
    }

private:
    DenseCore<solo_, oracle_, omode_, Index_, CachedValue_> my_core;
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

            my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(
                sanisizer::cast<Index_>(dims[0]),
                sanisizer::cast<Index_>(chunk_dims[0])
            );
            my_seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(
                sanisizer::cast<Index_>(dims[1]),
                sanisizer::cast<Index_>(chunk_dims[1])
            );
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
    template<bool oracle_, template<bool, bool, DenseMatrix_internal::OracularMode, typename, typename, typename> class Extractor_, typename ... Args_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate(
        bool row,
        Index_ non_target_length,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Args_&& ... args
    ) const {
        bool by_h5_row = (row != my_transpose);
        const auto& dim_stats = (by_h5_row ? my_firstdim_stats : my_seconddim_stats);

        tatami_chunked::SlabCacheStats<Index_> slab_stats(
            dim_stats.chunk_length,
            non_target_length,
            dim_stats.num_chunks,
            my_cache_size_in_elements,
            my_require_minimum_cache
        );

        if (slab_stats.max_slabs_in_cache == 0) {
            return std::make_unique<Extractor_<true, oracle_, DenseMatrix_internal::OracularMode::NOT_APPLICABLE, Value_, Index_, CachedValue_> >(
                my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats, by_h5_row
            );
        }

        if constexpr(oracle_) {
            typedef DenseMatrix_internal::Index<false, oracle_, DenseMatrix_internal::OracularMode::INDEXED, Value_, Index_, CachedValue_> CurIndex;
            if constexpr(std::is_same<CurIndex, Extractor_<false, oracle_, DenseMatrix_internal::OracularMode::INDEXED, Value_, Index_, CachedValue_> >::value) {
                return std::make_unique<CurIndex>(
                    my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats, by_h5_row
                );
            } else {
                if (by_h5_row) {
                    return std::make_unique<Extractor_<false, oracle_, DenseMatrix_internal::OracularMode::BLOCK_NORMAL, Value_, Index_, CachedValue_> >(
                        my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats, by_h5_row
                    );
                } else {
                    return std::make_unique<Extractor_<false, oracle_, DenseMatrix_internal::OracularMode::BLOCK_TRANSPOSED, Value_, Index_, CachedValue_> >(
                        my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats, by_h5_row
                    );
                }
            }
        } else {
            return std::make_unique<Extractor_<false, oracle_, DenseMatrix_internal::OracularMode::NOT_APPLICABLE, Value_, Index_, CachedValue_> >(
                my_file_name, my_dataset_name, dim_stats, std::move(oracle), std::forward<Args_>(args)..., slab_stats, by_h5_row
            );
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
        const tatami::Options&
    ) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<true, DenseMatrix_internal::Full>(row, full_non_target, std::move(oracle), full_non_target);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options&
    ) const {
        return populate<true, DenseMatrix_internal::Block>(row, block_length, std::move(oracle), block_start, block_length);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options&
    ) const {
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
        const tatami::Options& opt
    ) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), opt), full_non_target, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt
    ) const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt
    ) const {
        auto ptr = dense(row, std::move(oracle), indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(std::move(ptr), std::move(indices_ptr), opt);
    }
};

}

#endif
