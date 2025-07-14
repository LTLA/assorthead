#ifndef TATAMI_CHUNKED_CUSTOM_CHUNK_COORDINATOR_HPP
#define TATAMI_CHUNKED_CUSTOM_CHUNK_COORDINATOR_HPP

#include "tatami/tatami.hpp"
#include "DenseSlabFactory.hpp"
#include "SparseSlabFactory.hpp"
#include "OracularSlabCache.hpp"
#include "OracularSubsettedSlabCache.hpp"
#include "ChunkDimensionStats.hpp"

#include <vector>
#include <type_traits>
#include <algorithm>
#include <tuple>

#include "sanisizer/sanisizer.hpp"

namespace tatami_chunked {

namespace CustomChunkedMatrix_internal {

/******************
 *** Workspaces ***
 ******************/

template<typename ChunkValue_>
using DenseSingleWorkspace = std::vector<ChunkValue_>;

template<typename ChunkValue_, typename Index_>
class SparseSingleWorkspace {
public:
    SparseSingleWorkspace(Index_ target_chunkdim, Index_ non_target_chunkdim, bool needs_value, bool needs_index) : my_number(target_chunkdim) {
        if (needs_value) {
            my_value_pool.resize(sanisizer::product<decltype(my_value_pool.size())>(target_chunkdim, non_target_chunkdim));
            my_values.reserve(target_chunkdim);
            auto vptr = my_value_pool.data();
            for (Index_ p = 0; p < target_chunkdim; ++p, vptr += non_target_chunkdim) {
                my_values.push_back(vptr);
            }
        }
        if (needs_index) {
            my_index_pool.resize(sanisizer::product<decltype(my_index_pool.size())>(target_chunkdim, non_target_chunkdim));
            my_indices.reserve(target_chunkdim);
            auto iptr = my_index_pool.data();
            for (Index_ p = 0; p < target_chunkdim; ++p, iptr += non_target_chunkdim) {
                my_indices.push_back(iptr);
            }
        }
    }

    // Delete the copy constructors as we're passing out pointers.
    SparseSingleWorkspace(const SparseSingleWorkspace&) = delete;
    SparseSingleWorkspace& operator=(const SparseSingleWorkspace&) = delete;

    // Move constructors are okay though.
    SparseSingleWorkspace(SparseSingleWorkspace&&) = default;
    SparseSingleWorkspace& operator=(SparseSingleWorkspace&&) = default;

private:
    std::vector<ChunkValue_> my_value_pool;
    std::vector<Index_> my_index_pool;
    std::vector<ChunkValue_*> my_values;
    std::vector<Index_*> my_indices;
    std::vector<Index_> my_number;

public:
    std::vector<ChunkValue_*>& get_values() {
        return my_values;
    }

    std::vector<Index_*>& get_indices() {
        return my_indices;
    }

    std::vector<Index_>& get_number() {
        return my_number;
    }
};

/*******************
 *** Coordinator ***
 *******************/

template<bool sparse_, class ChunkValue_, typename Index_> 
class ChunkCoordinator {
public:
    ChunkCoordinator(ChunkDimensionStats<Index_> row_stats, ChunkDimensionStats<Index_> col_stats) :
        my_row_stats(std::move(row_stats)),
        my_col_stats(std::move(col_stats))
    {}

private:
    ChunkDimensionStats<Index_> my_row_stats;
    ChunkDimensionStats<Index_> my_col_stats;

public:
    // Number of chunks along the rows is equal to the number of chunks for
    // each column, and vice versa; hence the flipped definitions.
    Index_ get_num_chunks_per_row() const {
        return my_col_stats.num_chunks;
    }

    Index_ get_num_chunks_per_column() const {
        return my_row_stats.num_chunks;
    }

    Index_ get_nrow() const {
        return my_row_stats.dimension_extent;
    }

    Index_ get_ncol() const {
        return my_col_stats.dimension_extent;
    }

    Index_ get_chunk_nrow() const {
        return my_row_stats.chunk_length;
    }

    Index_ get_chunk_ncol() const {
        return my_col_stats.chunk_length;
    }

public:
    Index_ get_non_target_dim(bool row) const {
        if (row) {
            return my_col_stats.dimension_extent;
        } else {
            return my_row_stats.dimension_extent;
        }
    }

    Index_ get_target_chunkdim(bool row) const {
        if (row) {
            return my_row_stats.chunk_length;
        } else {
            return my_col_stats.chunk_length;
        }
    }

    Index_ get_non_target_chunkdim(bool row) const {
        if (row) {
            return my_col_stats.chunk_length;
        } else {
            return my_row_stats.chunk_length;
        }
    }

    // Overload that handles the truncated chunk at the bottom/right edges of each matrix.
    Index_ get_target_chunkdim(bool row, Index_ chunk_id) const {
        return get_chunk_length(row ? my_row_stats : my_col_stats, chunk_id);
    }

private:
    template<class ExtractFunction_>
    void extract_non_target_block(
        bool row,
        Index_ target_chunk_id, 
        Index_ non_target_block_start, 
        Index_ non_target_block_length, 
        ExtractFunction_ extract)
    const {
        auto non_target_chunkdim = get_non_target_chunkdim(row);
        Index_ non_target_start_chunk_index = non_target_block_start / non_target_chunkdim;
        Index_ non_target_start_pos = non_target_start_chunk_index * non_target_chunkdim;
        Index_ non_target_block_end = non_target_block_start + non_target_block_length;
        Index_ non_target_end_chunk_index = non_target_block_end / non_target_chunkdim + (non_target_block_end % non_target_chunkdim > 0); // i.e., integer ceiling.

        for (Index_ non_target_chunk_id = non_target_start_chunk_index; non_target_chunk_id < non_target_end_chunk_index; ++non_target_chunk_id) {
            Index_ from = (non_target_chunk_id == non_target_start_chunk_index ? non_target_block_start - non_target_start_pos : 0);
            Index_ to = (non_target_chunk_id + 1 == non_target_end_chunk_index ? non_target_block_end - non_target_start_pos : non_target_chunkdim);
            Index_ len = to - from;

            auto row_id = (row ? target_chunk_id : non_target_chunk_id);
            auto col_id = (row ? non_target_chunk_id : target_chunk_id);

            // No need to protect against a zero length, as it should be impossible
            // here (otherwise, start_chunk_index == end_chunk_index and we'd never iterate).
            if constexpr(sparse_) {
                extract(row_id, col_id, from, len, non_target_start_pos);
            } else {
                extract(row_id, col_id, from, len);
            }

            // yes, this is deliberate; '+ to' means that either we add 'non_target_chunkdim' or set it to 'non_target_block_end', the latter of which avoids overflow.
            non_target_start_pos += to;
        }
    }

    template<class ExtractFunction_>
    void extract_non_target_index(
        bool row,
        Index_ target_chunk_id, 
        const std::vector<Index_>& non_target_indices, 
        std::vector<Index_>& chunk_indices_buffer, 
        ExtractFunction_ extract)
    const {
        auto non_target_chunkdim = get_non_target_chunkdim(row);
        auto non_target_dim = get_non_target_dim(row);
        auto iIt = non_target_indices.begin();
        auto iEnd = non_target_indices.end();

        while (iIt != iEnd) {
            Index_ non_target_chunk_id = *iIt / non_target_chunkdim;
            Index_ non_target_start_pos = non_target_chunk_id * non_target_chunkdim;
            Index_ non_target_end_pos = std::min(non_target_dim - non_target_start_pos, non_target_chunkdim) + non_target_start_pos; // this convoluted method avoids overflow.

            chunk_indices_buffer.clear();
            do {
                chunk_indices_buffer.push_back(*iIt - non_target_start_pos);
                ++iIt;
            } while (iIt != iEnd && *iIt < non_target_end_pos);

            auto row_id = (row ? target_chunk_id : non_target_chunk_id);
            auto col_id = (row ? non_target_chunk_id : target_chunk_id);
            if constexpr(sparse_) {
                extract(row_id, col_id, chunk_indices_buffer, non_target_start_pos);
            } else {
                extract(row_id, col_id, chunk_indices_buffer);
            }
        }
    }

    typedef typename std::conditional<sparse_, typename SparseSlabFactory<ChunkValue_, Index_>::Slab, typename DenseSlabFactory<ChunkValue_>::Slab>::type Slab;
    typedef typename std::conditional<sparse_, SparseSingleWorkspace<ChunkValue_, Index_>, DenseSingleWorkspace<ChunkValue_> >::type SingleWorkspace;

public:
    // Extract a single element of the target dimension, using a contiguous
    // block on the non_target dimension. 
    template<class ChunkWorkspace_>
    std::pair<const Slab*, Index_> fetch_single(
        bool row,
        Index_ i,
        Index_ non_target_block_start, 
        Index_ non_target_block_length, 
        ChunkWorkspace_& chunk_workspace,
        SingleWorkspace& tmp_work,
        Slab& final_slab)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        Index_ target_chunk_id = i / target_chunkdim;
        Index_ target_chunk_offset = i % target_chunkdim;

        if constexpr(sparse_) {
            auto& final_num = *final_slab.number;
            final_num = 0;
            bool needs_value = !final_slab.values.empty();
            bool needs_index = !final_slab.indices.empty();

            extract_non_target_block(
                row,
                target_chunk_id,
                non_target_block_start,
                non_target_block_length, 
                [&](Index_ row_id, Index_ column_id, Index_ from, Index_ len, Index_ non_target_start_pos) -> void {
                    auto& tmp_values = tmp_work.get_values();
                    auto& tmp_indices = tmp_work.get_indices();
                    auto& tmp_number = tmp_work.get_number();

                    tmp_number[target_chunk_offset] = 0;
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        static_cast<Index_>(1),
                        from,
                        len,
                        tmp_values,
                        tmp_indices,
                        tmp_number.data(),
                        non_target_start_pos
                    );

                    auto count = tmp_number[target_chunk_offset];
                    if (needs_value) {
                        std::copy_n(tmp_values[target_chunk_offset], count, final_slab.values[0] + final_num);
                    }
                    if (needs_index) {
                        std::copy_n(tmp_indices[target_chunk_offset], count, final_slab.indices[0] + final_num);
                    }
                    final_num += count;
                }
            );

        } else {
            auto final_slab_ptr = final_slab.data;
            auto tmp_buffer_ptr = tmp_work.data();
            typedef decltype(tmp_work.size()) Size;

            extract_non_target_block(
                row,
                target_chunk_id,
                non_target_block_start,
                non_target_block_length, 
                [&](Index_ row_id, Index_ column_id, Index_ from, Index_ len) -> void {

                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        static_cast<Index_>(1),
                        from,
                        len,
                        tmp_buffer_ptr,
                        len
                    );

                    Size tmp_offset = sanisizer::product_unsafe<Size>(len, target_chunk_offset);
                    std::copy_n(tmp_buffer_ptr + tmp_offset, len, final_slab_ptr);
                    final_slab_ptr += len;
                }
            );
        }

        return std::make_pair(&final_slab, static_cast<Index_>(0));
    }

    // Extract a single element of the target dimension, using an indexed
    // subset on the non_target dimension.
    template<class ChunkWorkspace_>
    std::pair<const Slab*, Index_> fetch_single(
        bool row,
        Index_ i,
        const std::vector<Index_>& non_target_indices, 
        std::vector<Index_>& chunk_indices_buffer,
        ChunkWorkspace_& chunk_workspace,
        SingleWorkspace& tmp_work,
        Slab& final_slab)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        Index_ target_chunk_id = i / target_chunkdim;
        Index_ target_chunk_offset = i % target_chunkdim;

        if constexpr(sparse_) {
            auto& final_num = *final_slab.number;
            final_num = 0;
            bool needs_value = !final_slab.values.empty();
            bool needs_index = !final_slab.indices.empty();

            extract_non_target_index(
                row,
                target_chunk_id,
                non_target_indices,
                chunk_indices_buffer,
                [&](Index_ row_id, Index_ column_id, const std::vector<Index_>& chunk_indices, Index_ non_target_start_pos) -> void {
                    auto& tmp_values = tmp_work.get_values();
                    auto& tmp_indices = tmp_work.get_indices();
                    auto& tmp_number = tmp_work.get_number();

                    tmp_number[target_chunk_offset] = 0;
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        static_cast<Index_>(1),
                        chunk_indices,
                        tmp_values,
                        tmp_indices,
                        tmp_number.data(),
                        non_target_start_pos
                    );

                    auto count = tmp_number[target_chunk_offset];
                    if (needs_value) {
                        std::copy_n(tmp_values[target_chunk_offset], count, final_slab.values[0] + final_num);
                    }
                    if (needs_index) {
                        std::copy_n(tmp_indices[target_chunk_offset], count, final_slab.indices[0] + final_num);
                    }
                    final_num += count;
                }
            );

        } else {
            auto final_slab_ptr = final_slab.data;
            auto tmp_buffer_ptr = tmp_work.data();
            typedef decltype(tmp_work.size()) Size;

            extract_non_target_index(
                row,
                target_chunk_id,
                non_target_indices,
                chunk_indices_buffer,
                [&](Index_ row_id, Index_ column_id, const std::vector<Index_>& chunk_indices) -> void {
                    auto nidx = chunk_indices.size();
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        static_cast<Index_>(1),
                        chunk_indices,
                        tmp_buffer_ptr,
                        nidx
                    );

                    Size tmp_offset = static_cast<Size>(nidx) * static_cast<Size>(target_chunk_offset);
                    std::copy_n(tmp_buffer_ptr + tmp_offset, nidx, final_slab_ptr);
                    final_slab_ptr += nidx;
                }
            );
        }

        return std::make_pair(&final_slab, static_cast<Index_>(0));
    }

private:
    // Extract a contiguous block of the target dimension, using a contiguous block on the non_target dimension.
    template<class ChunkWorkspace_>
    void fetch_block(
        bool row,
        Index_ target_chunk_id, 
        Index_ target_chunk_offset, 
        Index_ target_chunk_length, 
        Index_ non_target_block_start, 
        Index_ non_target_block_length, 
        Slab& slab, 
        ChunkWorkspace_& chunk_workspace)
    const {
        if constexpr(sparse_) {
            std::fill_n(slab.number, get_target_chunkdim(row), 0);

            extract_non_target_block(
                row,
                target_chunk_id,
                non_target_block_start,
                non_target_block_length, 
                [&](Index_ row_id, Index_ column_id, Index_ from, Index_ len, Index_ non_target_start_pos) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        target_chunk_length,
                        from,
                        len,
                        slab.values,
                        slab.indices,
                        slab.number,
                        non_target_start_pos
                    );
                }
            );

        } else {
            auto slab_ptr = slab.data;

            extract_non_target_block(
                row,
                target_chunk_id,
                non_target_block_start,
                non_target_block_length, 
                [&](Index_ row_id, Index_ column_id, Index_ from, Index_ len) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        target_chunk_length,
                        from,
                        len,
                        slab_ptr,
                        non_target_block_length
                    );
                    slab_ptr += len;
                }
            );
        }
    }

    // Extract a contiguous block of the target dimension, using an indexed subset on the non_target dimension.
    template<class ChunkWorkspace_>
    void fetch_block(
        bool row,
        Index_ target_chunk_id, 
        Index_ target_chunk_offset, 
        Index_ target_chunk_length, 
        const std::vector<Index_>& non_target_indices, 
        std::vector<Index_>& chunk_indices_buffer,
        Slab& slab, 
        ChunkWorkspace_& chunk_workspace)
    const {
        if constexpr(sparse_) {
            std::fill_n(slab.number, get_target_chunkdim(row), 0);

            extract_non_target_index(
                row,
                target_chunk_id,
                non_target_indices,
                chunk_indices_buffer,
                [&](Index_ row_id, Index_ column_id, const std::vector<Index_>& chunk_indices, Index_ non_target_start_pos) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        target_chunk_length,
                        chunk_indices,
                        slab.values,
                        slab.indices,
                        slab.number,
                        non_target_start_pos
                    );
                }
            );

        } else {
            auto slab_ptr = slab.data;
            Index_ stride = non_target_indices.size();
            extract_non_target_index(
                row,
                target_chunk_id,
                non_target_indices,
                chunk_indices_buffer,
                [&](Index_ row_id, Index_ column_id, const std::vector<Index_>& chunk_indices) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_chunk_offset,
                        target_chunk_length,
                        chunk_indices,
                        slab_ptr,
                        stride
                    );
                    slab_ptr += chunk_indices.size();
                }
            );
        }
    }

private:
    // Extract an indexed subset of the target dimension, using a contiguous block on the non_target dimension.
    template<class ChunkWorkspace_>
    void fetch_index(
        bool row,
        Index_ target_chunk_id,
        const std::vector<Index_>& target_indices, 
        Index_ non_target_block_start, 
        Index_ non_target_block_length, 
        Slab& slab, 
        ChunkWorkspace_& chunk_workspace)
    const {
        if constexpr(sparse_) {
            std::fill_n(slab.number, get_target_chunkdim(row), 0);
            extract_non_target_block(
                row,
                target_chunk_id,
                non_target_block_start,
                non_target_block_length,
                [&](Index_ row_id, Index_ column_id, Index_ from, Index_ len, Index_ non_target_start_pos) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_indices,
                        from,
                        len,
                        slab.values,
                        slab.indices,
                        slab.number,
                        non_target_start_pos
                    );
                }
            );

        } else {
            auto slab_ptr = slab.data;
            extract_non_target_block(
                row,
                target_chunk_id,
                non_target_block_start,
                non_target_block_length,
                [&](Index_ row_id, Index_ column_id, Index_ from, Index_ len) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_indices,
                        from,
                        len,
                        slab_ptr,
                        non_target_block_length
                    );
                    slab_ptr += len;
                }
            );
        }
    }

    // Extract an indexed subset of the target dimension, using an indexed subset on the non_target dimension.
    template<class ChunkWorkspace_>
    void fetch_index(
        bool row,
        Index_ target_chunk_id,
        const std::vector<Index_>& target_indices,
        const std::vector<Index_>& non_target_indices,
        std::vector<Index_>& chunk_indices_buffer,
        Slab& slab, 
        ChunkWorkspace_& chunk_workspace)
    const {
        if constexpr(sparse_) {
            std::fill_n(slab.number, get_target_chunkdim(row), 0);
            extract_non_target_index(
                row,
                target_chunk_id,
                non_target_indices,
                chunk_indices_buffer,
                [&](Index_ row_id, Index_ column_id, const std::vector<Index_>& chunk_indices, Index_ non_target_start_pos) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_indices,
                        chunk_indices,
                        slab.values,
                        slab.indices,
                        slab.number,
                        non_target_start_pos
                    );
                }
            );

        } else {
            auto slab_ptr = slab.data;
            Index_ stride = non_target_indices.size();
            extract_non_target_index(
                row,
                target_chunk_id,
                non_target_indices,
                chunk_indices_buffer, 
                [&](Index_ row_id, Index_ column_id, const std::vector<Index_>& chunk_indices) -> void {
                    chunk_workspace.extract(
                        row_id,
                        column_id,
                        row,
                        target_indices,
                        chunk_indices,
                        slab_ptr,
                        stride
                    );
                    slab_ptr += chunk_indices.size();
                }
            );
        }
    }

public:
    // Obtain the slab containing the 'i'-th element of the target dimension.
    template<class ChunkWorkspace_, class Cache_, class Factory_>
    std::pair<const Slab*, Index_> fetch_myopic(
        bool row,
        Index_ i, 
        Index_ block_start,
        Index_ block_length,
        ChunkWorkspace_& chunk_workspace,
        Cache_& cache,
        Factory_& factory)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        Index_ target_chunk_id = i / target_chunkdim;
        Index_ target_chunk_offset = i % target_chunkdim;
        auto& out = cache.find(
            target_chunk_id,
            /* create = */ [&]() -> Slab {
                return factory.create();
            },
            /* populate = */ [&](Index_ id, Slab& slab) -> void {
                fetch_block(row, id, 0, get_target_chunkdim(row, id), block_start, block_length, slab, chunk_workspace);
            }
        );
        return std::make_pair(&out, target_chunk_offset);
    }

    template<class ChunkWorkspace_, class Cache_, class Factory_>
    std::pair<const Slab*, Index_> fetch_myopic(
        bool row,
        Index_ i, 
        const std::vector<Index_>& indices,
        std::vector<Index_>& tmp_indices,
        ChunkWorkspace_& chunk_workspace,
        Cache_& cache,
        Factory_& factory)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        Index_ target_chunk_id = i / target_chunkdim;
        Index_ target_chunk_offset = i % target_chunkdim;
        auto& out = cache.find(
            target_chunk_id,
            /* create = */ [&]() -> Slab {
                return factory.create();
            },
            /* populate = */ [&](Index_ id, Slab& slab) -> void {
                fetch_block(row, id, 0, get_target_chunkdim(row, id), indices, tmp_indices, slab, chunk_workspace);
            }
        );
        return std::make_pair(&out, target_chunk_offset);
    }

public:
    template<class ChunkWorkspace_, class Cache_, class Factory_>
    std::pair<const Slab*, Index_> fetch_oracular(
        bool row,
        Index_ block_start,
        Index_ block_length,
        ChunkWorkspace_& chunk_workspace,
        Cache_& cache,
        Factory_& factory)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        return cache.next(
            /* identify = */ [&](Index_ i) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(i / target_chunkdim, i % target_chunkdim);
            },
            /* create = */ [&]() -> Slab {
                return factory.create();
            },
            /* populate =*/ [&](std::vector<std::pair<Index_, Slab*> >& to_populate) -> void {
                for (auto& p : to_populate) {
                    fetch_block(row, p.first, 0, get_target_chunkdim(row, p.first), block_start, block_length, *(p.second), chunk_workspace);
                }
            }
        );
    }

    template<class ChunkWorkspace_, class Cache_, class Factory_>
    std::pair<const Slab*, Index_> fetch_oracular(
        bool row,
        const std::vector<Index_>& indices,
        std::vector<Index_>& chunk_indices_buffer,
        ChunkWorkspace_& chunk_workspace,
        Cache_& cache,
        Factory_& factory)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        return cache.next(
            /* identify = */ [&](Index_ i) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(i / target_chunkdim, i % target_chunkdim);
            },
            /* create = */ [&]() -> Slab {
                return factory.create();
            },
            /* populate =*/ [&](std::vector<std::pair<Index_, Slab*> >& to_populate) -> void {
                for (auto& p : to_populate) {
                    fetch_block(row, p.first, 0, get_target_chunkdim(row, p.first), indices, chunk_indices_buffer, *(p.second), chunk_workspace);
                }
            }
        );
    }

public:
    template<class ChunkWorkspace_, class Cache_, class Factory_>
    std::pair<const Slab*, Index_> fetch_oracular_subsetted(
        bool row,
        Index_ block_start,
        Index_ block_length,
        ChunkWorkspace_& chunk_workspace,
        Cache_& cache,
        Factory_& factory)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        return cache.next(
            /* identify = */ [&](Index_ i) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(i / target_chunkdim, i % target_chunkdim);
            },
            /* create = */ [&]() -> Slab {
                return factory.create();
            },
            /* populate =*/ [&](std::vector<std::tuple<Index_, Slab*, const OracularSubsettedSlabCacheSelectionDetails<Index_>*> >& in_need) -> void {
                for (const auto& p : in_need) {
                    auto id = std::get<0>(p);
                    auto ptr = std::get<1>(p);
                    auto sub = std::get<2>(p);
                    switch (sub->selection) {
                        case OracularSubsettedSlabCacheSelectionType::FULL:
                            fetch_block(row, id, 0, get_target_chunkdim(row, id), block_start, block_length, *ptr, chunk_workspace);
                            break;
                        case OracularSubsettedSlabCacheSelectionType::BLOCK:
                            fetch_block(row, id, sub->block_start, sub->block_length, block_start, block_length, *ptr, chunk_workspace);
                            break;
                        case OracularSubsettedSlabCacheSelectionType::INDEX:
                            fetch_index(row, id, sub->indices, block_start, block_length, *ptr, chunk_workspace);
                            break;
                    }
                }
            }
        );
    }

    template<class ChunkWorkspace_, class Cache_, class Factory_>
    std::pair<const Slab*, Index_> fetch_oracular_subsetted(
        bool row,
        const std::vector<Index_>& indices,
        std::vector<Index_>& chunk_indices_buffer,
        ChunkWorkspace_& chunk_workspace,
        Cache_& cache,
        Factory_& factory)
    const {
        Index_ target_chunkdim = get_target_chunkdim(row);
        return cache.next(
            /* identify = */ [&](Index_ i) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(i / target_chunkdim, i % target_chunkdim);
            },
            /* create = */ [&]() -> Slab {
                return factory.create();
            },
            /* populate =*/ [&](std::vector<std::tuple<Index_, Slab*, const OracularSubsettedSlabCacheSelectionDetails<Index_>*> >& in_need) -> void {
                for (const auto& p : in_need) {
                    auto id = std::get<0>(p);
                    auto ptr = std::get<1>(p);
                    auto sub = std::get<2>(p);
                    switch (sub->selection) {
                        case OracularSubsettedSlabCacheSelectionType::FULL:
                            fetch_block(row, id, 0, get_target_chunkdim(row, id), indices, chunk_indices_buffer, *ptr, chunk_workspace);
                            break;
                        case OracularSubsettedSlabCacheSelectionType::BLOCK:
                            fetch_block(row, id, sub->block_start, sub->block_length, indices, chunk_indices_buffer, *ptr, chunk_workspace);
                            break;
                        case OracularSubsettedSlabCacheSelectionType::INDEX:
                            fetch_index(row, id, sub->indices, indices, chunk_indices_buffer, *ptr, chunk_workspace);
                            break;
                    }
                }
            }
        );
    }
};

}

}

#endif
