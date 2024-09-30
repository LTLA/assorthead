#ifndef TATAMI_TILEDB_SPARSE_MATRIX_HPP
#define TATAMI_TILEDB_SPARSE_MATRIX_HPP

#include "tatami_chunked/tatami_chunked.hpp"
#include <tiledb/tiledb>

#include "serialize.hpp"

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

/**
 * @file SparseMatrix.hpp
 * @brief TileDB-backed sparse matrix.
 */

namespace tatami_tiledb {

/**
 * @brief Options for sparse TileDB extraction.
 */
struct SparseMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all tiles required to read a row/column during a `tatami::SparseExtractor::fetch()` or `tatami::SparseExtractor::Fetch()` call.
     * This allows us to re-use the cached tiles when adjacent rows/columns are requested, rather than re-reading them from disk.
     *
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored if `require_minimum_cache` is `true`.
     */
    size_t maximum_cache_size = 100000000;

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that all tiles overlapping one row (or a slice/subset thereof) can be retained in memory,
     * so that the same tiles are not repeatedly re-read from disk when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

/**
 * @cond
 */
namespace SparseMatrix_internal {

typedef ::tatami_tiledb::internal::Components Components;
typedef ::tatami_tiledb::internal::VariablyTypedDimension Dimension;
typedef ::tatami_tiledb::internal::VariablyTypedVector CacheBuffer;

struct Workspace {
    CacheBuffer values;
    CacheBuffer target_indices;
    CacheBuffer non_target_indices;
};

inline size_t execute_query(
    const Components& tdb_comp,
    tiledb::Subarray& subarray,
    const std::string& attribute,
    bool row, 
    const std::string& target_dimname,
    const std::string& non_target_dimname,
    Workspace& work,
    size_t general_work_offset,
    size_t target_index_work_offset,
    size_t work_length,
    bool needs_value,
    bool needs_index)
{
    tiledb::Query query(tdb_comp.ctx, tdb_comp.array);
    query.set_subarray(subarray);
    query.set_layout(row ? TILEDB_ROW_MAJOR : TILEDB_COL_MAJOR);

    work.target_indices.set_data_buffer(query, target_dimname, target_index_work_offset, work_length);
    if (needs_value) {
        work.values.set_data_buffer(query, attribute, general_work_offset, work_length);
    }
    if (needs_index) {
        work.non_target_indices.set_data_buffer(query, non_target_dimname, general_work_offset, work_length);
    }

    if (query.submit() != tiledb::Query::Status::COMPLETE) {
        throw std::runtime_error("failed to read sparse data from TileDB");
    }

    return query.result_buffer_elements()[target_dimname].second;
}

/********************
 *** Core classes ***
 ********************/

template<typename Index_>
struct MyopicCacheParameters {
    Index_ chunk_length;
    size_t slab_size_in_elements;
    size_t max_slabs_in_cache;
};

template<typename Index_>
class MyopicCore {
public:
    MyopicCore(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        [[maybe_unused]] Index_ non_target_length, // provided for consistency with the other constructors.
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> oracle, 
        const MyopicCacheParameters<Index_>& cache_stats,
        bool needs_value,
        bool needs_index) :
        my_tdb_comp(tdb_comp),
        my_attribute(attribute),
        my_row(row),
        my_target_dim_extent(target_dim_extent),
        my_tdb_target_dim(tdb_target_dim),
        my_target_dimname(target_dimname),
        my_tdb_non_target_dim(tdb_non_target_dim),
        my_non_target_dimname(non_target_dimname),
        my_target_chunk_length(cache_stats.chunk_length),
        my_slab_size(cache_stats.slab_size_in_elements),
        my_needs_value(needs_value),
        my_needs_index(needs_index),
        my_cache(cache_stats.max_slabs_in_cache)
    {
        // Only storing one slab at a time for the target indices.
        my_work.target_indices.reset(my_tdb_target_dim.type(), my_slab_size);

        size_t total_cache_size = my_slab_size * cache_stats.max_slabs_in_cache;
        if (my_needs_value) {
            my_work.values.reset(tdb_type, total_cache_size);
        }
        if (my_needs_index) {
            my_work.non_target_indices.reset(my_tdb_non_target_dim.type(), total_cache_size);
        }
    }

private:
    const Components& my_tdb_comp;
    const std::string& my_attribute;

    bool my_row;
    Index_ my_target_dim_extent;
    const Dimension& my_tdb_target_dim;
    const std::string& my_target_dimname;
    const Dimension& my_tdb_non_target_dim;
    const std::string& my_non_target_dimname;

    Index_ my_target_chunk_length;
    size_t my_slab_size;
    bool my_needs_value;
    bool my_needs_index;
    Workspace my_work;
    std::vector<std::pair<Index_, Index_> > my_counts;

    struct Slab {
        size_t offset;
        std::vector<size_t> indptrs;
    };
    size_t my_offset = 0;
    tatami_chunked::LruSlabCache<Index_, Slab> my_cache;

private:
    template<class Configure_>
    std::pair<size_t, size_t> fetch_raw(Index_ i, Configure_ configure) {
        Index_ chunk = i / my_target_chunk_length;
        Index_ index = i % my_target_chunk_length;

        const auto& info = my_cache.find(
            chunk, 
            /* create = */ [&]() -> Slab {
                Slab output;
                output.offset = my_offset;
                my_offset += my_slab_size;
                return output;
            },
            /* populate = */ [&](Index_ id, Slab& contents) -> void {
                Index_ chunk_start = id * my_target_chunk_length;
                Index_ chunk_length = std::min(my_target_dim_extent - chunk_start, my_target_chunk_length);

                size_t num_nonzero = 0;
                serialize([&]() {
                    tiledb::Subarray subarray(my_tdb_comp.ctx, my_tdb_comp.array);
                    int rowdex = my_row;
                    my_tdb_target_dim.add_range(subarray, 1 - rowdex, chunk_start, chunk_length);
                    configure(subarray, rowdex);
                    num_nonzero = execute_query(
                        my_tdb_comp,
                        subarray,
                        my_attribute, 
                        my_row,
                        my_target_dimname,
                        my_non_target_dimname,
                        my_work,
                        contents.offset,
                        0,
                        my_slab_size,
                        my_needs_value,
                        my_needs_index
                    );
                });

                auto& indptrs = contents.indptrs;
                indptrs.clear();
                indptrs.resize(chunk_length + 1);

                if (num_nonzero) {
                    my_work.target_indices.compact(0, num_nonzero, my_tdb_target_dim, my_counts);
                    for (const auto& cnts : my_counts) {
                        indptrs[cnts.first - chunk_start + 1] = cnts.second;
                    }
                    for (Index_ i = 1; i <= chunk_length; ++i) {
                        indptrs[i] += indptrs[i - 1];
                    }
                }
            }
        );

        auto start = info.indptrs[index];
        return std::make_pair(info.offset + start, info.indptrs[index + 1] - start);
    }

public:
    std::pair<size_t, size_t> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        return fetch_raw(i, [&](tiledb::Subarray& subarray, int rowdex) {
            my_tdb_non_target_dim.add_range(subarray, rowdex, block_start, block_length);
        });
    }

    std::pair<size_t, size_t> fetch_indices(Index_ i, const std::vector<Index_>& indices) {
        return fetch_raw(i, [&](tiledb::Subarray& subarray, int rowdex) {
            tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
                my_tdb_non_target_dim.add_range(subarray, rowdex, s, l);
            });
        });
    }

public:
    const Workspace& get_workspace() const {
        return my_work;
    }

    bool get_needs_value() const {
        return my_needs_value;
    }

    bool get_needs_index() const {
        return my_needs_index;
    }

    const Dimension& get_tdb_non_target_dim() const {
        return my_tdb_non_target_dim;
    }
};

// The general idea with the oracular extractors is to either:
//
// - Extract each target dimension element directly, if the cell order within each tile corresponds to the desired target dimension (i.e., 'row').
// - Extract the tile-wise chunk of target dimension elements, if the cell order within each tile is not the same as the target dimension.
//
// This means that we need to vary the chunk length of each slab from 1 or the tile extent, depending on the cell order of the TileDB array.
// In addition, we use a variable slab cache that adjusts to the number of non-zero elements in each slab.

template<typename Index_>
struct OracularCacheParameters {
    Index_ chunk_length;
    size_t max_cache_size_in_elements;
};

template<typename Index_>
class OracularCore {
public:
    OracularCore(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        Index_ non_target_length,
        tatami::MaybeOracle<true, Index_> oracle, 
        const OracularCacheParameters<Index_>& cache_stats,
        bool needs_value,
        bool needs_index) :
        my_tdb_comp(tdb_comp),
        my_attribute(attribute),
        my_row(row),
        my_target_dim_extent(target_dim_extent),
        my_tdb_target_dim(tdb_target_dim),
        my_target_dimname(target_dimname),
        my_tdb_non_target_dim(tdb_non_target_dim),
        my_non_target_dimname(non_target_dimname),
        my_target_chunk_length(cache_stats.chunk_length),
        my_max_slab_size(static_cast<size_t>(non_target_length) * my_target_chunk_length),
        my_needs_value(needs_value),
        my_needs_index(needs_index),
        my_cache(std::move(oracle), cache_stats.max_cache_size_in_elements)
    {
        my_work.target_indices.reset(my_tdb_target_dim.type(), cache_stats.max_cache_size_in_elements);
        if (my_needs_value) {
            my_work.values.reset(tdb_type, cache_stats.max_cache_size_in_elements);
        }
        if (my_needs_index) {
            my_work.non_target_indices.reset(my_tdb_non_target_dim.type(), cache_stats.max_cache_size_in_elements);
        }
    }

private:
    const Components& my_tdb_comp;
    const std::string& my_attribute;

    bool my_row;
    Index_ my_target_dim_extent;
    const Dimension& my_tdb_target_dim;
    const std::string& my_target_dimname;
    const Dimension& my_tdb_non_target_dim;
    const std::string& my_non_target_dimname;

    Index_ my_target_chunk_length;
    size_t my_max_slab_size;
    bool my_needs_value;
    bool my_needs_index;
    Workspace my_work;
    std::vector<std::pair<Index_, Index_> > my_counts;

    struct Slab {
        size_t offset;
        std::vector<size_t> indptrs;
    };
    tatami_chunked::OracularVariableSlabCache<Index_, Index_, Slab, size_t> my_cache;

private:
    template<class Function_>
    static void sort_by_field(std::vector<std::pair<Index_, size_t> >& indices, Function_ field) {
        auto comp = [&field](const std::pair<Index_, size_t>& l, const std::pair<Index_, size_t>& r) -> bool {
            return field(l) < field(r);
        };
        if (!std::is_sorted(indices.begin(), indices.end(), comp)) {
            std::sort(indices.begin(), indices.end(), comp);
        }
    }

    template<class Configure_>
    std::pair<size_t, size_t> fetch_raw([[maybe_unused]] Index_ i, Configure_ configure) {
        auto info = my_cache.next(
            /* identify = */ [&](Index_ current) -> std::pair<Index_, Index_> {
                return std::pair<Index_, Index_>(current / my_target_chunk_length, current % my_target_chunk_length);
            }, 
            /* upper_size = */ [&](Index_) -> size_t {
                return my_max_slab_size;
            },
            /* actual_size = */ [&](Index_, const Slab& slab) -> size_t {
                return slab.indptrs.back();
            },
            /* create = */ [&]() -> Slab {
                return Slab();
            },
            /* populate = */ [&](std::vector<std::pair<Index_, size_t> >& to_populate, std::vector<std::pair<Index_, size_t> >& to_reuse, std::vector<Slab>& all_slabs) {
                // Defragmenting the existing chunks. We sort by offset to make 
                // sure that we're not clobbering in-use slabs during the copy().
                sort_by_field(to_reuse, [&](const std::pair<Index_, size_t>& x) -> size_t { return all_slabs[x.second].offset; });
                size_t running_offset = 0;
                for (auto& x : to_reuse) {
                    auto& reused_slab = all_slabs[x.second];
                    auto& cur_offset = reused_slab.offset;
                    auto num_nonzero = reused_slab.indptrs.back();
                    if (cur_offset != running_offset) {
                        if (my_needs_value) {
                            my_work.values.shift(cur_offset, num_nonzero, running_offset);
                        }
                        if (my_needs_index) {
                            my_work.non_target_indices.shift(cur_offset, num_nonzero, running_offset);
                        }
                        cur_offset = running_offset;
                    }
                    running_offset += num_nonzero;
                }

                // Collapsing runs of consecutive ranges into a single range;
                // otherwise, making union of ranges. This allows a single TileDb call
                // to populate the contiguous memory pool that we made available after
                // defragmentation; then we just update the slab pointers to refer
                // to the slices of memory corresponding to each slab.
                sort_by_field(to_populate, [](const std::pair<Index_, size_t>& x) -> Index_ { return x.first; });

                size_t num_nonzero = 0;
                serialize([&]() -> void {
                    tiledb::Subarray subarray(my_tdb_comp.ctx, my_tdb_comp.array);
                    int rowdex = my_row;
                    configure(subarray, rowdex);

                    Index_ run_chunk_id = to_populate.front().first;
                    Index_ run_chunk_start = run_chunk_id * my_target_chunk_length;
                    Index_ run_length = std::min(my_target_dim_extent - run_chunk_start, my_target_chunk_length);

                    int dimdex = 1 - rowdex;
                    for (size_t ci = 1, cend = to_populate.size(); ci < cend; ++ci) {
                        Index_ current_chunk_id = to_populate[ci].first;
                        Index_ current_chunk_start = current_chunk_id * my_target_chunk_length;

                        if (current_chunk_id - run_chunk_id > 1) { // save the existing run of to_populate as one range, and start a new run.
                            my_tdb_target_dim.add_range(subarray, dimdex, run_chunk_start, run_length);
                            run_chunk_id = current_chunk_id;
                            run_chunk_start = current_chunk_start;
                            run_length = 0;
                        }

                        run_length += std::min(my_target_dim_extent - current_chunk_start, my_target_chunk_length);
                    }

                    my_tdb_target_dim.add_range(subarray, dimdex, run_chunk_start, run_length);
                    num_nonzero = execute_query(
                        my_tdb_comp,
                        subarray,
                        my_attribute, 
                        my_row,
                        my_target_dimname,
                        my_non_target_dimname,
                        my_work,
                        running_offset,
                        running_offset,
                        to_populate.size() * my_max_slab_size,
                        my_needs_value,
                        my_needs_index
                    );
                });

                my_work.target_indices.compact(running_offset, num_nonzero, my_tdb_target_dim, my_counts);

                auto cIt = my_counts.begin(), cEnd = my_counts.end();
                for (auto& si : to_populate) {
                    auto& populate_slab = all_slabs[si.second];
                    populate_slab.offset = running_offset;

                    Index_ chunk_start = si.first * my_target_chunk_length;
                    Index_ chunk_length = std::min(my_target_dim_extent - chunk_start, my_target_chunk_length);
                    Index_ chunk_end = chunk_start + chunk_length;

                    auto& slab_indptrs = populate_slab.indptrs;
                    slab_indptrs.clear();
                    slab_indptrs.resize(chunk_length + 1);

                    while (cIt != cEnd && cIt->first < chunk_end) {
                        slab_indptrs[cIt->first - chunk_start + 1] = cIt->second;
                        ++cIt;
                    }

                    for (Index_ i = 1; i <= chunk_length; ++i) {
                        slab_indptrs[i] += slab_indptrs[i - 1];
                    }
                    running_offset += slab_indptrs.back();
                }
            }
        );

        const auto& indptrs = info.first->indptrs;
        auto start = indptrs[info.second];
        return std::make_pair(info.first->offset + start, indptrs[info.second + 1] - start);
    }

public:
    std::pair<size_t, size_t> fetch_block(Index_ i, Index_ block_start, Index_ block_length) {
        return fetch_raw(i, [&](tiledb::Subarray& subarray, int rowdex) {
            my_tdb_non_target_dim.add_range(subarray, rowdex, block_start, block_length);
        });
    }

    std::pair<size_t, size_t> fetch_indices(Index_ i, const std::vector<Index_>& indices) {
        return fetch_raw(i, [&](tiledb::Subarray& subarray, int rowdex) {
            tatami::process_consecutive_indices<Index_>(indices.data(), indices.size(), [&](Index_ s, Index_ l) {
                my_tdb_non_target_dim.add_range(subarray, rowdex, s, l);
            });
        });
    }

public:
    const Workspace& get_workspace() const {
        return my_work;
    }

    bool get_needs_value() const {
        return my_needs_value;
    }

    bool get_needs_index() const {
        return my_needs_index;
    }

    const Dimension& get_tdb_non_target_dim() const {
        return my_tdb_non_target_dim;
    }
};

template<bool oracle_, typename Index_>
using SparseCore = typename std::conditional<oracle_, OracularCore<Index_>, MyopicCore<Index_> >::type;

template<bool oracle_, typename Index_>
using CacheParameters = typename std::conditional<oracle_, OracularCacheParameters<Index_>, MyopicCacheParameters<Index_> >::type;

/*************************
 *** Sparse subclasses ***
 *************************/

template<typename Value_, typename Index_>
tatami::SparseRange<Value_, Index_> fill_sparse_range(
    const Workspace& work,
    size_t work_start,
    size_t work_length,
    const Dimension& non_target_dim,
    Value_* vbuffer,
    Index_* ibuffer,
    bool needs_value,
    bool needs_index)
{
    tatami::SparseRange<Value_, Index_> output;
    output.number = work_length;
    if (needs_value) {
        work.values.copy(work_start, work_length, vbuffer);
        output.value = vbuffer;
    }
    if (needs_index) {
        work.non_target_indices.copy(work_start, work_length, non_target_dim, ibuffer);
        output.index = ibuffer;
    }
    return output;
}

template<bool oracle_, typename Value_, typename Index_>
class SparseFull : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseFull(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        Index_ non_target_dim,
        const CacheParameters<oracle_, Index_>& cache_parameters,
        bool needs_value,
        bool needs_index) :
        my_core(
            tdb_comp,
            attribute,
            row,
            target_dim_extent,
            target_dimname,
            tdb_target_dim,
            non_target_dimname,
            tdb_non_target_dim,
            tdb_type,
            non_target_dim, 
            std::move(oracle),
            cache_parameters,
            needs_value,
            needs_index
        ),
        my_non_target_dim(non_target_dim)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto info = my_core.fetch_block(i, 0, my_non_target_dim);
        return fill_sparse_range(my_core.get_workspace(), info.first, info.second, my_core.get_tdb_non_target_dim(), vbuffer, ibuffer, my_core.get_needs_value(), my_core.get_needs_index());
    }

private:
    SparseCore<oracle_, Index_> my_core;
    Index_ my_non_target_dim;
};

template<bool oracle_, typename Value_, typename Index_>
class SparseBlock : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseBlock(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        Index_ block_start,
        Index_ block_length,
        const CacheParameters<oracle_, Index_>& cache_parameters,
        bool needs_value,
        bool needs_index) :
        my_core( 
            tdb_comp,
            attribute,
            row,
            target_dim_extent,
            target_dimname,
            tdb_target_dim,
            non_target_dimname,
            tdb_non_target_dim,
            tdb_type,
            block_length, 
            std::move(oracle),
            cache_parameters,
            needs_value,
            needs_index
        ),
        my_block_start(block_start),
        my_block_length(block_length)
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto info = my_core.fetch_block(i, my_block_start, my_block_length);
        return fill_sparse_range(my_core.get_workspace(), info.first, info.second, my_core.get_tdb_non_target_dim(), vbuffer, ibuffer, my_core.get_needs_value(), my_core.get_needs_index());
    }

private:
    SparseCore<oracle_, Index_> my_core;
    Index_ my_block_start, my_block_length;
};

template<bool oracle_, typename Value_, typename Index_>
class SparseIndex : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseIndex(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const CacheParameters<oracle_, Index_>& cache_parameters,
        bool needs_value,
        bool needs_index) :
        my_core(
            tdb_comp,
            attribute,
            row,
            target_dim_extent,
            target_dimname,
            tdb_target_dim,
            non_target_dimname,
            tdb_non_target_dim,
            tdb_type,
            indices_ptr->size(), 
            std::move(oracle),
            cache_parameters,
            needs_value,
            needs_index
        ),
        my_indices_ptr(std::move(indices_ptr))
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* vbuffer, Index_* ibuffer) {
        auto info = my_core.fetch_indices(i, *my_indices_ptr);
        return fill_sparse_range(my_core.get_workspace(), info.first, info.second, my_core.get_tdb_non_target_dim(), vbuffer, ibuffer, my_core.get_needs_value(), my_core.get_needs_index());
    }

private:
    SparseCore<oracle_, Index_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
};

/************************
 *** Dense subclasses ***
 ************************/

template<bool oracle_, typename Value_, typename Index_>
class DenseFull : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseFull(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_dim_extent,
        const CacheParameters<oracle_, Index_>& cache_parameters,
        [[maybe_unused]] bool needs_value, // for consistency with Sparse* constructors.
        [[maybe_unused]] bool needs_index) :
        my_core(
            tdb_comp,
            attribute,
            row,
            target_dim_extent,
            target_dimname,
            tdb_target_dim,
            non_target_dimname,
            tdb_non_target_dim,
            tdb_type,
            non_target_dim_extent, 
            std::move(oracle),
            cache_parameters,
            /* needs_value = */ true,
            /* needs_index = */ true
        ),
        my_non_target_dim_extent(non_target_dim_extent),
        my_holding_value(my_non_target_dim_extent),
        my_holding_index(my_non_target_dim_extent)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto info = my_core.fetch_block(i, 0, my_non_target_dim_extent);
        const auto& work = my_core.get_workspace();
        work.values.copy(info.first, info.second, my_holding_value.data());
        work.non_target_indices.copy(info.first, info.second, my_core.get_tdb_non_target_dim(), my_holding_index.data());
        std::fill_n(buffer, my_non_target_dim_extent, 0);
        for (size_t i = 0; i < info.second; ++i) {
            buffer[my_holding_index[i]] = my_holding_value[i];
        }
        return buffer;
    }

private:
    SparseCore<oracle_, Index_> my_core;
    Index_ my_non_target_dim_extent;
    std::vector<Value_> my_holding_value;
    std::vector<Index_> my_holding_index;
};

template<bool oracle_, typename Value_, typename Index_> 
class DenseBlock : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseBlock(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start,
        Index_ block_length,
        const CacheParameters<oracle_, Index_>& cache_parameters,
        [[maybe_unused]] bool needs_value, // for consistency with Sparse* constructors.
        [[maybe_unused]] bool needs_index) :
        my_core( 
            tdb_comp,
            attribute,
            row,
            target_dim_extent,
            target_dimname,
            tdb_target_dim,
            non_target_dimname,
            tdb_non_target_dim,
            tdb_type,
            block_length, 
            std::move(oracle),
            cache_parameters,
            /* needs_value = */ true,
            /* needs_index = */ true
        ),
        my_block_start(block_start),
        my_block_length(block_length),
        my_holding_value(block_length),
        my_holding_index(block_length)
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto info = my_core.fetch_block(i, my_block_start, my_block_length);
        const auto& work = my_core.get_workspace();
        work.values.copy(info.first, info.second, my_holding_value.data());
        work.non_target_indices.copy(info.first, info.second, my_core.get_tdb_non_target_dim(), my_holding_index.data());
        std::fill_n(buffer, my_block_length, 0);
        for (size_t i = 0; i < info.second; ++i) {
            buffer[my_holding_index[i] - my_block_start] = my_holding_value[i];
        }
        return buffer;
    }

private:
    SparseCore<oracle_, Index_> my_core;
    Index_ my_block_start, my_block_length;
    std::vector<Value_> my_holding_value;
    std::vector<Index_> my_holding_index;
};

template<bool oracle_, typename Value_, typename Index_>
class DenseIndex : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseIndex(
        const Components& tdb_comp,
        const std::string& attribute, 
        bool row,
        Index_ target_dim_extent,
        const std::string& target_dimname, 
        const Dimension& tdb_target_dim,
        const std::string& non_target_dimname, 
        const Dimension& tdb_non_target_dim,
        tiledb_datatype_t tdb_type,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr,
        const CacheParameters<oracle_, Index_>& cache_parameters,
        [[maybe_unused]] bool needs_value, // for consistency with Sparse* constructors.
        [[maybe_unused]] bool needs_index) :
        my_core(
            tdb_comp,
            attribute,
            row,
            target_dim_extent,
            target_dimname,
            tdb_target_dim,
            non_target_dimname,
            tdb_non_target_dim,
            tdb_type,
            indices_ptr->size(), 
            std::move(oracle),
            cache_parameters,
            /* needs_value = */ true,
            /* needs_index = */ true
        ),
        my_indices_ptr(std::move(indices_ptr)),
        my_holding_value(my_indices_ptr->size()),
        my_holding_index(my_indices_ptr->size())
    {
        const auto& indices = *my_indices_ptr;
        if (!indices.empty()) {
            auto idx_start = indices.front();
            my_remapping.resize(indices.back() - idx_start + 1);
            for (size_t j = 0, end = indices.size(); j < end; ++j) {
                my_remapping[indices[j] - idx_start] = j;
            }
        }
    }

    const Value_* fetch(Index_ i, Value_* buffer) {
        const auto& indices = *my_indices_ptr;

        if (!indices.empty()) {
            auto info = my_core.fetch_indices(i, indices);
            const auto& work = my_core.get_workspace();
            work.values.copy(info.first, info.second, my_holding_value.data());
            work.non_target_indices.copy(info.first, info.second, my_core.get_tdb_non_target_dim(), my_holding_index.data());
            auto idx_start = indices.front();
            std::fill_n(buffer, indices.size(), 0);
            for (size_t i = 0; i < info.second; ++i) {
                buffer[my_remapping[my_holding_index[i] - idx_start]] = my_holding_value[i];
            }
        }

        return buffer;
    }

private:
    SparseCore<oracle_, Index_> my_core;
    tatami::VectorPtr<Index_> my_indices_ptr; 
    std::vector<Index_> my_remapping;
    std::vector<Value_> my_holding_value;
    std::vector<Index_> my_holding_index;
};

}
/**
 * @endcond
 */

/**
 * @brief TileDB-backed sparse matrix.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 *
 * Numeric sparse matrix stored in a 2-dimensional TileDB array.
 * Chunks of data are loaded from file as needed by `tatami::Extractor` objects, with additional caching to avoid repeated reads from disk.
 * The size of cached chunks is determined by the extent of the tiles in the TileDB array.
 *
 * The TileDB library is thread-safe so no additional work is required to use this class in parallel code.
 * Nonetheless, users can force all calls to TileDB to occur in serial by defining the `TATAMI_TILEDB_PARALLEL_LOCK` macro.
 * This should be a function-like macro that accepts a function and executes it inside a user-defined serial section.
 */
template<typename Value_, typename Index_>
class SparseMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     * @param options Further options.
     */
    SparseMatrix(const std::string& uri, std::string attribute, const SparseMatrixOptions& options) : my_attribute(std::move(attribute)) {
        serialize([&]() {
            // Serializing the deleter. 
            my_tdb_comp.reset(new SparseMatrix_internal::Components(uri), [](SparseMatrix_internal::Components* ptr) {
                serialize([&]() {
                    delete ptr;
                });
            });

            auto schema = my_tdb_comp->array.schema();
            if (schema.array_type() != TILEDB_SPARSE) {
                throw std::runtime_error("TileDB array should be sparse");
            }
            my_cell_order = schema.cell_order();

            my_cache_size_in_bytes = options.maximum_cache_size;
            my_require_minimum_cache = options.require_minimum_cache;

            if (!schema.has_attribute(my_attribute)) {
                throw std::runtime_error("no attribute '" + my_attribute + "' is present in the TileDB array");
            }
            auto attr = schema.attribute(my_attribute);
            my_tdb_type = attr.type();

            tiledb::Domain domain = schema.domain();
            if (domain.ndim() != 2) {
                throw std::runtime_error("TileDB array should have exactly two dimensions");
            }

            tiledb::Dimension first_dim = domain.dimension(0);
            my_first_dimname = first_dim.name();
            my_tdb_first_dim.reset(first_dim);
            Index_ first_extent = my_tdb_first_dim.extent<Index_>();
            Index_ first_tile = my_tdb_first_dim.tile<Index_>();
            my_firstdim_stats = tatami_chunked::ChunkDimensionStats<Index_>(first_extent, first_tile);

            tiledb::Dimension second_dim = domain.dimension(1);
            my_second_dimname = second_dim.name();
            my_tdb_second_dim.reset(second_dim);
            Index_ second_extent = my_tdb_second_dim.extent<Index_>();
            Index_ second_tile = my_tdb_second_dim.tile<Index_>();
            my_seconddim_stats = tatami_chunked::ChunkDimensionStats<Index_>(second_extent, second_tile);

            // Favoring extraction on the dimension that involves pulling out fewer chunks per dimension element.
            auto tiles_per_firstdim = (second_extent / second_tile) + (second_extent % second_tile > 0);
            auto tiles_per_seconddim = (first_extent / first_tile) + (first_extent % first_tile > 0);
            my_prefer_firstdim = tiles_per_firstdim <= tiles_per_seconddim;
        });
    }

    /**
     * @param uri File path (or some other appropriate location) of the TileDB array.
     * @param attribute Name of the attribute containing the data of interest.
     */
    SparseMatrix(const std::string& uri, std::string attribute) : SparseMatrix(uri, std::move(attribute), SparseMatrixOptions()) {}

private:
    std::shared_ptr<SparseMatrix_internal::Components> my_tdb_comp;
    tiledb_layout_t my_cell_order;
    tiledb_datatype_t my_tdb_type;

    std::string my_attribute;
    size_t my_cache_size_in_bytes;
    bool my_require_minimum_cache;

    std::string my_first_dimname, my_second_dimname;
    SparseMatrix_internal::Dimension my_tdb_first_dim, my_tdb_second_dim;
    tatami_chunked::ChunkDimensionStats<Index_> my_firstdim_stats, my_seconddim_stats;

    bool my_prefer_firstdim;

private:
    Index_ nrow_internal() const {
        return my_firstdim_stats.dimension_extent;
    }

    Index_ ncol_internal() const {
        return my_seconddim_stats.dimension_extent;
    }

public:
    Index_ nrow() const {
        return nrow_internal();
    }

    Index_ ncol() const {
        return ncol_internal();
    }

    bool is_sparse() const {
        return true;
    }

    double is_sparse_proportion() const {
        return 1;
    }

    bool prefer_rows() const {
        return my_prefer_firstdim;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(my_prefer_firstdim);
    }

    bool uses_oracle(bool) const {
        // It won't necessarily be used, but if the cache is empty,
        // the oracle definitely _won't_ be used.
        return my_cache_size_in_bytes > 0;
    }

private:
    template<
        bool oracle_,
        template<typename, typename> class Interface_, 
        template<bool, typename, typename> class Extractor_, 
        typename ... Args_
    >
    std::unique_ptr<Interface_<Value_, Index_> > populate(
        bool row,
        Index_ non_target_length,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        const tatami::Options& opt,
        Args_&& ... args) 
    const {
        const auto& target_dim_stats = (row ? my_firstdim_stats : my_seconddim_stats);
        const auto& target_dimname = (row ? my_first_dimname : my_second_dimname);
        const auto& non_target_dimname = (row ? my_second_dimname : my_first_dimname);
        const auto& tdb_target_dim = (row ? my_tdb_first_dim : my_tdb_second_dim);
        const auto& tdb_non_target_dim = (row ? my_tdb_second_dim : my_tdb_first_dim);

        size_t nonzero_size = 0;
        if (opt.sparse_extract_value) {
            nonzero_size += ::tatami_tiledb::internal::determine_type_size(my_tdb_type);
        }
        if (opt.sparse_extract_index) {
            nonzero_size += ::tatami_tiledb::internal::determine_type_size(tdb_non_target_dim.type());
        }

        if constexpr(oracle_) {
            // Add the target index size because we always need it for bulk
            // reads in the oracular case. This is not needed in the
            // myopic case because we only read one slab at a time.
            nonzero_size += ::tatami_tiledb::internal::determine_type_size(tdb_target_dim.type());

            SparseMatrix_internal::OracularCacheParameters<Index_> cache_params;
            cache_params.max_cache_size_in_elements = my_cache_size_in_bytes / nonzero_size;

            // If we're asking for rows and the cell order is row-major or
            // we want columns and the cell order is column-major, each
            // element of the target dimension has its contents stored
            // contiguously in TileDB's data tiles and can be easily
            // extracted on an individual basis; thus each element is
            // considered a separate slab and we set the chunk_length to 1.
            // 
            // Otherwise, it's likely that an element of the target
            // dimension will overlap multiple data tiles within each space
            // tile, so we might as well extract the entire space tile's 
            // elements on the target dimension.
            cache_params.chunk_length = (row == (my_cell_order == TILEDB_ROW_MAJOR) ? 1 : target_dim_stats.chunk_length);

            // Ensure that there's enough space for every dimension element.
            // If this can't be guaranteed, we set the cache to only be able to
            // hold a single dimension element. This is effectively the same as
            // not doing any caching at all, as a hypothetical SoloCore would
            // still need to allocate enough memory for a single dimension
            // element to create a buffer for the TileDB libary.
            size_t max_slab_size = static_cast<size_t>(non_target_length) * cache_params.chunk_length; // cast to avoid overflow.
            if (my_require_minimum_cache) {
                cache_params.max_cache_size_in_elements = std::max(cache_params.max_cache_size_in_elements, max_slab_size);
            } else if (cache_params.max_cache_size_in_elements < max_slab_size) {
                cache_params.max_cache_size_in_elements = non_target_length;
                cache_params.chunk_length = 1;
            }

            return std::make_unique<Extractor_<oracle_, Value_, Index_> >(
                *my_tdb_comp,
                my_attribute, 
                row,
                target_dim_stats.dimension_extent,
                target_dimname,
                tdb_target_dim,
                non_target_dimname,
                tdb_non_target_dim,
                my_tdb_type,
                std::move(oracle),
                std::forward<Args_>(args)...,
                cache_params,
                opt.sparse_extract_value,
                opt.sparse_extract_index
            );

        } else {
            tatami_chunked::SlabCacheStats raw_params(
                target_dim_stats.chunk_length,
                non_target_length,
                target_dim_stats.num_chunks,
                my_cache_size_in_bytes,
                nonzero_size,
                my_require_minimum_cache
            );

            // No need to have a dedicated SoloCore for uncached extraction,
            // because it would still need to hold a single Workspace. We
            // instead reuse the MyopicCore's code with a chunk length of 1 to
            // achieve the same memory usage. This has a mild perf hit from the
            // LRU but perf already sucks without caching so who cares.
            SparseMatrix_internal::MyopicCacheParameters<Index_> cache_params;
            if (raw_params.max_slabs_in_cache > 0) {
                cache_params.chunk_length = target_dim_stats.chunk_length;
                cache_params.slab_size_in_elements = raw_params.slab_size_in_elements;
                cache_params.max_slabs_in_cache = raw_params.max_slabs_in_cache;
            } else {
                cache_params.chunk_length = 1;
                cache_params.slab_size_in_elements = non_target_length;
                cache_params.max_slabs_in_cache = 1;
            }

            return std::make_unique<Extractor_<oracle_, Value_, Index_> >(
                *my_tdb_comp, 
                my_attribute,
                row,
                target_dim_stats.dimension_extent,
                target_dimname,
                tdb_target_dim,
                non_target_dimname,
                tdb_non_target_dim,
                my_tdb_type,
                std::move(oracle), 
                std::forward<Args_>(args)...,
                cache_params,
                opt.sparse_extract_value,
                opt.sparse_extract_index
            );
        }
    }

    static tatami::Options set_extract_all(tatami::Options opt) {
        // Resetting these options so that the slab size estimates are
        // correctly estimated for dense extractors, regardless of 'opt'. 
        opt.sparse_extract_value = true;
        opt.sparse_extract_index = true;
        return opt;
    }

    /********************
     *** Myopic dense ***
     ********************/
public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, tatami::MyopicDenseExtractor, SparseMatrix_internal::DenseFull>(row, full_non_target, false, set_extract_all(opt), full_non_target);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::MyopicDenseExtractor, SparseMatrix_internal::DenseBlock>(row, block_length, false, set_extract_all(opt), block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto nidx = indices_ptr->size();
        return populate<false, tatami::MyopicDenseExtractor, SparseMatrix_internal::DenseIndex>(row, nidx, false, set_extract_all(opt), std::move(indices_ptr));
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<false, tatami::MyopicSparseExtractor, SparseMatrix_internal::SparseFull>(row, full_non_target, false, opt, full_non_target);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate<false, tatami::MyopicSparseExtractor, SparseMatrix_internal::SparseBlock>(row, block_length, false, opt, block_start, block_length);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto nidx = indices_ptr->size();
        return populate<false, tatami::MyopicSparseExtractor, SparseMatrix_internal::SparseIndex>(row, nidx, false, opt, std::move(indices_ptr));
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > oracle,
        const tatami::Options& opt) 
    const {
        Index_ full_non_target = (row ? ncol_internal() : nrow_internal());
        return populate<true, tatami::OracularDenseExtractor, SparseMatrix_internal::DenseFull>(row, full_non_target, std::move(oracle), set_extract_all(opt), full_non_target);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return populate<true, tatami::OracularDenseExtractor, SparseMatrix_internal::DenseBlock>(row, block_length, std::move(oracle), set_extract_all(opt), block_start, block_length);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, tatami::OracularDenseExtractor, SparseMatrix_internal::DenseIndex>(row, nidx, std::move(oracle), set_extract_all(opt), std::move(indices_ptr));
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
        return populate<true, tatami::OracularSparseExtractor, SparseMatrix_internal::SparseFull>(row, full_non_target, std::move(oracle), opt, full_non_target);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return populate<true, tatami::OracularSparseExtractor, SparseMatrix_internal::SparseBlock>(row, block_length, std::move(oracle), opt, block_start, block_length);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto nidx = indices_ptr->size();
        return populate<true, tatami::OracularSparseExtractor, SparseMatrix_internal::SparseIndex>(row, nidx, std::move(oracle), opt, std::move(indices_ptr));
    }
};

}

#endif
