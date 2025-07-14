#ifndef TATAMI_CHUNKED_CUSTOM_SPARSE_CHUNKED_MATRIX_HPP
#define TATAMI_CHUNKED_CUSTOM_SPARSE_CHUNKED_MATRIX_HPP

#include "tatami/tatami.hpp"
#include "custom_internals.hpp"
#include "SparseSlabFactory.hpp"
#include "SlabCacheStats.hpp"
#include "LruSlabCache.hpp"
#include "OracularSlabCache.hpp"
#include "OracularSubsettedSlabCache.hpp"

#include <vector>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

/**
 * @file CustomSparseChunkedMatrix.hpp
 * @brief Custom sparse chunked matrix.
 */

namespace tatami_chunked {

/**
 * @brief Options for data extraction from a `CustomSparseChunkedMatrix`.
 */
struct CustomSparseChunkedMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored if `require_minimum_cache` is `true`.
     */
    std::size_t maximum_cache_size = sanisizer::cap<std::size_t>(100000000);

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that a single slab can be retained in memory,
     * so that the same chunks are not repeatedly re-read when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;

    /**
     * Whether to only extract and cache a subset of elements along the target dimension when an oracle is available.
     * This involves more overhead to determine which elements are needed but may improve access speed for chunking strategies that support partial extraction.
     */
    bool cache_subset = false;
};

/**
 * @brief Workspace for extracting data from a `CustomSparseChunkedMatrixManager`.
 * @tparam ChunkValue_ Numeric type of the data values in each chunk.
 * @tparam Index_ Integer type for the row/column indices of the `CustomSparseChunkedMatrix`.
 */
template<typename ChunkValue_, typename Index_>
class CustomSparseChunkedMatrixWorkspace {
public:
    /**
     * @cond
     */
    CustomSparseChunkedMatrixWorkspace() = default;
    CustomSparseChunkedMatrixWorkspace(const CustomSparseChunkedMatrixWorkspace&) = default;
    CustomSparseChunkedMatrixWorkspace(CustomSparseChunkedMatrixWorkspace&&) = default;
    CustomSparseChunkedMatrixWorkspace& operator=(const CustomSparseChunkedMatrixWorkspace&) = default;
    CustomSparseChunkedMatrixWorkspace& operator=(CustomSparseChunkedMatrixWorkspace&&) = default;
    virtual ~CustomSparseChunkedMatrixWorkspace() = default;
    /**
     * @endcond
     */

    /**
     * @param chunk_row_id Row of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param chunk_column_id Column of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param row Whether to extract rows from the chunk, i.e., the rows are the target dimension.
     * @param target_start Index of the first element on the target dimension to be extracted.
     * If `row = true`, this is the first row, otherwise it is the first column.
     * @param target_length Number of elements on the target dimension to be extracted.
     * If `row = true`, this is the number of rows, otherwise it is the number of columns.
     * This is guaranteed to be positive.
     * @param non_target_start Index of the start of the continguous block of the non-target dimension to be extracted.
     * If `row = true`, this is the first column, otherwise it is the first row.
     * @param non_target_length Length of the contiguous block of the non-target dimension to be extracted.
     * If `row = true`, this is the number of columns, otherwise it is the number of rows.
     * This is guaranteed to be positive.
     * @param[out] output_values Vector of pointers in which to store the values of non-zero elements.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no values should be stored.
     * @param[out] output_indices Vector of vectors in which to store the indices of the non-zero elements along the non-target dimension.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no indices should be stored.
     * @param[in,out] output_number Pointer to an array of length equal to the extent of the target dimension.
     * Each entry `i` specifies the number of non-zero elements that are already present in `output_values[i]` and `output_indices[i]`.
     * @param shift Shift to be added to the chunk's reported indices when storing them in `output_indices`.
     *
     * Given a chunk of interest, this method extracts a contiguous block of rows/columns.
     * If `row = true`, we consider a block of rows `[target_start, target_start + target_length) * and a block of columns `[non_target_start, non_target_start + non_target_length)`;
     * conversely, if `row = false`, we would consider a block of target columns and a block of non-target rows.
     * For a target dimension element `p`, the values of non-zero elements from the requested non-target block should be stored at `output_values[p] + output_number[p]`.
     * The non-target indices for those non-zero elements should be increased by `shift` and stored at `output_indices[p] + output_number[p]` in ascending order.
     * `output_number[p]` should then be increased by the number of non-zero entries stored in this manner.
     * 
     * - Storing the non-zero values from the `output_number[p]`-th element onwards (and shifting their indices) allows interleaving of data from multiple chunks.
     *   This ensures that the values/indices from the same target dimension element are contiguous for easier fetching in the `CustomSparseChunkedMatrix`.
     * - `p` should lie in `[target_start, target_start + target_length)`, not `[0, target_length)`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        Index_ target_start,
        Index_ target_length,
        Index_ non_target_start,
        Index_ non_target_length,
        const std::vector<ChunkValue_*>& output_values,
        const std::vector<Index_*>& output_indices,
        Index_* output_number,
        Index_ shift
    ) = 0;

    /**
     * @param chunk_row_id Row of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param chunk_column_id Column of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param row Whether to extract rows from the chunk, i.e., the rows are the target dimension.
     * @param target_start Index of the first element on the target dimension to be extracted.
     * If `row = true`, this is the first row, otherwise it is the first column.
     * @param target_length Number of elements on the target dimension to be extracted.
     * If `row = true`, this is the number of rows, otherwise it is the number of columns.
     * This is guaranteed to be positive.
     * @param non_target_indices Indexed subset of the non-target dimension to be extracted.
     * If `row = true`, these are column indices, otherwise these are row indices.
     * This is guaranteed to be non-empty with unique and sorted indices.
     * @param[out] output_values Vector of pointers in which to store the values of non-zero elements.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no values should be stored.
     * @param[out] output_indices Vector of vectors in which to store the indices of the non-zero elements along the non-target dimension.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no indices should be stored.
     * @param[in,out] output_number Pointer to an array of length equal to the extent of the target dimension.
     * Each entry `i` specifies the number of non-zero elements that are already present in `output_values[i]` and `output_indices[i]`.
     * @param shift Shift to be added to the chunk's reported indices when storing them in `output_indices`.
     *
     * Given a chunk of interest, this method extracts a contiguous block along the target dimension and an indexed subset along the non-target dimension.
     * If `row = true`, we consider a block of rows `[target_start, target_start + target_length)` and a subset of columns `non_target_indices`;
     * conversely, if `row = false`, we would extract data for all columns and a subset of rows.
     * For a target dimension element `p`, the values of non-zero elements from the requested non-target subset should be stored at `output_values[p] + output_number[p]`.
     * The non-target indices for those non-zero elements should be increased by `shift` and stored at `output_indices[p] + output_number[p]` in ascending order.
     * `output_number[p]` should then be increased by the number of non-zero entries stored in this manner.
     *
     * - Storing the non-zero values from the `output_number[p]`-th element onwards (and shifting their indices) allows interleaving of data from multiple chunks.
     *   This ensures that the values/indices from the same target dimension element are contiguous for easier fetching in the `CustomSparseChunkedMatrix`.
     * - `p` should lie in `[target_start, target_start + target_length)`, not `[0, target_length)`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        Index_ target_start,
        Index_ target_length,
        const std::vector<Index_>& non_target_indices,
        const std::vector<ChunkValue_*>& output_values,
        const std::vector<Index_*>& output_indices,
        Index_* output_number,
        Index_ shift
    ) = 0;

    /**
     * @param chunk_row_id Row of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param chunk_column_id Column of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param row Whether to extract rows from the chunk, i.e., the rows are the target dimension.
     * @param target_indices Indices of the elements of the target dimension to be extracted.
     * If `row = true`, these are row indices, otherwise these are column indices.
     * @param non_target_start Index of the start of the contiguous block of the non-target dimension to be extracted.
     * If `row = true`, this is the first column, otherwise it is the first row.
     * @param non_target_length Length of the contiguous block of the non-target dimension to be extracted.
     * If `row = true`, this is the number of columns, otherwise it is the number of rows.
     * This is guaranteed to be positive.
     * @param[out] output_values Vector of pointers in which to store the values of non-zero elements.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no values should be stored.
     * @param[out] output_indices Vector of vectors in which to store the indices of the non-zero elements along the non-target dimension.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no indices should be stored.
     * @param[in,out] output_number Pointer to an array of length equal to the extent of the target dimension.
     * Each entry `i` specifies the number of non-zero elements that are already present in `output_values[i]` and `output_indices[i]`.
     * @param shift Shift to be added to the chunk's reported indices when storing them in `output_indices`.
     *
     * Given a chunk of interest, this method extracts an indexed subset along the target dimension and a contiguous block along the non-target dimension.
     * If `row = true`, we consider a subset of rows `target_indices` and a block of columns `[non_target_start, non_target_start + non_target_length)`;
     * conversely, if `row = false`, we would extract a block of columns as the target and the block of rows as the non_target.
     * For a target dimension element `p`, the values of non-zero elements from the requested non-target block should be stored at `output_values[p] + output_number[p]`.
     * The non-target indices for those non-zero elements should be increased by `shift` and stored at `output_indices[p] + output_number[p]` in ascending order.
     * `output_number[p]` should then be increased by the number of non-zero entries stored in this manner.
     * 
     * - Storing the non-zero values from the `output_number[p]`-th element onwards (and shifting their indices) allows interleaving of data from multiple chunks.
     *   This ensures that the values/indices from the same target dimension element are contiguous for easier fetching in the `CustomSparseChunkedMatrix`.
     * - `p` should be a value in `target_indices`, not `[0, target_indices.size())`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        const std::vector<Index_>& target_indices,
        Index_ non_target_start,
        Index_ non_target_length,
        const std::vector<ChunkValue_*>& output_values,
        const std::vector<Index_*>& output_indices,
        Index_* output_number,
        Index_ shift
    ) = 0;

    /**
     * @param chunk_row_id Row of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param chunk_column_id Column of the chunk grid containing the chunk of interest.
     * This considers the grid of chunks that is obtained by partitioning each dimension of the matrix. 
     * @param row Whether to extract rows from the chunk, i.e., the rows are the target dimension.
     * @param target_indices Indices of the elements on the target dimension to be extracted.
     * If `row = true`, these are row indices, otherwise these are column indices.
     * This is guaranteed to be non-empty with unique and sorted indices.
     * @param non_target_indices Indices of the elements on the non-target dimension to be extracted.
     * If `row = true`, these are column indices, otherwise these are row indices.
     * This is guaranteed to be non-empty with unique and sorted indices.
     * @param[out] output_values Vector of pointers in which to store the values of non-zero elements.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no values should be stored.
     * @param[out] output_indices Vector of vectors in which to store the indices of the non-zero elements along the non-target dimension.
     * This has length equal to the extent of the target dimension for this chunk.
     * Each pointer corresponds to an element of the target dimension and refers to an array of length no less than the extent of the non-target dimension of the chunk.
     * Alternatively, this vector may be empty, in which case no indices should be stored.
     * @param[in,out] output_number Pointer to an array of length equal to the extent of the target dimension.
     * Each entry `i` specifies the number of non-zero elements that are already present in `output_values[i]` and `output_indices[i]`.
     * @param shift Shift to be added to the chunk's reported indices when storing them in `output_indices`.
     *
     * Given a chunk of interest, this method extracts data for an indexed subset of rows/columns.
     * If `row = true`, we would extract a subset of rows in `target_indices` and a subset columns in `non_target_indices`.
     * conversely, if `row = false`, we would consider a subset of target columns and a subset of non-target rows.
     * For a target dimension element `p`, the values of non-zero elements from the requested non-target subset should be stored at `output_values[p] + output_number[p]`.
     * The non-target indices for those non-zero elements should be increased by `shift` and stored at `output_indices[p] + output_number[p]` in ascending order.
     * `output_number[p]` should then be increased by the number of non-zero entries stored in this manner.
     *
     * - Storing the non-zero values from the `output_number[p]`-th element onwards (and shifting their indices) allows interleaving of data from multiple chunks.
     *   This ensures that the values/indices from the same target dimension element are contiguous for easier fetching in the `CustomSparseChunkedMatrix`.
     * - `p` should be a value in `target_indices`, not `[0, target_indices.size())`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        const std::vector<Index_>& target_indices,
        const std::vector<Index_>& non_target_indices,
        const std::vector<ChunkValue_*>& output_values,
        const std::vector<Index_*>& output_indices,
        Index_* output_number,
        Index_ shift
    ) = 0;
};

/**
 * @brief Manager of chunks for a `CustomSparseChunkedMatrix`.
 * @tparam ChunkValue_ Numeric type of the data values in each chunk.
 * @tparam Index_ Integer type of the row/column indices of the `CustomSparseChunkedMatrix`.
 */
template<typename ChunkValue_, typename Index_>
class CustomSparseChunkedMatrixManager {
public:
    /**
     * @cond
     */
    CustomSparseChunkedMatrixManager() = default;
    CustomSparseChunkedMatrixManager(const CustomSparseChunkedMatrixManager&) = default;
    CustomSparseChunkedMatrixManager(CustomSparseChunkedMatrixManager&&) = default;
    CustomSparseChunkedMatrixManager& operator=(const CustomSparseChunkedMatrixManager&) = default;
    CustomSparseChunkedMatrixManager& operator=(CustomSparseChunkedMatrixManager&&) = default;
    virtual ~CustomSparseChunkedMatrixManager() = default;
    /**
     * @endcond
     */

    /**
     * @return A `CustomSparseChunkedMatrixWorkspace` instance to unpack each chunk of interest.
     */
    virtual std::unique_ptr<CustomSparseChunkedMatrixWorkspace<ChunkValue_, Index_> > new_workspace() const = 0;

    /**
     * @return A `CustomSparseChunkedMatrixWorkspace` instance (or any of its subclasses) to unpack each chunk of interest.
     *
     * This is a non-virtual counterpart to `new_workspace()` that may optionally be shadowed by a method with the same signature in each subclass.
     * If implemented in a subclass, this should return a pointer to its specific `CustomSparseChunkedMatrixWorkspace` subclass, rather than to the base class. 
     * This provides some devirtualization opportunities within `CustomSparseChunkMatrix` when `Manager_` is hard-coded to a `CustomSparseChunkedMatrixManager` subclass.
     */
    std::unique_ptr<CustomSparseChunkedMatrixWorkspace<ChunkValue_, Index_> > new_workspace_exact() const {
        return new_workspace();
    }

    /**
     * @return Whether extraction of rows is the preferred access pattern.
     */
    virtual bool prefer_rows() const = 0;

    /**
     * @return Statistics for the rows, i.e., the number of rows in the matrix, the number of rows in each chunk, the number of chunks along all rows.
     * 
     * In all calls to `CustomSparseChunkedMatrixManager::extract()`, each `chunk_row_id` will be less than the `ChunkDimensionsStats::num_chunks` of the return value.
     */
    virtual const ChunkDimensionStats<Index_>& row_stats() const = 0;

    /**
     * @return Statistics for the columns, i.e., the number of columns in the matrix, the number of columns in each chunk, the number of chunks along all columns.
     * 
     * In all calls to `CustomSparseChunkedMatrixManager::extract()`, each `chunk_column_id` will be less than the `ChunkDimensionsStats::num_chunks` of the return value.
     */
    virtual const ChunkDimensionStats<Index_>& column_stats() const = 0;
};

/**
 * @cond
 */
namespace CustomChunkedMatrix_internal {

/*********************
 **** Base classes ***
 *********************/

template<bool oracle_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class SoloSparseCore {
    WorkspacePtr_ my_chunk_workspace;
    const ChunkCoordinator<true, ChunkValue_, Index_>& my_coordinator;

    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, tatami::PredictionIndex, bool>::type my_counter = 0;

    SparseSlabFactory<ChunkValue_, Index_, Index_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;

    // These two instances are not fully allocated Slabs; rather, tmp_solo just
    // holds the content for a single chunk, while final_solo holds the content
    // across chunks but only for the requested dimension element. Both cases
    // are likely to be much smaller than a full Slab, so we're already more
    // memory-efficient than 'require_minimum_cache = true'.
    SparseSingleWorkspace<ChunkValue_, Index_> my_tmp_solo;
    Slab my_final_solo;

public:
    SoloSparseCore(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        [[maybe_unused]] const SlabCacheStats<Index_>& slab_stats, // for consistency with the other base classes.
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_length,
        bool needs_value,
        bool needs_index
    ) :
        my_chunk_workspace(std::move(chunk_workspace)),
        my_coordinator(coordinator),
        my_oracle(std::move(oracle)),
        my_factory(1, non_target_length, 1, needs_value, needs_index),
        my_tmp_solo(
            my_coordinator.get_target_chunkdim(row),
            my_coordinator.get_non_target_chunkdim(row), 
            needs_value, 
            needs_index
        ),
        my_final_solo(my_factory.create())
    {}

    template<typename ... Args_>
    std::pair<const Slab*, Index_> fetch_raw(Index_ i, bool row, Args_&& ... args) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        return my_coordinator.fetch_single(row, i, std::forward<Args_>(args)..., *my_chunk_workspace, my_tmp_solo, my_final_solo);
    }
};

template<typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class MyopicSparseCore {
    WorkspacePtr_ my_chunk_workspace;
    const ChunkCoordinator<true, ChunkValue_, Index_>& my_coordinator;

    SparseSlabFactory<ChunkValue_, Index_, Index_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;

    LruSlabCache<Index_, Slab> my_cache;

public:
    MyopicSparseCore(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator,
        const SlabCacheStats<Index_>& slab_stats, 
        bool row,
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> oracle, // for consistency with the other base classes
        Index_ non_target_length,
        bool needs_value,
        bool needs_index
    ) : 
        my_chunk_workspace(std::move(chunk_workspace)),
        my_coordinator(coordinator),
        my_factory(coordinator.get_target_chunkdim(row), non_target_length, slab_stats, needs_value, needs_index),
        my_cache(slab_stats.max_slabs_in_cache) 
    {}

    template<typename ... Args_>
    std::pair<const Slab*, Index_> fetch_raw(Index_ i, bool row, Args_&& ... args) {
        return my_coordinator.fetch_myopic(row, i, std::forward<Args_>(args)..., *my_chunk_workspace, my_cache, my_factory);
    }
};

template<bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class OracularSparseCore {
protected:
    WorkspacePtr_ my_chunk_workspace;
    const ChunkCoordinator<true, ChunkValue_, Index_>& my_coordinator;

    SparseSlabFactory<ChunkValue_, Index_, Index_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;

    typename std::conditional<use_subset_, OracularSubsettedSlabCache<Index_, Index_, Slab>, OracularSlabCache<Index_, Index_, Slab> >::type my_cache;

public:
    OracularSparseCore(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator,
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<true, Index_> oracle,
        Index_ non_target_length,
        bool needs_value,
        bool needs_index
    ) : 
        my_chunk_workspace(std::move(chunk_workspace)),
        my_coordinator(coordinator), 
        my_factory(coordinator.get_target_chunkdim(row), non_target_length, slab_stats, needs_value, needs_index),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache) 
    {}

    template<typename ... Args_>
    std::pair<const Slab*, Index_> fetch_raw([[maybe_unused]] Index_ i, bool row, Args_&& ... args) {
        if constexpr(use_subset_) {
            return my_coordinator.fetch_oracular_subsetted(row, std::forward<Args_>(args)..., *my_chunk_workspace, my_cache, my_factory);
        } else {
            return my_coordinator.fetch_oracular(row, std::forward<Args_>(args)..., *my_chunk_workspace, my_cache, my_factory);
        }
    }
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
using SparseCore = typename std::conditional<solo_, 
      SoloSparseCore<oracle_, Value_, Index_, ChunkValue_, WorkspacePtr_>,
      typename std::conditional<oracle_,
          OracularSparseCore<use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_>,
          MyopicSparseCore<Value_, Index_, ChunkValue_, WorkspacePtr_>
      >::type
>::type;

/***********************
 **** Sparse classes ***
 ***********************/

template<class Slab_, typename Index_, typename Value_>
tatami::SparseRange<Value_, Index_> process_sparse_slab(const std::pair<const Slab_*, Index_>& fetched, Value_* value_buffer, Index_* index_buffer, bool needs_value, bool needs_index) {
    auto num = fetched.first->number[fetched.second];

    if (needs_value) {
        auto vptr = fetched.first->values[fetched.second];
        std::copy_n(vptr, num, value_buffer);
    } else {
        value_buffer = NULL;
    }

    if (needs_index) {
        auto iptr = fetched.first->indices[fetched.second];
        std::copy_n(iptr, num, index_buffer);
    } else {
        index_buffer = NULL;
    }

    return tatami::SparseRange<Value_, Index_>(num, value_buffer, index_buffer);
}

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class SparseFull : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseFull(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        const tatami::Options& opt
    ) :
        my_row(row),
        my_non_target_dim(coordinator.get_non_target_dim(row)),
        my_needs_value(opt.sparse_extract_value),
        my_needs_index(opt.sparse_extract_index),
        my_core(
            std::move(chunk_workspace),
            coordinator, 
            slab_stats,
            row,
            std::move(oracle), 
            my_non_target_dim,
            opt.sparse_extract_value,
            opt.sparse_extract_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto fetched = my_core.fetch_raw(i, my_row, 0, my_non_target_dim);
        return process_sparse_slab(fetched, value_buffer, index_buffer, my_needs_value, my_needs_index); 
    }

private:
    bool my_row;
    Index_ my_non_target_dim;
    bool my_needs_value, my_needs_index;
    SparseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class SparseBlock : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseBlock(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt
    ) :
        my_row(row),
        my_block_start(block_start),
        my_block_length(block_length),
        my_needs_value(opt.sparse_extract_value),
        my_needs_index(opt.sparse_extract_index),
        my_core(
            std::move(chunk_workspace),
            coordinator,
            slab_stats,
            row,
            std::move(oracle),
            block_length,
            my_needs_value,
            my_needs_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto fetched = my_core.fetch_raw(i, my_row, my_block_start, my_block_length);
        return process_sparse_slab(fetched, value_buffer, index_buffer, my_needs_value, my_needs_index);
    }

private:
    bool my_row;
    Index_ my_block_start, my_block_length;
    bool my_needs_value, my_needs_index;
    SparseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class SparseIndex : public tatami::SparseExtractor<oracle_, Value_, Index_> {
public:
    SparseIndex(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt
    ) :
        my_row(row),
        my_indices_ptr(std::move(indices_ptr)),
        my_needs_value(opt.sparse_extract_value),
        my_needs_index(opt.sparse_extract_index),
        my_core(
            std::move(chunk_workspace),
            coordinator, 
            slab_stats,
            row,
            std::move(oracle), 
            my_indices_ptr->size(),
            my_needs_value,
            my_needs_index
        )
    {}

    tatami::SparseRange<Value_, Index_> fetch(Index_ i, Value_* value_buffer, Index_* index_buffer) {
        auto fetched = my_core.fetch_raw(i, my_row, *my_indices_ptr, my_tmp_indices);
        return process_sparse_slab(fetched, value_buffer, index_buffer, my_needs_value, my_needs_index);
    }

private:
    bool my_row;
    tatami::VectorPtr<Index_> my_indices_ptr;
    std::vector<Index_> my_tmp_indices;
    bool my_needs_value, my_needs_index;
    SparseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

/**************************
 **** Densified classes ***
 **************************/

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class DensifiedFull : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DensifiedFull(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        const tatami::Options&
    ) :
        my_row(row),
        my_non_target_dim(coordinator.get_non_target_dim(row)),
        my_core(
            std::move(chunk_workspace),
            coordinator,
            slab_stats,
            row,
            std::move(oracle), 
            my_non_target_dim,
            true, 
            true
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto contents = my_core.fetch_raw(i, my_row, 0, my_non_target_dim);

        Index_ num = contents.first->number[contents.second];
        auto vptr = contents.first->values[contents.second];
        auto iptr = contents.first->indices[contents.second];

        std::fill_n(buffer, my_non_target_dim, 0);
        for (Index_ x = 0; x < num; ++x, ++iptr, ++vptr) {
            buffer[*iptr] = *vptr;
        }
        return buffer;
    }

private:
    bool my_row;
    Index_ my_non_target_dim;
    SparseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class DensifiedBlock : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DensifiedBlock(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ block_start, 
        Index_ block_length,
        const tatami::Options&
    ) :
        my_row(row),
        my_block_start(block_start),
        my_block_length(block_length),
        my_core(
            std::move(chunk_workspace),
            coordinator,
            slab_stats,
            row,
            std::move(oracle), 
            block_length,
            true,
            true
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto contents = my_core.fetch_raw(i, my_row, my_block_start, my_block_length);

        auto vptr = contents.first->values[contents.second];
        auto iptr = contents.first->indices[contents.second];
        auto num = contents.first->number[contents.second];

        std::fill_n(buffer, my_block_length, 0);
        for (Index_ x = 0; x < num; ++x, ++iptr, ++vptr) {
            buffer[*iptr - my_block_start] = *vptr;
        }
        return buffer;
    }

private:
    bool my_row;
    Index_ my_block_start, my_block_length;
    SparseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class DensifiedIndex : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DensifiedIndex(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<true, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options&
    ) :
        my_row(row),
        my_indices_ptr(std::move(indices_ptr)),
        my_core(
            std::move(chunk_workspace),
            coordinator, 
            slab_stats,
            row,
            std::move(oracle), 
            my_indices_ptr->size(),
            true,
            true
        )
    {
        const auto& indices = *my_indices_ptr;
        if (!indices.empty()) {
            my_remap_offset = indices.front();
            Index_ alloc = indices.back() - my_remap_offset + 1; // alloc must be <= dim extent, which should fit in an Index_.
            tatami::resize_container_to_Index_size(my_remap, alloc);
            Index_ counter = 0;
            for (auto i : indices) {
                my_remap[i - my_remap_offset] = counter;
                ++counter;
            }
        }
    }

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto contents = my_core.fetch_raw(i, my_row, *my_indices_ptr, my_tmp_indices);

        auto vptr = contents.first->values[contents.second];
        auto iptr = contents.first->indices[contents.second];
        auto num = contents.first->number[contents.second];

        auto nidx = my_indices_ptr->size();
        std::fill_n(buffer, nidx, 0);
        for (Index_ x = 0; x <num; ++x, ++iptr, ++vptr) {
            buffer[my_remap[*iptr - my_remap_offset]] = *vptr;
        }
        return buffer;
    }

private:
    bool my_row;
    tatami::VectorPtr<Index_> my_indices_ptr;
    Index_ my_remap_offset = 0;
    std::vector<Index_> my_remap;
    std::vector<Index_> my_tmp_indices;
    SparseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

}
/**
 * @endcond
 */

/**
 * @brief Matrix of custom sparse chunks.
 *
 * @tparam Value_ Numeric type for the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam ChunkValue_ Numeric type of the values in each chunk.
 * @tparam Manager_ Class that implements the `CustomSparseChunkedMatrixManager` interface.
 *
 * Implements a `tatami::Matrix` subclass where data is contained in sparse rectangular chunks.
 * These chunks are typically compressed in some manner to reduce memory usage compared to, e.g., a `tatami::CompressedSparseMatrix`.
 * On access, the relevant chunks are decompressed and the desired values are extracted.
 * Each dimension should be divided into chunk boundaries at regular intervals starting from zero;
 * this partitions the matrix according to a regular grid where each grid entry is a single chunk of the same size.
 * The exception is for chunks at the non-zero boundaries of the matrix dimensions, which may be truncated.
 */
template<typename Value_, typename Index_, typename ChunkValue_, class Manager_ = CustomSparseChunkedMatrixManager<ChunkValue_, Index_> >
class CustomSparseChunkedMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param manager Pointer to a `CustomSparseChunkedMatrixManager` instance that manages the chunks for this matrix.
     * @param opt Further options for chunked extraction.
     */
    CustomSparseChunkedMatrix(std::shared_ptr<Manager_> manager, const CustomSparseChunkedMatrixOptions& opt) : 
        my_manager(std::move(manager)),
        my_coordinator(my_manager->row_stats(), my_manager->column_stats()),
        my_cache_size_in_bytes(opt.maximum_cache_size),
        my_require_minimum_cache(opt.require_minimum_cache),
        my_cache_subset(opt.cache_subset)
    {}

private:
    std::shared_ptr<Manager_> my_manager;
    CustomChunkedMatrix_internal::ChunkCoordinator<true, ChunkValue_, Index_> my_coordinator;
    std::size_t my_cache_size_in_bytes;
    bool my_require_minimum_cache;
    bool my_cache_subset;

public:
    Index_ nrow() const { 
        return my_coordinator.get_nrow();
    }

    Index_ ncol() const { 
        return my_coordinator.get_ncol();
    }

    bool prefer_rows() const { 
        return my_manager->prefer_rows();
    }

    bool uses_oracle(bool) const { 
        return true; 
    }

    double prefer_rows_proportion() const { 
        return static_cast<double>(my_manager->prefer_rows());
    }

    bool is_sparse() const {
        return true;
    }

    double is_sparse_proportion() const {
        return 1;
    }

    using tatami::Matrix<Value_, Index_>::dense;

    using tatami::Matrix<Value_, Index_>::sparse;

    /********************
     *** Myopic dense ***
     ********************/
private:
    template<
        template<bool, typename, typename> class Interface_, 
        bool oracle_, 
        template<bool, bool, bool, typename, typename, typename, class> class Extractor_,
        typename ... Args_
    >
    std::unique_ptr<Interface_<oracle_, Value_, Index_> > raw_internal(bool row, Index_ non_target_length, const tatami::Options& opt, Args_&& ... args) const {
        std::size_t element_size = (opt.sparse_extract_value ? sizeof(ChunkValue_) : 0) + (opt.sparse_extract_index ? sizeof(Index_) : 0);
        auto stats = [&]{
            if (row) {
                // Remember, the num_chunks_per_column is the number of slabs needed to divide up all the *rows* of the matrix.
                return SlabCacheStats<Index_>(
                    my_coordinator.get_chunk_nrow(),
                    non_target_length,
                    my_coordinator.get_num_chunks_per_column(),
                    my_cache_size_in_bytes,
                    element_size,
                    my_require_minimum_cache
                );
            } else {
                // Remember, the num_chunks_per_row is the number of slabs needed to divide up all the *columns* of the matrix.
                return SlabCacheStats<Index_>(
                    my_coordinator.get_chunk_ncol(),
                    non_target_length,
                    my_coordinator.get_num_chunks_per_row(),
                    my_cache_size_in_bytes,
                    element_size,
                    my_require_minimum_cache
                );
            }
        }();

        auto wrk = my_manager->new_workspace_exact();
        if (stats.max_slabs_in_cache == 0) {
            return std::make_unique<Extractor_<true, oracle_, false, Value_, Index_, ChunkValue_, decltype(wrk)> >(std::move(wrk), my_coordinator, stats, row, std::forward<Args_>(args)...);
        } else if constexpr(oracle_) {
            if (my_cache_subset) {
                return std::make_unique<Extractor_<false, true, true, Value_, Index_, ChunkValue_, decltype(wrk)> >(std::move(wrk), my_coordinator, stats, row, std::forward<Args_>(args)...);
            } else {
                return std::make_unique<Extractor_<false, true, false, Value_, Index_, ChunkValue_, decltype(wrk)> >(std::move(wrk), my_coordinator, stats, row, std::forward<Args_>(args)...);
            }
        } else {
            return std::make_unique<Extractor_<false, false, false, Value_, Index_, ChunkValue_, decltype(wrk)> >(std::move(wrk), my_coordinator, stats, row, std::forward<Args_>(args)...);
        }
    }

public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        return raw_internal<tatami::DenseExtractor, false, CustomChunkedMatrix_internal::DensifiedFull>(row, my_coordinator.get_non_target_dim(row), opt, false, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return raw_internal<tatami::DenseExtractor, false, CustomChunkedMatrix_internal::DensifiedBlock>(row, block_length, opt, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto num_indices = indices_ptr->size();
        return raw_internal<tatami::DenseExtractor, false, CustomChunkedMatrix_internal::DensifiedIndex>(row, num_indices, opt, false, std::move(indices_ptr), opt);
    }

    /***********************
     *** Oracular dense ***
     ***********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        const tatami::Options& opt) 
    const {
        return raw_internal<tatami::DenseExtractor, true, CustomChunkedMatrix_internal::DensifiedFull>(row, my_coordinator.get_non_target_dim(row), opt, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return raw_internal<tatami::DenseExtractor, true, CustomChunkedMatrix_internal::DensifiedBlock>(row, block_length, opt, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto num_indices = indices_ptr->size();
        return raw_internal<tatami::DenseExtractor, true, CustomChunkedMatrix_internal::DensifiedIndex>(row, num_indices, opt, std::move(oracle), std::move(indices_ptr), opt);
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        return raw_internal<tatami::SparseExtractor, false, CustomChunkedMatrix_internal::SparseFull>(row, my_coordinator.get_non_target_dim(row), opt, false, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return raw_internal<tatami::SparseExtractor, false, CustomChunkedMatrix_internal::SparseBlock>(row, block_length, opt, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto num_indices = indices_ptr->size();
        return raw_internal<tatami::SparseExtractor, false, CustomChunkedMatrix_internal::SparseIndex>(row, num_indices, opt, false, std::move(indices_ptr), opt);
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
        return raw_internal<tatami::SparseExtractor, true, CustomChunkedMatrix_internal::SparseFull>(row, my_coordinator.get_non_target_dim(row), opt, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return raw_internal<tatami::SparseExtractor, true, CustomChunkedMatrix_internal::SparseBlock>(row, block_length, opt, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        auto num_indices = indices_ptr->size();
        return raw_internal<tatami::SparseExtractor, true, CustomChunkedMatrix_internal::SparseIndex>(row, num_indices, opt, std::move(oracle), std::move(indices_ptr), opt);
    }
};

}

#endif
