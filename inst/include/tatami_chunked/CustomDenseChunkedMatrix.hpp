#ifndef TATAMI_CHUNKED_CUSTOM_DENSE_CHUNKED_MATRIX_HPP
#define TATAMI_CHUNKED_CUSTOM_DENSE_CHUNKED_MATRIX_HPP

#include "tatami/tatami.hpp"
#include "custom_internals.hpp"
#include "SlabCacheStats.hpp"
#include "DenseSlabFactory.hpp"
#include "LruSlabCache.hpp"
#include "OracularSlabCache.hpp"
#include "OracularSubsettedSlabCache.hpp"

#include <type_traits>
#include <vector>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

/**
 * @file CustomDenseChunkedMatrix.hpp
 * @brief Custom dense chunked matrix.
 */

namespace tatami_chunked {

/**
 * @brief Options for data extraction from a `CustomDenseChunkedMatrix`.
 */
struct CustomDenseChunkedMatrixOptions {
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
 * @brief Workspace for extracting data from a `CustomDenseChunkedMatrixManager`.
 * @tparam ChunkValue_ Numeric type of the data values in each chunk.
 * @tparam Index_ Integer type of the row/column indices of the `CustomDenseChunkedMatrix`.
 */
template<typename ChunkValue_, typename Index_>
class CustomDenseChunkedMatrixWorkspace {
public:
    /**
     * @cond
     */
    CustomDenseChunkedMatrixWorkspace() = default;
    CustomDenseChunkedMatrixWorkspace(const CustomDenseChunkedMatrixWorkspace&) = default;
    CustomDenseChunkedMatrixWorkspace(CustomDenseChunkedMatrixWorkspace&&) = default;
    CustomDenseChunkedMatrixWorkspace& operator=(const CustomDenseChunkedMatrixWorkspace&) = default;
    CustomDenseChunkedMatrixWorkspace& operator=(CustomDenseChunkedMatrixWorkspace&&) = default;
    virtual ~CustomDenseChunkedMatrixWorkspace() = default;
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
     * @param[out] output Pointer to an output array of length no less than `stride * P`,
     * where `P` is the number of rows (if `row = true`) or columns (otherwise) in this chunk.
     * @param stride Distance between corresponding values from adjacent elements of the target dimension when they are being stored in `output`.
     * This is guaranteed to be greater than or equal to `non_target_length`.
     *
     * Given a chunk of interest, this method extracts a contiguous block of rows/columns.
     * If `row = true`, we consider a block of rows `[target_start, target_start + target_length) * and a block of columns `[non_target_start, non_target_start + non_target_length)`;
     * conversely, if `row = false`, we would consider a block of target columns and a block of non-target rows.
     * For a target dimension index `p` and non-target dimension index `non_target_start + q`, the value from the chunk should be stored in `output[p * stride + q]`.
     *
     * - The `stride` allows interleaving of multiple chunks into a single array where values from the same target dimension element are contiguous.
     *   This enables easier fetching in the `CustomDenseChunkedMatrix`.
     * - `p` should lie in `[target_start, target_start + target_length)`, whereas `q` should lie in `[0, non_target_length)`.
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
        ChunkValue_* output,
        Index_ stride
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
     * @param[out] output Pointer to an output array of length no less than `stride * P`,
     * where `P` is the number of rows (if `row = true`) or columns (otherwise) in this chunk.
     * @param stride Distance between corresponding values from consecutive target dimension elements when stored in `output`.
     * This is guaranteed to be greater than or equal to `non_target_indices.size()`.
     *
     * Given a chunk of interest, this method extracts a contiguous block along the target dimension and an indexed subset along the non-target dimension.
     * If `row = true`, we consider a block of rows `[target_start, target_start + target_length)` and a subset of columns `non_target_indices`;
     * conversely, if `row = false`, we would extract data for all columns and a subset of rows.
     * For a target dimension index `p` and non-target dimension index `non_target_indices[q]`, the value from the chunk should be stored in `output[p * stride + q]`.
     *
     * - The `stride` allows interleaving of multiple chunks into a single array where values from the same target dimension element are contiguous.
     *   This enables easier fetching in the `CustomDenseChunkedMatrix`.
     * - `p` should lie in `[target_start, target_start + target_length)`, whereas `q` should lie in `[0, non_target_indices.size())`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        Index_ target_start,
        Index_ target_length,
        const std::vector<Index_>& non_target_indices,
        ChunkValue_* output,
        Index_ stride
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
     * @param[out] output Pointer to an output array of length no less than `stride * target_length + non_target_length`.
     * @param stride Distance between corresponding values from consecutive target dimension elements when stored in `output`.
     * This is guaranteed to be greater than or equal to `non_target_length`.
     *
     * Given a chunk of interest, this method extracts an indexed subset along the target dimension and a contiguous block along the non-target dimension.
     * If `row = true`, we consider a subset of rows `target_indices` and a block of columns `[non_target_start, non_target_start + non_target_length)`;
     * conversely, if `row = false`, we would extract a block of columns as the target and the block of rows as the non_target.
     * For a target dimension index `p` and non-target dimension index `non_target_start + q`, the value from the chunk should be stored in `output[p * stride + q]`.
     *
     * - The `stride` allows interleaving of multiple chunks into a single array where values from the same target dimension element are contiguous.
     *   This enables easier fetching in the `CustomDenseChunkedMatrix`.
     * - `p` should be a value in `target_indices`, whereas `q` should lie in `[0, non_target_length)`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        const std::vector<Index_>& target_indices,
        Index_ non_target_start,
        Index_ non_target_length,
        ChunkValue_* output,
        Index_ stride
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
     * @param[out] output Pointer to an output array of length no less than `stride * (target_indices.back() + 1)`.
     * @param stride Distance between corresponding values from consecutive target dimension elements when stored in `output`.
     * This is guaranteed to be greater than or equal to `non_target_indices.size()`.
     *
     * Given a chunk of interest, this method extracts data for an indexed subset of rows/columns.
     * If `row = true`, we would extract a subset of rows in `target_indices` and a subset columns in `non_target_indices`.
     * conversely, if `row = false`, we would consider a subset of target columns and a subset of non-target rows.
     * For a target dimension index `p` and non-target dimension index `non_target_indices[q]`, the value from the chunk should be stored in `output[p * stride + q]`.
     *
     * - The `stride` allows interleaving of multiple chunks into a single array where values from the same target dimension element are contiguous.
     *   This enables easier fetching in the `CustomDenseChunkedMatrix`.
     * - `p` should be a value in `target_indices` whereas `q` should lie in `[0, non_target_indices.size())`.
     *   This difference is deliberate and enables easy extraction of the target dimension element of interest.
     */
    virtual void extract(
        Index_ chunk_row_id,
        Index_ chunk_column_id,
        bool row,
        const std::vector<Index_>& target_indices,
        const std::vector<Index_>& non_target_indices,
        ChunkValue_* output,
        Index_ stride
    ) = 0;
};

/**
 * @brief Manager of chunks for a `CustomDenseChunkedMatrix`.
 * @tparam ChunkValue_ Numeric type of the data values in each chunk.
 * @tparam Index_ Integer type for the row/column indices of the `CustomDenseChunkedMatrix`.
 */
template<typename ChunkValue_, typename Index_>
class CustomDenseChunkedMatrixManager {
public:
    /**
     * @cond
     */
    CustomDenseChunkedMatrixManager() = default;
    CustomDenseChunkedMatrixManager(const CustomDenseChunkedMatrixManager&) = default;
    CustomDenseChunkedMatrixManager(CustomDenseChunkedMatrixManager&&) = default;
    CustomDenseChunkedMatrixManager& operator=(const CustomDenseChunkedMatrixManager&) = default;
    CustomDenseChunkedMatrixManager& operator=(CustomDenseChunkedMatrixManager&&) = default;
    virtual ~CustomDenseChunkedMatrixManager() = default;
    /**
     * @endcond
     */

    /**
     * @return A `CustomDenseChunkedMatrixWorkspace` instance to unpack each chunk of interest.
     */
    virtual std::unique_ptr<CustomDenseChunkedMatrixWorkspace<ChunkValue_, Index_> > new_workspace() const = 0;

    /**
     * @return A `CustomDenseChunkedMatrixWorkspace` instance (or any of its subclasses) to unpack each chunk of interest.
     *
     * This is a non-virtual counterpart to `new_workspace()` that may optionally be shadowed by a method with the same signature in each subclass.
     * If implemented in a subclass, this should return a pointer to its specific `CustomDenseChunkedMatrixWorkspace` subclass, rather than to the base class. 
     * This provides some devirtualization opportunities within `CustomDenseChunkMatrix` when `Manager_` is hard-coded to a specific `CustomDenseChunkedMatrixManager` subclass.
     */
    std::unique_ptr<CustomDenseChunkedMatrixWorkspace<ChunkValue_, Index_> > new_workspace_exact() const {
        return new_workspace();
    }

    /**
     * @return Whether extraction of rows is the preferred access pattern.
     */
    virtual bool prefer_rows() const = 0;

    /**
     * @return Statistics for the rows, i.e., the number of rows in the matrix, the number of rows in each chunk, the number of chunks along all rows.
     * 
     * In all calls to `CustomDenseChunkedMatrixManager::extract()`, each `chunk_row_id` will be less than the `ChunkDimensionsStats::num_chunks` of the return value.
     */
    virtual const ChunkDimensionStats<Index_>& row_stats() const = 0;

    /**
     * @return Statistics for the columns, i.e., the number of columns in the matrix, the number of columns in each chunk, the number of chunks along all columns.
     * 
     * In all calls to `CustomDenseChunkedMatrixManager::extract()`, each `chunk_column_id` will be less than the `ChunkDimensionsStats::num_chunks` of the return value.
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
class SoloDenseCore {
private:
    WorkspacePtr_ my_chunk_workspace;
    const ChunkCoordinator<false, ChunkValue_, Index_>& my_coordinator;

    tatami::MaybeOracle<oracle_, Index_> my_oracle;
    typename std::conditional<oracle_, tatami::PredictionIndex, bool>::type my_counter = 0;

    DenseSlabFactory<ChunkValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;

    // These two instances are not fully allocated Slabs; rather, tmp_solo just
    // holds the content for a single chunk, while final_solo holds the content
    // across chunks but only for the requested dimension element. Both cases
    // are likely to be much smaller than a full Slab, so we're already more
    // memory-efficient than 'require_minimum_cache = true`. 
    DenseSingleWorkspace<ChunkValue_> my_tmp_solo;
    Slab my_final_solo;

    typedef decltype(my_tmp_solo.size()) TmpSize;

public:
    SoloDenseCore(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<false, ChunkValue_, Index_>& coordinator, 
        [[maybe_unused]] const SlabCacheStats<Index_>& slab_stats, // for consistency with the other base classes.
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Index_ non_target_length
    ) :
        my_chunk_workspace(std::move(chunk_workspace)),
        my_coordinator(coordinator),
        my_oracle(std::move(oracle)),
        my_factory(non_target_length, 1),
        my_tmp_solo(static_cast<TmpSize>(my_coordinator.get_chunk_nrow()) * static_cast<TmpSize>(my_coordinator.get_chunk_ncol())),
        my_final_solo(my_factory.create())
    {}

    template<typename ... Args_>
    std::pair<const Slab*, Index_> fetch_raw(bool row, Index_ i, Args_&& ... args) {
        if constexpr(oracle_) {
            i = my_oracle->get(my_counter++);
        }
        return my_coordinator.fetch_single(row, i, std::forward<Args_>(args)..., *my_chunk_workspace, my_tmp_solo, my_final_solo);
    }
};

template<typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class MyopicDenseCore {
private:
    WorkspacePtr_ my_chunk_workspace;
    const ChunkCoordinator<false, ChunkValue_, Index_>& my_coordinator;

    DenseSlabFactory<ChunkValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;

    LruSlabCache<Index_, Slab> my_cache;

public:
    MyopicDenseCore(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<false, ChunkValue_, Index_>& coordinator,
        const SlabCacheStats<Index_>& slab_stats, 
        [[maybe_unused]] tatami::MaybeOracle<false, Index_> ora, // for consistency with the other base classes
        [[maybe_unused]] Index_ non_target_length
    ) :
        my_chunk_workspace(std::move(chunk_workspace)),
        my_coordinator(coordinator),
        my_factory(slab_stats),
        my_cache(slab_stats.max_slabs_in_cache)
    {}

    template<typename ... Args_>
    std::pair<const Slab*, Index_> fetch_raw(bool row, Index_ i, Args_&& ... args) {
        return my_coordinator.fetch_myopic(row, i, std::forward<Args_>(args)..., *my_chunk_workspace, my_cache, my_factory);
    }
};

template<bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class OracularDenseCore {
private:
    WorkspacePtr_ my_chunk_workspace;
    const ChunkCoordinator<false, ChunkValue_, Index_>& my_coordinator;

    DenseSlabFactory<ChunkValue_> my_factory;
    typedef typename decltype(my_factory)::Slab Slab;

    typename std::conditional<use_subset_, OracularSubsettedSlabCache<Index_, Index_, Slab>, OracularSlabCache<Index_, Index_, Slab> >::type my_cache;

public:
    OracularDenseCore(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<false, ChunkValue_, Index_>& coordinator,
        const SlabCacheStats<Index_>& slab_stats,
        tatami::MaybeOracle<true, Index_> oracle, 
        [[maybe_unused]] Index_ non_target_length
    ) :
        my_chunk_workspace(std::move(chunk_workspace)),
        my_coordinator(coordinator), 
        my_factory(slab_stats),
        my_cache(std::move(oracle), slab_stats.max_slabs_in_cache)
    {}

    template<typename ... Args_>
    std::pair<const Slab*, Index_> fetch_raw(bool row, [[maybe_unused]] Index_ i, Args_&& ... args) {
        if constexpr(use_subset_) {
            return my_coordinator.fetch_oracular_subsetted(row, std::forward<Args_>(args)..., *my_chunk_workspace, my_cache, my_factory);
        } else {
            return my_coordinator.fetch_oracular(row, std::forward<Args_>(args)..., *my_chunk_workspace, my_cache, my_factory);
        }
    }
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
using DenseCore = typename std::conditional<solo_, 
      SoloDenseCore<oracle_, Value_, Index_, ChunkValue_, WorkspacePtr_>,
      typename std::conditional<oracle_,
          OracularDenseCore<use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_>,
          MyopicDenseCore<Value_, Index_, ChunkValue_, WorkspacePtr_>
      >::type
>::type;

/***********************
 **** Actual classes ***
 ***********************/

template<class Slab_, typename Index_, typename Value_>
const Value_* process_dense_slab(const std::pair<const Slab_*, Index_>& fetched, Value_* buffer, Index_ non_target_length) {
    auto ptr = fetched.first->data + static_cast<std::size_t>(fetched.second) * static_cast<std::size_t>(non_target_length); // cast to size_t to avoid overflow.
    std::copy_n(ptr, non_target_length, buffer);
    return buffer;
}

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class DenseFull : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseFull(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<false, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle
    ) :
        my_row(row),
        my_non_target_dim(coordinator.get_non_target_dim(row)),
        my_core(
            std::move(chunk_workspace),
            coordinator,
            slab_stats,
            std::move(oracle),
            my_non_target_dim
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto fetched = my_core.fetch_raw(my_row, i, 0, my_non_target_dim);
        return process_dense_slab(fetched, buffer, my_non_target_dim);
    }

private:
    bool my_row;
    Index_ my_non_target_dim;
    DenseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class DenseBlock : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseBlock(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<false, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> ora, 
        Index_ block_start, 
        Index_ block_length
    ) :
        my_row(row),
        my_block_start(block_start),
        my_block_length(block_length),
        my_core(
            std::move(chunk_workspace),
            coordinator, 
            slab_stats,
            std::move(ora),
            block_length
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto fetched = my_core.fetch_raw(my_row, i, my_block_start, my_block_length);
        return process_dense_slab(fetched, buffer, my_block_length);
    }

private:
    bool my_row;
    Index_ my_block_start, my_block_length;
    DenseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

template<bool solo_, bool oracle_, bool use_subset_, typename Value_, typename Index_, typename ChunkValue_, class WorkspacePtr_>
class DenseIndex : public tatami::DenseExtractor<oracle_, Value_, Index_> {
public:
    DenseIndex(
        WorkspacePtr_ chunk_workspace,
        const ChunkCoordinator<false, ChunkValue_, Index_>& coordinator, 
        const SlabCacheStats<Index_>& slab_stats,
        bool row,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        tatami::VectorPtr<Index_> indices_ptr) :
        my_row(row),
        my_indices_ptr(std::move(indices_ptr)),
        my_core(
            std::move(chunk_workspace),
            coordinator, 
            slab_stats,
            std::move(oracle),
            my_indices_ptr->size()
        )
    {}

    const Value_* fetch(Index_ i, Value_* buffer) {
        auto fetched = my_core.fetch_raw(my_row, i, *my_indices_ptr, my_tmp_indices);
        return process_dense_slab(fetched, buffer, static_cast<Index_>(my_indices_ptr->size()));
    }

private:
    bool my_row;
    tatami::VectorPtr<Index_> my_indices_ptr;
    std::vector<Index_> my_tmp_indices;
    DenseCore<solo_, oracle_, use_subset_, Value_, Index_, ChunkValue_, WorkspacePtr_> my_core;
};

}
/**
 * @endcond
 */

/**
 * @brief Matrix of custom dense chunks.
 *
 * @tparam Value_ Numeric type for the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam ChunkValue_ Numeric type of the values in each chunk.
 * @tparam Manager_ Class that implements the `CustomDenseChunkedMatrixManager` interface.
 *
 * Implements a `tatami::Matrix` subclass where data is contained in dense rectangular chunks.
 * These chunks are typically compressed in some manner to reduce memory usage compared to, e.g., a `tatami::DenseMatrix`.
 * On access, the relevant chunks are decompressed and the desired values are extracted.
 * Each dimension should be divided into chunk boundaries at regular intervals starting from zero;
 * this partitions the matrix according to a regular grid where each grid entry is a single chunk of the same size.
 * The exception is for chunks at the non-zero boundaries of the matrix dimensions, which may be truncated.
 */
template<typename Value_, typename Index_, typename ChunkValue_, class Manager_ = CustomDenseChunkedMatrixManager<ChunkValue_, Index_> >
class CustomDenseChunkedMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * @param manager Pointer to a `CustomDenseChunkedMatrixManager` instance that manages the chunks for this matrix.
     * @param opt Further options for chunked extraction.
     */
    CustomDenseChunkedMatrix(std::shared_ptr<Manager_> manager, const CustomDenseChunkedMatrixOptions& opt) : 
        my_manager(std::move(manager)),
        my_coordinator(my_manager->row_stats(), my_manager->column_stats()),
        my_cache_size_in_elements(opt.maximum_cache_size / sizeof(ChunkValue_)),
        my_require_minimum_cache(opt.require_minimum_cache),
        my_cache_subset(opt.cache_subset)
    {}

private:
    std::shared_ptr<Manager_> my_manager;
    CustomChunkedMatrix_internal::ChunkCoordinator<false, ChunkValue_, Index_> my_coordinator;
    std::size_t my_cache_size_in_elements;
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
        return false;
    }

    double is_sparse_proportion() const {
        return 0;
    }

    using tatami::Matrix<Value_, Index_>::dense;

    using tatami::Matrix<Value_, Index_>::sparse;

    /********************
     *** Myopic dense ***
     ********************/
private:
    template<bool oracle_, template<bool, bool, bool, typename, typename, typename, class> class Extractor_, typename ... Args_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > raw_dense_internal(bool row, Index_ non_target_length, Args_&& ... args) const {
        auto stats = [&]{
            if (row) {
                // Remember, the num_chunks_per_column is the number of slabs needed to divide up all the *rows* of the matrix.
                return SlabCacheStats<Index_>(my_coordinator.get_chunk_nrow(), non_target_length, my_coordinator.get_num_chunks_per_column(), my_cache_size_in_elements, my_require_minimum_cache);
            } else {
                // Remember, the num_chunks_per_row is the number of slabs needed to divide up all the *columns* of the matrix.
                return SlabCacheStats<Index_>(my_coordinator.get_chunk_ncol(), non_target_length, my_coordinator.get_num_chunks_per_row(), my_cache_size_in_elements, my_require_minimum_cache);
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

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > dense_internal(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, const tatami::Options&) const {
        auto non_target = (row ? my_coordinator.get_ncol() : my_coordinator.get_nrow());
        return raw_dense_internal<oracle_, CustomChunkedMatrix_internal::DenseFull>(row, non_target, std::move(oracle));
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > dense_internal(
        bool row, 
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options&) 
    const {
        return raw_dense_internal<oracle_, CustomChunkedMatrix_internal::DenseBlock>(row, block_length, std::move(oracle), block_start, block_length);
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > dense_internal(
        bool row, 
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options&) 
    const {
        auto num_indices = indices_ptr->size();
        return raw_dense_internal<oracle_, CustomChunkedMatrix_internal::DenseIndex>(row, num_indices, std::move(oracle), std::move(indices_ptr));
    }

public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        return dense_internal<false>(row, false, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return dense_internal<false>(row, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return dense_internal<false>(row, false, std::move(indices_ptr), opt);
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
        return dense_internal<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        Index_ block_start, 
        Index_ block_length, 
        const tatami::Options& opt) 
    const {
        return dense_internal<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        bool row, 
        std::shared_ptr<const tatami::Oracle<Index_> > oracle, 
        tatami::VectorPtr<Index_> indices_ptr, 
        const tatami::Options& opt) 
    const {
        return dense_internal<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(dense(row, opt), my_coordinator.get_non_target_dim(row), opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return std::make_unique<tatami::BlockSparsifiedWrapper<false, Value_, Index_> >(dense(row, block_start, block_length, opt), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        auto d = dense(row, indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<false, Value_, Index_> >(std::move(d), std::move(indices_ptr), opt);
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
        return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(dense(row, std::move(oracle), opt), my_coordinator.get_non_target_dim(row), opt);
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
        auto d = dense(row, std::move(oracle), indices_ptr, opt);
        return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(std::move(d), std::move(indices_ptr), opt);
    }
};

}

#endif
