#ifndef TATAMI_HDF5_SPARSE_MATRIX_HPP
#define TATAMI_HDF5_SPARSE_MATRIX_HPP

#include "sparse_primary.hpp"
#include "sparse_secondary.hpp"
#include "serialize.hpp"
#include "utils.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <cstddef>

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file CompressedSparseMatrix.hpp
 *
 * @brief Defines a class for a HDF5-backed compressed sparse matrix.
 */

namespace tatami_hdf5 {

/**
 * @brief Options for HDF5 extraction.
 */
struct CompressedSparseMatrixOptions {
    /**
     * Size of the in-memory cache in bytes.
     *
     * We cache all chunks required to read a row/column in `tatami::MyopicDenseExtractor::fetch()` and related methods.
     * This allows us to re-use the cached chunks when adjacent rows/columns are requested, rather than re-reading them from disk.
     *
     * Larger caches improve access speed at the cost of memory usage.
     * Small values may be ignored as `CompressedSparseMatrix` will always allocate enough to cache a single element of the target dimension.
     */
    std::size_t maximum_cache_size = sanisizer::cap<std::size_t>(100000000);
};

/**
 * @brief Compressed sparse matrix in a HDF5 file.
 *
 * This class retrieves sparse data from the HDF5 file on demand rather than loading it all in at the start.
 * This allows us to handle very large datasets in limited memory at the cost of speed.
 *
 * We manually handle the chunk caching to speed up access for consecutive rows or columns (for compressed sparse row and column matrices, respectively).
 * The policy is to minimize the number of calls to the HDF5 library - and thus expensive file reads - by requesting large contiguous slices where possible,
 * i.e., multiple columns or rows for CSC and CSR matrices, respectively.
 * These are held in memory in the `Extractor` while the relevant column/row is returned to the user by `row()` or `column()`.
 * The size of the slice is determined by the `options` in the constructor.
 *
 * Callers should follow the suggestion of `prefer_rows()` when extracting data,
 * as this tries to minimize the number of chunks that need to be read per access request.
 * This recommendation is even stronger than for the `DenseMatrix`,
 * as the access pattern on disk for the non-preferred dimension is very suboptimal.
 *
 * As the HDF5 library is not generally thread-safe, the HDF5-related operations should only be run in a single thread.
 * This is normally handled automatically but developers can check out `serialize()` to customize the locking scheme.
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 * @tparam CachedValue_ Type of the matrix value to store in the cache.
 * This can be set to a narrower type than `Value_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all values (based on their HDF5 type or other knowledge).
 * @tparam CachedIndex_ Type of the index value to store in the cache.
 * This can be set to a narrower type than `Index_` to save memory and improve cache performance,
 * if a smaller type is known to be able to store all indices (based on their HDF5 type or other knowledge).
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class CompressedSparseMatrix final : public tatami::Matrix<Value_, Index_> {
    Index_ my_nrow, my_ncol;
    std::string my_file_name, my_value_name, my_index_name;
    std::vector<hsize_t> pointers;
    bool my_csr;

    std::size_t my_slab_cache_size; // our own cache of slabs.
    Index_ my_max_non_zeros;
    CompressedSparseMatrix_internal::ChunkCacheSizes my_chunk_cache_sizes; // HDF5's cache of uncompressed chunks.

public:
    /**
     * @param nrow Number of rows in the matrix.
     * @param ncol Number of columns in the matrix.
     * @param file_name Path to the file.
     * @param value_name Name of the 1D dataset inside `file_name` containing the values of the structural non-zero elements.
     * @param index_name Name of the 1D dataset inside `file_name` containing the indices of the structural non-zero elements.
     * If `csr = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
     * @param pointer_name Name of the 1D dataset inside `file_name` containing the index pointers for the start and end of each row (if `row = true`) or column (otherwise).
     * This should have length equal to the number of rows (if `row = true`) or columns (otherwise).
     * @param csr Whether the matrix is stored on disk in compressed sparse row format.
     * If false, the matrix is assumed to be stored in compressed sparse column format.
     * @param options Further options.
     */
    CompressedSparseMatrix(
        Index_ nrow,
        Index_ ncol,
        std::string file_name,
        std::string value_name,
        std::string index_name,
        std::string pointer_name,
        bool csr,
        const CompressedSparseMatrixOptions& options
    ) :
        my_nrow(nrow), 
        my_ncol(ncol), 
        my_file_name(std::move(file_name)), 
        my_value_name(std::move(value_name)), 
        my_index_name(std::move(index_name)), 
        my_csr(csr),
        my_slab_cache_size(options.maximum_cache_size)
    {
        // Here, the 'primary' dimension refers to the dimension by which the non-zero elements are grouped.
        // The secondary dimension is, well, the other dimension.
        Index_ primary_dim = my_csr ? my_nrow : my_ncol;
        Index_ secondary_dim = my_csr ? my_ncol : my_nrow;

        auto dim_as_str = [](bool row) -> std::string {
            if (row) {
                return "rows";
            } else {
                return "columns";
            }
        };

        serialize([&]() -> void {
            H5::H5File file_handle(my_file_name, H5F_ACC_RDONLY);
            auto dhandle = open_and_check_dataset<false>(file_handle, my_value_name);
            hsize_t nonzeros = get_array_dimensions<1>(dhandle, "value_name")[0];

            auto ihandle = open_and_check_dataset<true>(file_handle, my_index_name);
            if (get_array_dimensions<1>(ihandle, "index_name")[0] != nonzeros) {
                throw std::runtime_error("number of non-zero elements is not consistent between 'value_name' and 'index_name'");
            }

            auto phandle = open_and_check_dataset<true>(file_handle, pointer_name);
            auto ptr_size = get_array_dimensions<1>(phandle, "pointer_name")[0];
            if (ptr_size == 0 || !sanisizer::is_equal(ptr_size - 1, primary_dim)) {
                throw std::runtime_error("'pointer_name' dataset should have length equal to the number of " + dim_as_str(my_csr) + " plus 1");
            }

            auto dparms = dhandle.getCreatePlist();
            if (dparms.getLayout() == H5D_CHUNKED) {
                hsize_t dchunk_length;
                dparms.getChunk(1, &dchunk_length);
                my_chunk_cache_sizes.value = CompressedSparseMatrix_internal::compute_chunk_cache_size(nonzeros, dchunk_length, dhandle.getDataType().getSize());
            }

            auto iparms = ihandle.getCreatePlist();
            if (iparms.getLayout() == H5D_CHUNKED) {
                hsize_t ichunk_length;
                iparms.getChunk(1, &ichunk_length);
                my_chunk_cache_sizes.index = CompressedSparseMatrix_internal::compute_chunk_cache_size(nonzeros, ichunk_length, ihandle.getDataType().getSize());
            }

            // Checking the contents of the index pointers.
            pointers.resize(sanisizer::cast<decltype(pointers.size())>(ptr_size));
            phandle.read(pointers.data(), H5::PredType::NATIVE_HSIZE);
            if (pointers[0] != 0) {
                throw std::runtime_error("first index pointer should be zero");
            }
            if (pointers.back() != nonzeros) {
                throw std::runtime_error("last index pointer should be equal to the number of non-zero elements");
            }
        });

        my_max_non_zeros = 0;
        for (Index_ i = 0; i < primary_dim; ++i) {
            if (pointers[i+1] < pointers[i]) {
                throw std::runtime_error("pointers should be ordered");
            }
            auto diff = pointers[i+1] - pointers[i];
            if (sanisizer::is_greater_than(diff, secondary_dim)) {
                throw std::runtime_error("differences between pointers should be no greater than the number of " + dim_as_str(!my_csr));
            }
            if (sanisizer::is_greater_than(diff, my_max_non_zeros)) {
                my_max_non_zeros = diff; // cast is safe, because we know that it's less than the secondary_dim.
            }
        }
    }

    /**
     * Overload that uses the default `CompressedSparseMatrixOptions`.
     * @param nrow Number of rows in the matrix.
     * @param ncol Number of columns in the matrix.
     * @param file_name Path to the file.
     * @param value_name Name of the 1D dataset inside `file_name` containing the values of the structural non-zero elements.
     * @param index_name Name of the 1D dataset inside `file_name` containing the indices of the structural non-zero elements.
     * @param pointer_name Name of the 1D dataset inside `file_name` containing the index pointers for the start and end of each csr (if `csr = true`) or column (otherwise).
     * @param csr Whether the matrix is stored in compressed sparse csr format.
     */
    CompressedSparseMatrix(Index_ nrow, Index_ ncol, std::string file_name, std::string value_name, std::string index_name, std::string pointer_name, bool csr) :
        CompressedSparseMatrix(nrow, ncol, std::move(file_name), std::move(value_name), std::move(index_name), std::move(pointer_name), csr, CompressedSparseMatrixOptions()) {}

public:
    Index_ nrow() const {
        return my_nrow;
    }

    Index_ ncol() const {
        return my_ncol;
    }

    bool is_sparse() const {
        return true;
    }

    double is_sparse_proportion() const { 
        return 1;
    }

    bool prefer_rows() const {
        return my_csr;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(my_csr);
    }

    bool uses_oracle(bool) const {
        return true;
    }

    using tatami::Matrix<Value_, Index_>::dense;

    using tatami::Matrix<Value_, Index_>::sparse;

    /**************************************
     ************ Myopic dense ************
     **************************************/
private:
    CompressedSparseMatrix_internal::MatrixDetails<Index_> details() const {
        return CompressedSparseMatrix_internal::MatrixDetails<Index_>(
            my_file_name, 
            my_value_name, 
            my_index_name, 
            (my_csr ? my_nrow : my_ncol),
            (my_csr ? my_ncol : my_nrow),
            pointers, 
            my_slab_cache_size,
            my_max_non_zeros,
            my_chunk_cache_sizes
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, const tatami::Options&) const {
        if (row == my_csr) {
            return std::make_unique<CompressedSparseMatrix_internal::PrimaryFullDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle)
            );
        } else {
            return std::make_unique<CompressedSparseMatrix_internal::SecondaryFullDense<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle)
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const tatami::Options&) const {
        if (row == my_csr) {
            return std::make_unique<CompressedSparseMatrix_internal::PrimaryBlockDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), block_start, block_length
            );
        } else {
            return std::make_unique<CompressedSparseMatrix_internal::SecondaryBlockDense<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), block_start, block_length
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options&) const {
        if (row == my_csr) {
            return std::make_unique<CompressedSparseMatrix_internal::PrimaryIndexDense<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), std::move(indices_ptr)
            );
        } else {
            return std::make_unique<CompressedSparseMatrix_internal::SecondaryIndexDense<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), std::move(indices_ptr)
            );
        }
    }

public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, const tatami::Options& opt) const {
        return populate_dense<false>(row, false, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_dense<false>(row, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_dense<false>(row, false, std::move(indices_ptr), opt);
    }

    /***************************************
     ************ Myopic sparse ************
     ***************************************/
private:
    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, const tatami::Options& opt) const {
        if (row == my_csr) {
            return std::make_unique<CompressedSparseMatrix_internal::PrimaryFullSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), opt.sparse_extract_value, opt.sparse_extract_index
            );
        } else {
            return std::make_unique<CompressedSparseMatrix_internal::SecondaryFullSparse<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), opt.sparse_extract_value, opt.sparse_extract_index
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        if (row == my_csr) {
            return std::make_unique<CompressedSparseMatrix_internal::PrimaryBlockSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), block_start, block_length, opt.sparse_extract_value, opt.sparse_extract_index
            );
        } else {
            return std::make_unique<CompressedSparseMatrix_internal::SecondaryBlockSparse<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), block_start, block_length, opt.sparse_extract_value, opt.sparse_extract_index
            );
        }
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(bool row, tatami::MaybeOracle<oracle_, Index_> oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        if (row == my_csr) {
            return std::make_unique<CompressedSparseMatrix_internal::PrimaryIndexSparse<oracle_, Value_, Index_, CachedValue_, CachedIndex_> >(
                details(), std::move(oracle), std::move(indices_ptr), opt.sparse_extract_value, opt.sparse_extract_index
            );
        } else {
            return std::make_unique<CompressedSparseMatrix_internal::SecondaryIndexSparse<oracle_, Value_, Index_, CachedValue_> >(
                details(), std::move(oracle), std::move(indices_ptr), opt.sparse_extract_value, opt.sparse_extract_index
            );
        }
    }

public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, const tatami::Options& opt) const {
        return populate_sparse<false>(row, false, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_sparse<false>(row, false, block_start, block_length, opt);
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(bool row, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_sparse<false>(row, false, std::move(indices_ptr), opt);
    }

    /****************************************
     ************ Oracular dense ************
     ****************************************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, const tatami::Options& opt) const {
        return populate_dense<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_dense<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_dense<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }

    /*****************************************
     ************ Oracular sparse ************
     *****************************************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, const tatami::Options& opt) const {
        return populate_sparse<true>(row, std::move(oracle), opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, Index_ block_start, Index_ block_length, const tatami::Options& opt) const {
        return populate_sparse<true>(row, std::move(oracle), block_start, block_length, opt);
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(bool row, std::shared_ptr<const tatami::Oracle<Index_> > oracle, tatami::VectorPtr<Index_> indices_ptr, const tatami::Options& opt) const {
        return populate_sparse<true>(row, std::move(oracle), std::move(indices_ptr), opt);
    }
};

}

#endif
