#ifndef TATAMI_HDF5_LOAD_SPARSE_MATRIX_HPP
#define TATAMI_HDF5_LOAD_SPARSE_MATRIX_HPP

#include "utils.hpp"

#include <string>
#include <vector>
#include <cstddef>

#include "H5Cpp.h"
#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file load_compressed_sparse_matrix.hpp
 *
 * @brief Load a HDF5 group into a sparse in-memory matrix.
 */

namespace tatami_hdf5 {

/**
 * Create a `tatami::CompressedSparseMatrix` from a HDF5 group containing compressed sparse data.
 *
 * @tparam Value_ Type of the matrix values in the `Matrix` interface.
 * @tparam Index_ Type of the row/column indices.
 * @tparam ValueStorage_ Vector type for storing the values of the non-zero elements.
 * Elements of this vector may be of a different type than `Value_` for more efficient storage.
 * @tparam IndexStorage_ Vector type for storing the indices.
 * Elements of this vector may be of a different type than `Index_` for more efficient storage.
 * @tparam PointerStorage_ Vector type for storing the index pointers.
 *
 * @param nr Number of rows in the matrix.
 * @param nc Number of columns in the matrix.
 * @param file Path to the file.
 * @param vals Name of the 1D dataset inside `file` containing the non-zero elements.
 * @param idx Name of the 1D dataset inside `file` containing the indices of the non-zero elements.
 * If `row = true`, this should contain column indices sorted within each row, otherwise it should contain row indices sorted within each column.
 * @param ptr Name of the 1D dataset inside `file` containing the index pointers for the start and end of each row (if `row = true`) or column (otherwise).
 * This should have length equal to the number of rows (if `row = true`) or columns (otherwise) plus 1.
 * @param row Whether the matrix is stored on disk in compressed sparse row format.
 * If false, the matrix is assumed to be stored in compressed sparse column format.
 *
 * @return Pointer to a `tatami::CompressedSparseMatrix` containing all values and indices in memory.
 * This differs from a `tatami_hdf5::CompressedSparseMatrix`, where the loading of data is deferred until requested.
 */
template<typename Value_, typename Index_, class ValueStorage_ = std::vector<Value_>, class IndexStorage_ = std::vector<Index_>, class PointerStorage_ = std::vector<std::size_t> >
std::shared_ptr<tatami::Matrix<Value_, Index_> > load_compressed_sparse_matrix(
    Index_ nr, 
    Index_ nc, 
    const std::string& file, 
    const std::string& vals, 
    const std::string& idx, 
    const std::string& ptr,
    bool row) 
{
    H5::H5File file_handle(file, H5F_ACC_RDONLY);

    auto dhandle = open_and_check_dataset<false>(file_handle, vals);
    auto nonzeros = get_array_dimensions<1>(dhandle, "vals")[0];

    auto x = sanisizer::create<ValueStorage_>(nonzeros);
    dhandle.read(x.data(), define_mem_type<tatami::ElementType<ValueStorage_> >());
    
    auto ihandle = open_and_check_dataset<true>(file_handle, idx);
    if (get_array_dimensions<1>(ihandle, "idx")[0] != nonzeros) {
        throw std::runtime_error("number of non-zero elements is not consistent between 'data' and 'idx'");
    }
    auto i = sanisizer::create<IndexStorage_>(nonzeros);
    ihandle.read(i.data(), define_mem_type<tatami::ElementType<IndexStorage_> >());

    auto phandle = open_and_check_dataset<true>(file_handle, ptr);
    auto ptr_size = get_array_dimensions<1>(phandle, "ptr")[0];
    if (ptr_size == 0 || !sanisizer::is_equal(ptr_size - 1, row ? nr : nc)) {
        throw std::runtime_error("'ptr' dataset should have length equal to the number of " + (row ? std::string("rows") : std::string("columns")) + " plus 1");
    }

    // Because HDF5 doesn't have a native type for size_t.
    auto p = sanisizer::create<PointerStorage_>(ptr_size);
    if constexpr(std::is_same<std::size_t, tatami::ElementType<PointerStorage_> >::value) {
        if constexpr(std::is_same<std::size_t, hsize_t>::value) {
            phandle.read(p.data(), H5::PredType::NATIVE_HSIZE);
        } else {
            std::vector<hsize_t> p0(ptr_size);
            phandle.read(p0.data(), H5::PredType::NATIVE_HSIZE);
            std::copy(p0.begin(), p0.end(), p.begin());
        }
    } else {
        phandle.read(p.data(), define_mem_type<tatami::ElementType<PointerStorage_> >());
    }

    return std::make_shared<tatami::CompressedSparseMatrix<Value_, Index_, ValueStorage_, IndexStorage_, PointerStorage_> >(nr, nc, std::move(x), std::move(i), std::move(p), row);
}

}

#endif
