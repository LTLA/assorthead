#ifndef TATAMI_WRITE_SPARSE_MATRIX_TO_HDF5_HPP
#define TATAMI_WRITE_SPARSE_MATRIX_TO_HDF5_HPP

#include "tatami/tatami.hpp"
#include "utils.hpp"

#include "H5Cpp.h"

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <optional>

/**
 * @file write_compressed_sparse_matrix.hpp
 * @brief Write a compressed sparse matrix into a HDF5 file.
 */

namespace tatami_hdf5 {

/**
 * @brief Parameters for `write_compressed_sparse_matrix()`.
 */
struct WriteCompressedSparseMatrixOptions {
    /**
     * Name of the dataset in which to store the data values for non-zero elements.
     * If unset, defaults to `"data"`.
     */
    std::optional<std::string> data_name;

    /**
     * Name of the dataset in which to store the indices for non-zero elements.
     * If unset, defaults to `"indices"`.
     */
    std::optional<std::string> index_name;

    /**
     * Name of the dataset in which to store the column/row pointers.
     * If unset, defaults to `"indptr"`.
     */
    std::optional<std::string> ptr_name;

    /**
     * Whether to save in the compressed sparse column layout.
     * If unset, this is determined from the layout of the input matrix.
     */
    std::optional<WriteStorageLayout> columnar;

    /**
     * Storage type for the matrix data.
     * If unset, it is automatically determined from the range and integralness of the data in the input matrix.
     */
    std::optional<WriteStorageType> data_type;

    /**
     * Whether to force non-integer floating point values into an integer storage mode.
     * Only relevant if `data_type` is unset. 
     * If `true` and/or all values are integers, the smallest integer storage mode that fits the (truncated) floats is used.
     * If `false` and any non-integer values are detected, the `DOUBLE` storage mode is used instead.
     */
    bool force_integer = false;

    /**
     * Storage type for the row/column indices.
     * If unset, it is automatically determined from the range of the indices in the input matrix.
     */
    std::optional<WriteStorageType> index_type;

    /**
     * Storage type for the index pointers.
     * If unset, it is automatically determined from the number of structural non-zero elements.
     */
    std::optional<WriteStorageType> ptr_type;

    /**
     * Compression level of DEFLATE.
     * This should be any integer between 0 and 9 inclusive.
     * At zero, no compression is performed.
     */
    int deflate_level = 6;

    /**
     * Size of the chunks used for compression.
     * Only used if `deflate_level > 0`.
     */
    hsize_t chunk_size = sanisizer::cap<hsize_t>(10000);

    /**
     * Whether to enable HDF5's shuffle filter, which reorders the bytes for better compression.
     * Check out HDF5's documentation for `H5Pset_shuffle()` for more details.
     * Only used if `deflate_level > 0`.
     */
    bool shuffle = true;

    /**
     * Whether to use a two-pass algorithm to first determine the number of non-zero elements before creating the dataset.
     * If `false`, a one-pass algorithm with extensible HDF5 datasets is used.
     * Ignored if either `data_type` or `index_type` is unset, in which case a two-pass algorithm is always used.
     */
    bool two_pass = false;

    /**
     * Number of threads to use for the first pass through the input matrix.
     * This is only used to determine the number of non-zero elements and check storage types.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
inline H5::DataSet create_1d_compressed_hdf5_dataset(H5::Group& location, WriteStorageType type, const std::string& name, hsize_t length, int deflate_level, hsize_t chunk, bool shuffle) {
    H5::DataSpace dspace(1, &length);
 	H5::DSetCreatPropList plist;

    if (deflate_level >= 0 && length) {
        plist.setDeflate(deflate_level);
        if (chunk > length) {
            plist.setChunk(1, &length);
        } else {
            plist.setChunk(1, &chunk);
        }
        if (shuffle) {
            plist.setShuffle();
        }
    }

    const auto dtype = choose_pred_type(type);
    return location.createDataSet(name, *dtype, dspace, plist);
}

template<typename Type_>
bool does_non_negative_integer_fit(const WriteStorageType type, const Type_ x) {
    static_assert(std::is_integral<Type_>::value);

    bool okay = false;
    switch (type) {
        case WriteStorageType::INT8:
            okay = fits_upper_limit<std::int8_t>(x);
            break;
        case WriteStorageType::UINT8:
            okay = fits_upper_limit<std::uint8_t>(x);
            break;
        case WriteStorageType::INT16:
            okay = fits_upper_limit<std::int16_t >(x);
            break;
        case WriteStorageType::UINT16:
            okay = fits_upper_limit<std::uint16_t>(x);
            break;
        case WriteStorageType::INT32:
            okay = fits_upper_limit<std::int32_t >(x);
            break;
        case WriteStorageType::UINT32:
            okay = fits_upper_limit<std::uint32_t>(x);
            break;
        case WriteStorageType::INT64:
            okay = fits_upper_limit<std::int64_t >(x);
            break;
        case WriteStorageType::UINT64:
            okay = fits_upper_limit<std::uint64_t>(x);
            break;
        default:
            // okay remains false, as the x must be integer.
            break;
    }
    return okay;
}

template<typename Index_>
WriteStorageType choose_index_type(const std::optional<WriteStorageType>& index_type, Index_ upper_index) {
    static_assert(std::is_integral<Index_>::value);

    if (!index_type.has_value()) {
        if (fits_upper_limit<std::uint8_t>(upper_index)) {
            return WriteStorageType::UINT8;
        } else if (fits_upper_limit<std::uint16_t>(upper_index)) {
            return WriteStorageType::UINT16;
        } else if (fits_upper_limit<std::uint32_t>(upper_index)) {
            return WriteStorageType::UINT32;
        } else if (fits_upper_limit<std::uint64_t>(upper_index)) {
            return WriteStorageType::UINT64;
        }
        throw std::runtime_error("no type can store the largest index");
    }

    const auto itype = *index_type;
    if (!does_non_negative_integer_fit(itype, upper_index)) {
        throw std::runtime_error("specified type cannot store the largest index");
    }

    return itype;
}

inline WriteStorageType choose_ptr_type(const std::optional<WriteStorageType>& ptr_type, hsize_t nnzero) {
    if (!ptr_type.has_value()) {
        if (fits_upper_limit<std::uint32_t>(nnzero)) {
            return WriteStorageType::UINT32;
        } else if (fits_upper_limit<std::uint64_t>(nnzero)) {
            return WriteStorageType::UINT64;
        }

        throw std::runtime_error("no type can store the number of non-zero elements");
    }

    const auto ptype = *ptr_type;
    if (!does_non_negative_integer_fit(ptype, nnzero)) {
        throw std::runtime_error("specified type cannot store the number of non-zero elements");
    }

    return ptype;
}

template<typename Value_, typename Index_>
struct WriteSparseHdf5Statistics {
    Value_ lower_data = 0;
    Value_ upper_data = 0;
    Index_ upper_index = 0;
    hsize_t non_zeros = 0;
    bool has_decimal = false;
    bool has_nonfinite = false;

    void add_value(Value_ val) {
        if constexpr(!std::is_integral<Value_>::value) {
            if (std::trunc(val) != val) {
                has_decimal = true;
            }
            if (!std::isfinite(val)) {
                has_nonfinite = true;
            }
        }

        if (val < lower_data) {
            lower_data = val;
        } else if (val > upper_data) {
            upper_data = val;
        }
    }

    void add_index(Index_ idx) {
        if (idx > upper_index) {
            upper_index = idx;
        }
    }
};

template<typename Value_, typename Index_>
void update_hdf5_stats(const tatami::SparseRange<Value_, Index_>& extracted, WriteSparseHdf5Statistics<Value_, Index_>& output) {
    // We need to protect the addition just in case it overflows from having too many non-zero elements.
    output.non_zeros = sanisizer::sum<hsize_t>(output.non_zeros, extracted.number);

    for (Index_ i = 0; i < extracted.number; ++i) {
        output.add_value(extracted.value[i]);
    }

    for (Index_ i = 0; i < extracted.number; ++i) {
        output.add_index(extracted.index[i]);
    }
}

template<typename Value_, typename Index_>
void update_hdf5_stats(const Value_* extracted, Index_ n, WriteSparseHdf5Statistics<Value_, Index_>& output) {
    Index_ local_nonzero = 0;
    for (Index_ i = 0; i < n; ++i) {
        auto val = extracted[i];
        if (val == 0) {
            continue;
        }
        ++local_nonzero;
        output.add_value(val);
        output.add_index(i);
    }

    // Checking that there aren't overflows, but doing so outside of the hot loop for perf.
    output.non_zeros = sanisizer::sum<hsize_t>(output.non_zeros, local_nonzero);
}

template<typename Value_, typename Index_>
WriteSparseHdf5Statistics<Value_, Index_> write_sparse_hdf5_statistics(const tatami::Matrix<Value_, Index_>& mat, int nthreads) {
    const auto NR = mat.nrow(), NC = mat.ncol();

    WriteSparseHdf5Statistics<Value_, Index_> output;
    auto collected = sanisizer::create<std::vector<WriteSparseHdf5Statistics<Value_, Index_> > >(nthreads - 1); // nthreads had better be >= 1.
    int num_used;

    if (mat.sparse()) {
        if (mat.prefer_rows()) {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<true>(mat, true, start, len);
                std::vector<Value_> xbuffer(NC);
                std::vector<Index_> ibuffer(NC);
                for (Index_ r = start, end = start + len; r < end; ++r) {
                    auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                    update_hdf5_stats(extracted, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NR, nthreads);

        } else {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<true>(mat, false, start, len);
                std::vector<Value_> xbuffer(NR);
                std::vector<Index_> ibuffer(NR);
                for (Index_ c = start, end = start + len; c < end; ++c) {
                    auto extracted = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                    update_hdf5_stats(extracted, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NC, nthreads);
        }

    } else {
        if (mat.prefer_rows()) {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<false>(mat, true, start, len);
                std::vector<Value_> xbuffer(NC);
                for (Index_ r = start, end = start + len; r < end; ++r) {
                    auto extracted = wrk->fetch(r, xbuffer.data());
                    update_hdf5_stats(extracted, NC, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NR, nthreads);

        } else {
            num_used = tatami::parallelize([&](int t, Index_ start, Index_ len) -> void {
                WriteSparseHdf5Statistics<Value_, Index_> current_output;

                auto wrk = tatami::consecutive_extractor<false>(mat, false, start, len);
                std::vector<Value_> xbuffer(NR);
                for (Index_ c = start, end = start + len; c < end; ++c) {
                    auto extracted = wrk->fetch(c, xbuffer.data());
                    update_hdf5_stats(extracted, NR, current_output);
                }

                // Only move to the result buffer at the end, to avoid false sharing between threads.
                (t ? collected[t - 1] : output) = std::move(current_output);
            }, NC, nthreads);
        }
    }

    for (int i = 1; i < num_used; ++i) {
        auto& current = collected[i - 1];
        output.lower_data = std::min(output.lower_data, current.lower_data);
        output.upper_data = std::max(output.upper_data, current.upper_data);
        output.upper_index = std::max(output.upper_index, current.upper_index);
        output.non_zeros = sanisizer::sum<hsize_t>(output.non_zeros, current.non_zeros);
        output.has_decimal = output.has_decimal || current.has_decimal;
        output.has_nonfinite = output.has_nonfinite || current.has_nonfinite;
    }

    return output;
}

template<typename Value_, typename Index_>
void write_compressed_sparse_matrix_two_pass(
    const tatami::Matrix<Value_, Index_>& mat,
    H5::Group& location,
    const WriteStorageLayout layout,
    const std::string& data_name,
    const std::string& index_name,
    const std::string& ptr_name,
    const WriteCompressedSparseMatrixOptions& params
) {
    auto stats = write_sparse_hdf5_statistics(mat, params.num_threads);
    const auto data_type = choose_data_type(params.data_type, stats.lower_data, stats.upper_data, stats.has_decimal, params.force_integer, stats.has_nonfinite);
    const auto index_type = choose_index_type(params.index_type, stats.upper_index);

    // And then saving it. This time we have no choice but to iterate by the desired dimension.
    const auto non_zeros = stats.non_zeros;
    H5::DataSet data_ds = create_1d_compressed_hdf5_dataset(location, data_type, data_name, non_zeros, params.deflate_level, params.chunk_size, params.shuffle);
    H5::DataSet index_ds = create_1d_compressed_hdf5_dataset(location, index_type, index_name, non_zeros, params.deflate_level, params.chunk_size, params.shuffle);
    hsize_t offset = 0;
    H5::DataSpace inspace(1, &non_zeros);
    H5::DataSpace outspace(1, &non_zeros);
    const auto& dstype = define_mem_type<Value_>();
    const auto& ixtype = define_mem_type<Index_>();

    const Index_ NR = mat.nrow(), NC = mat.ncol();
    std::vector<hsize_t> ptrs;

    auto fill_datasets = [&](const Value_* vptr, const Index_* iptr, hsize_t count) -> void {
        if (count) {
            inspace.setExtentSimple(1, &count);
            outspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);
            data_ds.write(vptr, dstype, inspace, outspace);
            index_ds.write(iptr, ixtype, inspace, outspace);
            offset += count; // sum is safe as we already know that the number of non-zeros fits in a hsize_t.
        }
    };

    if (mat.sparse()) {
        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NR, 1));
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);

            auto wrk = tatami::consecutive_extractor<true>(mat, true, static_cast<Index_>(0), NR);
            for (Index_ r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                fill_datasets(extracted.value, extracted.index, extracted.number);
                ptrs[r + 1] = offset;
            }

        } else {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NC, 1));
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);

            auto wrk = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), NC);
            for (Index_ c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                fill_datasets(extracted.value, extracted.index, extracted.number);
                ptrs[c + 1] = offset;
            }
        }

    } else {
        std::vector<Value_> sparse_xbuffer;
        std::vector<Index_> sparse_ibuffer;
        auto fill_datasets_from_dense = [&](const Value_* extracted, Index_ n) -> void {
            sparse_xbuffer.clear();
            sparse_ibuffer.clear();
            for (Index_ i = 0; i < n; ++i) {
                if (extracted[i]) {
                    sparse_xbuffer.push_back(extracted[i]);
                    sparse_ibuffer.push_back(i);
                }
            }

            hsize_t count = sparse_xbuffer.size();
            fill_datasets(sparse_xbuffer.data(), sparse_ibuffer.data(), count);
        };

        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NR, 1));
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto wrk = tatami::consecutive_extractor<false>(mat, true, static_cast<Index_>(0), NR);
            for (Index_ r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, dbuffer.data());
                fill_datasets_from_dense(extracted, NC);
                ptrs[r + 1] = offset;
            }

        } else {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NC, 1));
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            auto wrk = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), NC);
            for (Index_ c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, dbuffer.data());
                fill_datasets_from_dense(extracted, NR);
                ptrs[c + 1] = offset;
            }
        }
    }

    // Saving the pointers.
    auto ptr_len = sanisizer::cast<hsize_t>(ptrs.size());
    H5::DataSet ptr_ds = create_1d_compressed_hdf5_dataset(
        location,
        choose_ptr_type(params.ptr_type, ptrs.back()),
        ptr_name,
        ptr_len,
        params.deflate_level,
        params.chunk_size,
        params.shuffle
    );
    H5::DataSpace ptr_space(1, &ptr_len);
    ptr_ds.write(ptrs.data(), H5::PredType::NATIVE_HSIZE, ptr_space);

    return;
}

inline H5::DataSet create_1d_compressed_hdf5_dataset(H5::Group& location, WriteStorageType type, const std::string& name, int deflate_level, hsize_t chunk, bool shuffle) {
    const hsize_t length = 0;
    constexpr auto copy = H5S_UNLIMITED; // can't directly take an address to this, guess it's a macro.
    H5::DataSpace dspace(1, &length, &copy);
 	H5::DSetCreatPropList plist;

    // Extensible datasets must be chunked.
    if (deflate_level == 0) {
        throw std::runtime_error("'deflate_level' must be positive if 'two_pass = false'");
    }

    plist.setDeflate(deflate_level); 
    plist.setChunk(1, &chunk);
    if (shuffle) {
        plist.setShuffle();
    }

    const auto dtype = choose_pred_type(type);
    return location.createDataSet(name, *dtype, dspace, plist);
}

template<typename Value_, typename Index_>
void write_compressed_sparse_matrix_one_pass(
    const tatami::Matrix<Value_, Index_>& mat,
    H5::Group& location,
    const WriteStorageLayout layout,
    const std::string& data_name,
    const std::string& index_name,
    const std::string& ptr_name,
    const WriteCompressedSparseMatrixOptions& params
){
    const auto requested_dtype = *(params.data_type);
    const auto requested_itype = *(params.index_type);
    H5::DataSet data_ds = create_1d_compressed_hdf5_dataset(location, requested_dtype, data_name, params.deflate_level, params.chunk_size, params.shuffle);
    H5::DataSet index_ds = create_1d_compressed_hdf5_dataset(location, requested_itype, index_name, params.deflate_level, params.chunk_size, params.shuffle);

    hsize_t offset = 0;
    H5::DataSpace outspace;
    const auto& dstype = define_mem_type<Value_>();
    const auto& ixtype = define_mem_type<Index_>();

    const Index_ NR = mat.nrow(), NC = mat.ncol();
    std::vector<hsize_t> ptrs;

    auto fill_datasets = [&](const Value_* vptr, const Index_* iptr, hsize_t count, H5::DataSpace& inspace) -> void {
        if (count) {
            // We need to check this because we don't know that the number of non-zeros fits in a hsize_t.
            const hsize_t new_size = sanisizer::sum<hsize_t>(offset, count);
            data_ds.extend(&new_size);
            index_ds.extend(&new_size);

            constexpr hsize_t zero = 0;
            inspace.selectHyperslab(H5S_SELECT_SET, &count, &zero);
            outspace.setExtentSimple(1, &new_size);
            outspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);

            data_ds.write(vptr, dstype, inspace, outspace);
            index_ds.write(iptr, ixtype, inspace, outspace);
            offset = new_size;
        }
    };

    if (mat.sparse()) {
        auto fill_datasets_from_sparse = [&](const Value_* vptr, const Index_* iptr, Index_ n, H5::DataSpace& inspace) -> void {
            for (Index_ i = 0; i < n; ++i) {
                check_data_value_fit(requested_dtype, vptr[i]);
                if (!does_non_negative_integer_fit(requested_itype, iptr[i])) {
                    throw std::runtime_error("specified type cannot store the largest index");
                }
            }
            // We need to check this because we don't even know that the dimension extent fits in a hsize_t.
            const auto count = sanisizer::cast<hsize_t>(n);
            fill_datasets(vptr, iptr, count, inspace);
        };

        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NR, 1));
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NC);
            const hsize_t extent = NC;
            H5::DataSpace inspace(1, &extent);

            auto wrk = tatami::consecutive_extractor<true>(mat, true, static_cast<Index_>(0), NR);
            for (Index_ r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, xbuffer.data(), ibuffer.data());
                fill_datasets_from_sparse(extracted.value, extracted.index, extracted.number, inspace);
                ptrs[r + 1] = offset;
            }

        } else {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NC, 1));
            auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(NR);
            const hsize_t extent = NR;
            H5::DataSpace inspace(1, &extent);

            auto wrk = tatami::consecutive_extractor<true>(mat, false, static_cast<Index_>(0), NC);
            for (Index_ c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                fill_datasets_from_sparse(extracted.value, extracted.index, extracted.number, inspace);
                ptrs[c + 1] = offset;
            }
        }

    } else {
        std::vector<Value_> sparse_xbuffer;
        std::vector<Index_> sparse_ibuffer;
        auto fill_datasets_from_dense = [&](const Value_* extracted, Index_ n, H5::DataSpace& inspace) -> void {
            sparse_xbuffer.clear();
            sparse_ibuffer.clear();
            for (Index_ i = 0; i < n; ++i) {
                if (extracted[i]) {
                    check_data_value_fit(requested_dtype, extracted[i]);
                    sparse_xbuffer.push_back(extracted[i]);
                    if (!does_non_negative_integer_fit(requested_itype, i)) {
                        throw std::runtime_error("specified type cannot store the largest index");
                    }
                    sparse_ibuffer.push_back(i);
                }
            }

            const auto count = sanisizer::cast<hsize_t>(sparse_xbuffer.size());
            fill_datasets(sparse_xbuffer.data(), sparse_ibuffer.data(), count, inspace);
        };

        if (layout == WriteStorageLayout::ROW) {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NR, 1));
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NC);
            const hsize_t extent = NC;
            H5::DataSpace inspace(1, &extent);

            auto wrk = tatami::consecutive_extractor<false>(mat, true, static_cast<Index_>(0), NR);
            for (Index_ r = 0; r < NR; ++r) {
                auto extracted = wrk->fetch(r, dbuffer.data());
                fill_datasets_from_dense(extracted, NC, inspace);
                ptrs[r + 1] = offset;
            }

        } else {
            ptrs.resize(sanisizer::sum<decltype(ptrs.size())>(NC, 1));
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(NR);
            const hsize_t extent = NR;
            H5::DataSpace inspace(1, &extent);

            auto wrk = tatami::consecutive_extractor<false>(mat, false, static_cast<Index_>(0), NC);
            for (Index_ c = 0; c < NC; ++c) {
                auto extracted = wrk->fetch(c, dbuffer.data());
                fill_datasets_from_dense(extracted, NR, inspace);
                ptrs[c + 1] = offset;
            }
        }
    }

    // Saving the pointers.
    auto ptr_len = sanisizer::cast<hsize_t>(ptrs.size());
    H5::DataSet ptr_ds = create_1d_compressed_hdf5_dataset(
        location,
        choose_ptr_type(params.ptr_type, ptrs.back()),
        ptr_name,
        ptr_len,
        params.deflate_level,
        params.chunk_size,
        params.shuffle
    );
    H5::DataSpace ptr_space(1, &ptr_len);
    ptr_ds.write(ptrs.data(), H5::PredType::NATIVE_HSIZE, ptr_space);
}
/**
 * @endcond
 */

/**
 * Write a sparse matrix inside a HDF5 group.
 * On return, `location` will be populated with three datasets containing the matrix contents in a compressed sparse format.
 * Storage of dimensions and other metadata (e.g., related to column versus row layout) is left to the caller. 
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 *
 * @param mat Matrix to be written to disk, presumably sparse.
 * If a dense matrix is supplied, only the non-zero elements will be written.
 * @param location Handle to a HDF5 group in which to write the matrix contents.
 * @param params Parameters to use when writing the matrix.
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>& mat, H5::Group& location, const WriteCompressedSparseMatrixOptions& params) {
    // Choosing the layout.
    WriteStorageLayout layout;
    if (params.columnar.has_value()) {
        layout = *(params.columnar);
    } else {
        if (mat.prefer_rows()) {
            layout = WriteStorageLayout::ROW;
        } else {
            layout = WriteStorageLayout::COLUMN;
        }
    }

    // Choosing the names.
    std::string data_name;
    if (params.data_name.has_value()) {
        data_name = *(params.data_name);
    } else {
        data_name = "data";
    }

    std::string index_name;
    if (params.index_name.has_value()) {
        index_name = *(params.index_name);
    } else {
        index_name = "indices";
    }

    std::string ptr_name;
    if (params.ptr_name.has_value()) {
        ptr_name = *(params.ptr_name);
    } else {
        ptr_name = "indptr";
    }

    // Only executing a one-pass strategy if the types are already known.
    if (params.two_pass || !params.data_type.has_value() || !params.index_type.has_value()) {
        write_compressed_sparse_matrix_two_pass(mat, location, layout, data_name, index_name, ptr_name, params);
    } else {
        write_compressed_sparse_matrix_one_pass(mat, location, layout, data_name, index_name, ptr_name, params);
    }
}

/**
 * Overload of `write_compressed_sparse_matrix()` with default parameters.
 *
 * @tparam Value_ Type of the matrix values.
 * @tparam Index_ Type of the row/column indices.
 *
 * @param mat Matrix to be written to disk, presumably sparse.
 * @param location Handle to a HDF5 group in which to write the matrix contents.
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>& mat, H5::Group& location) {
    WriteCompressedSparseMatrixOptions params;
    write_compressed_sparse_matrix(mat, location, params);
    return;
}

/**
 * @cond
 */
template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>* mat, H5::Group& location, const WriteCompressedSparseMatrixOptions& params) {
    return write_compressed_sparse_matrix(*mat, location, params);
}

template<typename Value_, typename Index_>
void write_compressed_sparse_matrix(const tatami::Matrix<Value_, Index_>* mat, H5::Group& location) {
    return write_compressed_sparse_matrix(*mat, location);
}
/**
 * @endcond
 */

}

#endif
