#ifndef RITSUKO_HDF5_VLS_VALIDATE_HPP
#define RITSUKO_HDF5_VLS_VALIDATE_HPP

#include <string>
#include <vector>
#include <stdexcept>

#include "H5Cpp.h"

#include "../get_name.hpp"
#include "../pick_1d_block_size.hpp"
#include "../pick_nd_block_dimensions.hpp"
#include "../IterateNdDataset.hpp"
#include "Pointer.hpp"

/**
 * @file validate.hpp
 * @brief Helper functions to validate VLS arrays.
 */

namespace ritsuko {

namespace hdf5 {

namespace vls {

/**
 * Check that the pointers for a 1-dimensional VLS array is valid.
 * An error is thrown if any pointers are out of range of the associated heap dataset.
 *
 * @param handle Handle to the pointer dataset for a VLS array, see `open_pointers()`.
 * @param full_length Length of the dataset as a 1-dimensional vector.
 * @param heap_length Length of the heap dataset. 
 * @param buffer_size Size of the buffer for reading pointers by block. 
 */
template<typename Offset_, typename Length_>
inline void validate_1d_array(const H5::DataSet& handle, hsize_t full_length, hsize_t heap_length, hsize_t buffer_size) {
    hsize_t block_size = pick_1d_block_size(handle.getCreatePlist(), full_length, buffer_size);
    H5::DataSpace mspace(1, &block_size), dspace(1, &full_length);
    std::vector<Pointer<Offset_, Length_> > buffer(block_size);
    auto dtype = define_pointer_datatype<Offset_, Length_>();

    for (hsize_t i = 0; i < full_length; i += block_size) {
        auto available = std::min(full_length - i, block_size);
        constexpr hsize_t zero = 0;
        mspace.selectHyperslab(H5S_SELECT_SET, &available, &zero);
        dspace.selectHyperslab(H5S_SELECT_SET, &available, &i);

        handle.read(buffer.data(), dtype, mspace, dspace);
        for (hsize_t j = 0; j < available; ++j) {
            const auto& val = buffer[j];
            hsize_t start = val.offset;
            hsize_t count = val.length;
            if (start > heap_length || start + count > heap_length) {
                throw std::runtime_error("VLS array pointers at '" + get_name(handle) + "' are out of range of the heap");
            }
        }
    }
}

/**
 * Check that the pointers for an N-dimensional VLS array is valid.
 * An error is thrown if any pointers are out of range of the associated heap dataset.
 *
 * @param handle Handle to the pointer dataset for a VLS array, see `open_pointers()`.
 * @param dimensions Dimensions of the dataset. 
 * @param heap_length Length of the heap dataset. 
 * @param buffer_size Size of the buffer for reading pointers by block. 
 */
template<typename Offset_, typename Length_>
void validate_nd_array(const H5::DataSet& handle, const std::vector<hsize_t>& dimensions, hsize_t heap_length, hsize_t buffer_size) {
    std::vector<Pointer<Offset_, Length_> > buffer;
    auto dtype = define_pointer_datatype<Offset_, Length_>();
    auto blocks = pick_nd_block_dimensions(handle.getCreatePlist(), dimensions, buffer_size);
    IterateNdDataset iter(dimensions, blocks);

    while (!iter.finished()) {
        buffer.resize(iter.current_block_size());

        // Scope this to ensure that 'mspace' doesn't get changed by
        // 'iter.next()' before the destructor is called.
        {
            const auto& mspace = iter.memory_space();
            handle.read(buffer.data(), dtype, mspace, iter.file_space());
            for (const auto& val : buffer) {
                hsize_t start = val.offset;
                hsize_t count = val.length;
                if (start > heap_length || start + count > heap_length) {
                    throw std::runtime_error("VLS array pointers at '" + get_name(handle) + "' are out of range of the heap");
                }
            }
        }

        iter.next();
    }
}

}

}

}

#endif
