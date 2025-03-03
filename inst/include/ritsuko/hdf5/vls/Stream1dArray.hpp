#ifndef RITSUKO_HDF5_STREAM_1D_ARRAY_HPP
#define RITSUKO_HDF5_STREAM_1D_ARRAY_HPP

#include "H5Cpp.h"

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

#include "../pick_1d_block_size.hpp"
#include "../get_1d_length.hpp"
#include "../get_name.hpp"
#include "../utils_string.hpp"
#include "Pointer.hpp"

/**
 * @file Stream1dArray.hpp
 * @brief Stream a 1-dimensional VLS array into memory.
 */

namespace ritsuko {

namespace hdf5 {

namespace vls {

/**
 * @brief Stream a 1-dimensional VLS array into memory.
 *
 * @tparam Offset_ Unsigned integer type for the starting offset on the heap. 
 * @tparam Length_ Unsigned integer type for the length of the string.
 *
 * This streams in a 1-dimensional VLS array in contiguous blocks, using block sizes defined by `pick_1d_block_size()`.
 * Callers can then iterate over the individual strings.
 */
template<typename Offset_, typename Length_>
class Stream1dArray {
public:
    /**
     * @param pointers Pointer to a 1-dimensional HDF5 dataset containing the VLS pointers, see `open_pointers()`.
     * @param heap Pointer to a 1-dimensional HDF5 dataset containing the VLS heap, see `open_heap()`.
     * @param length Length of the `pointers` dataset as a 1-dimensional vector.
     * @param buffer_size Size of the buffer for holding streamed blocks of strings.
     * Larger buffers improve speed at the cost of some memory efficiency.
     */
    Stream1dArray(const H5::DataSet* pointers, const H5::DataSet* heap, hsize_t length, hsize_t buffer_size) : 
        my_pointers(pointers), 
        my_heap(heap),
        my_pointer_full_length(length), 
        my_heap_full_length(get_1d_length(my_heap->getSpace(), false)),
        my_pointer_block_size(pick_1d_block_size(my_pointers->getCreatePlist(), my_pointer_full_length, buffer_size)),
        my_pointer_mspace(1, &my_pointer_block_size),
        my_pointer_dspace(1, &my_pointer_full_length),
        my_heap_dspace(1, &my_heap_full_length),
        my_pointer_dtype(define_pointer_datatype<Offset_, Length_>()),
        my_pointer_buffer(my_pointer_block_size),
        my_final_buffer(my_pointer_block_size)
    {
    }

    /**
     * Overloaded constructor where the length is automatically determined.
     *
     * @param pointers Pointer to a 1-dimensional HDF5 dataset containing the VLS pointers, see `open_pointers()`.
     * @param heap Pointer to a 1-dimensional HDF5 dataset containing the VLS heap, see `open_heap()`.
     * @param buffer_size Size of the buffer for holding streamed blocks of strings.
     * Larger buffers improve speed at the cost of some memory efficiency.
     */
    Stream1dArray(const H5::DataSet* pointers, const H5::DataSet* heap, hsize_t buffer_size) : 
        Stream1dArray(pointers, heap, get_1d_length(pointers->getSpace(), false), buffer_size) 
    {}

public:
    /**
     * @return String at the current position of the stream.
     */
    std::string get() {
        while (my_consumed >= my_available) {
            my_consumed -= my_available;
            load(); 
        }
        return my_final_buffer[my_consumed];
    }

    /**
     * @return String at the current position of the stream.
     * Unlike `get()`, this avoids a copy by directly acquiring the string,
     * but it invalidates all subsequent `get()` and `steal()` requests until `next()` is called.
     */
    std::string steal() {
        while (my_consumed >= my_available) {
            my_consumed -= my_available;
            load(); 
        }
        return std::move(my_final_buffer[my_consumed]);
    }

    /**
     * Advance to the next position of the stream.
     *
     * @param jump Number of positions by which to advance the stream.
     */
    void next(size_t jump = 1) {
        my_consumed += jump;
    }

    /**
     * @return Length of the dataset.
     */
    hsize_t length() const {
        return my_pointer_full_length;
    }

    /**
     * @return Current position on the stream.
     */
    hsize_t position() const {
        return my_consumed + my_last_loaded;
    }

private:
    const H5::DataSet* my_pointers;
    const H5::DataSet* my_heap;
    hsize_t my_pointer_full_length, my_heap_full_length;
    hsize_t my_pointer_block_size;
    H5::DataSpace my_pointer_mspace, my_pointer_dspace;
    H5::DataSpace my_heap_mspace, my_heap_dspace;

    H5::DataType my_pointer_dtype;
    std::vector<Pointer<Offset_, Length_> > my_pointer_buffer;
    std::vector<uint8_t> my_heap_buffer;
    std::vector<std::string> my_final_buffer;

    hsize_t my_last_loaded = 0;
    hsize_t my_consumed = 0;
    hsize_t my_available = 0;

    void load() {
        if (my_last_loaded >= my_pointer_full_length) {
            throw std::runtime_error("requesting data beyond the end of the dataset at '" + get_name(*my_pointers) + "'");
        }
        my_available = std::min(my_pointer_full_length - my_last_loaded, my_pointer_block_size);

        constexpr hsize_t zero = 0;
        my_pointer_mspace.selectHyperslab(H5S_SELECT_SET, &my_available, &zero);
        my_pointer_dspace.selectHyperslab(H5S_SELECT_SET, &my_available, &my_last_loaded);
        my_heap_dspace.selectNone();
        my_pointers->read(my_pointer_buffer.data(), my_pointer_dtype, my_pointer_mspace, my_pointer_dspace);

        for (size_t i = 0; i < my_available; ++i) {
            const auto& val = my_pointer_buffer[i];
            hsize_t start = val.offset;
            hsize_t count = val.length;
            if (start > my_heap_full_length || start + count > my_heap_full_length) {
                throw std::runtime_error("VLS array pointers at '" + get_name(*my_pointers) + "' are out of range of the heap at '" + get_name(*my_heap) + "'");
            }

            auto& curstr = my_final_buffer[i];
            curstr.clear();

            if (count) {
                // Don't attempt to batch these reads as we aren't guaranteed
                // that they are non-overlapping or ordered. Hopefully HDF5 is
                // keeping enough things in cache for repeated reads.
                my_heap_mspace.setExtentSimple(1, &count);
                my_heap_mspace.selectAll();
                my_heap_dspace.selectHyperslab(H5S_SELECT_SET, &count, &start);
                my_heap_buffer.resize(count);
                my_heap->read(my_heap_buffer.data(), H5::PredType::NATIVE_UINT8, my_heap_mspace, my_heap_dspace);
                const char* text_ptr = reinterpret_cast<const char*>(my_heap_buffer.data());
                curstr.insert(curstr.end(), text_ptr, text_ptr + find_string_length(text_ptr, count));

                /*
                 * Is it generally portable to reinterpret_cast the bytes in a
                 * uint8_t array? I think so; according to the C standard,
                 * uint8_t is guaranteed to be exactly 8 bits
                 * (https://stackoverflow.com/questions/15039077/uint8-t-8-bits-guarantee),
                 * so a uint8_t value should have the same bit representation
                 * across all implementations that define the uint8_t type. If
                 * we save a byte to HDF5 as a UINT8 on one machine and read it
                 * back into to memory on another machine, we should recover
                 * the same bit pattern. Thus, the reinterpret_cast to a char*
                 * should yield the same bit pattern across machines, allowing
                 * us to portably interpret the array as a string following the
                 * ASCII/UTF-8 spec (which define each character in binary).
                 */
            }
        }

        my_last_loaded += my_available;
    }
};

}

}

}

#endif
