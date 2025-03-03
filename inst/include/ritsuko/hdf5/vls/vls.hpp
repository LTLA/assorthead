#ifndef RITSUKO_HDF5_VLS_HPP
#define RITSUKO_HDF5_VLS_HPP

#include "open.hpp"
#include "Pointer.hpp"
#include "Stream1dArray.hpp"
#include "validate.hpp"

/**
 * @file vls.hpp
 * @brief Utilities for handling **ritsuko**'s custom VLS arrays.
 */

namespace ritsuko {

namespace hdf5 {

/**
 * @namespace ritsuko::hdf5::vls
 * @brief Assorted functions for handling **ritsuko**'s custom VLS arrays.
 *
 * One weakness of HDF5 is its inability to efficiently handle variable length string (VLS) arrays.
 * Storing them as fixed-length strings requires padding all strings to the longest string,
 * causing an inflation in disk usage that cannot be completely negated by compression.
 * On the other hand, HDF5's own VLS datatype does not compress the strings themselves, only the pointers to those strings.
 *
 * To patch over this weakness in current versions of HDF5, **ritsuko** introduces its own concept of a VLS array.
 * This is defined by two HDF5 datasets - one storing a VLS heap, and another storing pointers into that heap.
 *
 * - The VLS heap is a 1-dimensional dataset of unsigned 8-bit integers, containing the concatenation of bytes from all variable length strings in the VLS array.
 *   Check out `open_heap()` for details.
 * - The pointer dataset is an N-dimensional dataset of a compound datatype defined by `define_pointer_datatype()`.
 *   Each entry of the pointer dataset contains the starting offset and length of a single VLS on the heap.
 *   Check out `open_pointers()` for details.
 *
 * The idea is to read the pointer dataset into an array of `Pointer` instances,
 * and then use the offset and length for each `Pointer` to extract a slice of characters from the heap.
 * Each slice defines the VLS corresponding to that `Pointer`.
 *
 * Typically the pointer and heap datasets for a single VLS array will be stored in their own group,
 * where they can be opened by `open_pointers()` and `open_heap()`, respectively.
 * See also `Stream1dArray` to quickly stream the contents of a VLS array.
 */
namespace vls {}

}

}

#endif
