#ifndef RITSUKO_HDF5_VLS_define_pointer_datatype_HPP
#define RITSUKO_HDF5_VLS_define_pointer_datatype_HPP

#include "H5Cpp.h"

#include <stdexcept>
#include <string>

#include "../as_numeric_datatype.hpp"
#include "../exceeds_limit.hpp"

/**
 * @file Pointer.hpp
 * @brief Compound datatype for the VLS heap pointer.
 */

namespace ritsuko {

namespace hdf5 {

namespace vls {

/**
 * @brief Pointer into the VLS heap.
 *
 * @tparam Offset_ Unsigned integer type for the starting offset on the heap. 
 * @tparam Length_ Unsigned integer type for the length of the string.
 *
 * Each `Pointer` instance defines a variable length string by referencing a slice of bytes on the VLS heap. 
 * It is expected that `Pointer` instances are stored as a compound datatype as defined by `define_pointer_datatype()`.
 */
template<typename Offset_, typename Length_>
struct Pointer {
    /**
     * Starting offset of the string on the heap, in terms of the number of bytes.
     * This should be no greater than the length of the heap dataset.
     */
    Offset_ offset;

    /**
     * Maximum length of the string on the heap, in terms of the number of bytes.
     * The sum of `offset` and `length` should be no greater than the length of the heap dataset. 
     *
     * It is not necessary to include the null terminator when setting `length`.
     * However, if the slice `[offset, offset + length)` on the heap includes a null terminator, the string should be terminated at the first occurrence.
     * This allows the slice to be easily reused for shorter strings when modifying a VLS inside an existing heap.
     */
    Length_ length;
};

/**
 * Define a compound datatype for a realization of a `Pointer` type.
 * 
 * @tparam Offset_ Unsigned integer type for the starting offset, see `Pointer::offset`.
 * @tparam Length_ Unsigned integer type for the string length, see `Pointer::length`.
 *
 * @return A HDF5 compound datatype.
 */
template<typename Offset_, typename Length_>
H5::CompType define_pointer_datatype() {
    typedef Pointer<Offset_, Length_> tmp;
    H5::CompType pointer_type(sizeof(tmp));
    pointer_type.insertMember("offset", HOFFSET(tmp, offset), as_numeric_datatype<Offset_>());
    pointer_type.insertMember("length", HOFFSET(tmp, length), as_numeric_datatype<Length_>());
    return pointer_type;
}

/**
 * Validate that a compound HDF5 datatype meets the expectations for storage in a `Pointer` instance, specifically:
 *
 * - It has exactly two members.
 * - The first member is named `offset` and is of an integer datatype that is unsigned and has no more than than `offset_precision` bits.
 * - The second member is named `length` and is of an integer datatype that is unsigned and has no more than than `length_precision` bits.
 *
 * The constraints on the precision of each integer type ensure that the pointer dataset can be represented in memory by the associated type.
 * For example, setting `offset_precision = 64` allows readers to safely assume that a `uint64_t` can be used for `Pointer::offset`.
 *
 * On success, the contents of the HDF5 dataset associated with `type` can be safely read into an array of appropriately parameterized `Pointer` instances.
 * Otherwise, an error is thrown.
 *
 * @param type Compound datatype, typically generated from a `H5::DataSet` instance.
 * @param offset_precision Maximum number of bits in the integer type used for the start position, see `Pointer::offset`.
 * @param length_precision Maximum number of bits in the integer type used for the string size, see `Pointer::length`.
 */
inline void validate_pointer_datatype(const H5::CompType& type, size_t offset_precision, size_t length_precision) {
    if (type.getNmembers() != 2) {
        throw std::runtime_error("expected VLS compound datatype to have two members");
    }

    if (type.getMemberName(0) != "offset") {
        throw std::runtime_error("first member of a VLS compound datatype should be named 'offset'");
    }
    if (type.getMemberClass(0) != H5T_INTEGER) {
        throw std::runtime_error("first member of a VLS compound datatype should have integer type");
    }
    auto offset_type = type.getMemberIntType(0);
    if (exceeds_integer_limit(offset_type, offset_precision, false)) {
        throw std::runtime_error("first member of a VLS compound datatype should not exceed a " + std::to_string(offset_precision) + "-bit unsigned integer");
    }

    if (type.getMemberName(1) != "length") {
        throw std::runtime_error("second member of a VLS compound datatype should be named 'length'");
    }
    if (type.getMemberClass(0) != H5T_INTEGER) {
        throw std::runtime_error("second member of a VLS compound datatype should have integer type");
    }
    auto length_type = type.getMemberIntType(1);
    if (exceeds_integer_limit(length_type, length_precision, false)) {
        throw std::runtime_error("second member of a VLS compound datatype should not exceed a " + std::to_string(length_precision) + "-bit unsigned integer");
    }
}

}

}

}

#endif 
