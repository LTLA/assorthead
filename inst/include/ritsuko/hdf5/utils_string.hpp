#ifndef RITSUKO_HDF5_UTILS_STRINGS_HPP
#define RITSUKO_HDF5_UTILS_STRINGS_HPP

#include "H5Cpp.h"

/**
 * @file utils_string.hpp
 * @brief Utilities for dealing with HDF5 strings.
 */

namespace ritsuko {

namespace hdf5 {

/**
 * Get the length of a string, either by searching for the first null terminator or by reaching the `max` length.
 *
 * @param ptr Pointer to the start of a C-style string.
 * @param max Maximum length of the array referenced by `ptr`.
 *
 * @return The number of characters to the first occurence of the null terminator or `max`, depending on which is smaller.
 */
inline size_t find_string_length(const char* ptr, size_t max) {
    size_t j = 0;
    for (; j < max && ptr[j] != '\0'; ++j) {}
    return j;
}

/**
 * @brief Release memory for HDF5's variable length strings.
 *
 * This provides an RAII interface for HDF5's variable length strings.
 * The idea is to create an instance of this class immediately after the `H5::DataSet::read()` call.
 * The allocated memory for each string is then reclaimed once the instance goes out of scope.
 */
class VariableStringCleaner {
public:
    /**
     * @param tid ID for the HDF5 datatype for the in-memory strings.
     * @param sid ID for the HDF5 dataspace for the in-memory strings.
     * @param buffer Array of C-style strings allocated by `H5::DataSet::read()` with the specified datatype and dataspace.
     */ 
    VariableStringCleaner(hid_t tid, hid_t sid, char** buffer) : my_tid(tid), my_sid(sid), my_buffer(buffer) {}

    /**
     * @cond
     */
    ~VariableStringCleaner() {
        H5Dvlen_reclaim(my_tid, my_sid, H5P_DEFAULT, my_buffer);
    }
    /**
     * @endcond
     */

private:
    hid_t my_tid, my_sid;
    char** my_buffer; 
};

}

}

#endif
