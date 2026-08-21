#ifndef TATAMI_HDF5_UTILS_HPP
#define TATAMI_HDF5_UTILS_HPP

#include "H5Cpp.h"

#include "tatami/tatami.hpp"

#include <cstdint>
#include <array>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <cassert>
#include <cmath>

/**
 * @file utils.hpp
 * @brief Utilities for HDF5 extraction.
 */

namespace tatami_hdf5 {

/**
 * Layout of the matrix inside the HDF5 file.
 */
enum class WriteStorageLayout { COLUMN, ROW };

/**
 * Numeric type of the HDF5 dataset in which to store matrix contents.
 */
enum class WriteStorageType { INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, DOUBLE };

/**
 * @cond
 */
template<typename Left_, typename Right_>
bool is_less_than_or_equal(Left_ l, Right_ r) {
    static_assert(std::is_integral<Left_>::value);
    static_assert(std::is_integral<Right_>::value);

    constexpr bool lsigned = std::is_signed<Left_>::value;
    constexpr bool rsigned = std::is_signed<Right_>::value;
    if constexpr(lsigned == rsigned) {
        return l <= r;
    } else if constexpr(lsigned) {
        return l <= 0 || static_cast<typename std::make_unsigned<Left_>::type>(l) <= r;
    } else {
        return r >= 0 && l <= static_cast<typename std::make_unsigned<Right_>::type>(r);
    }
}

template<typename Native_, typename Max_>
bool fits_upper_limit(Max_ max) {
    static_assert(std::is_integral<Native_>::value);

    if constexpr(std::is_integral<Max_>::value) { // Native_ is already integral, so no need to check that.
        constexpr auto native_max = std::numeric_limits<Native_>::max();
        return is_less_than_or_equal(max, native_max);
    } else {
        // We don't compare values directly as the Native_-to-float conversion might not be exact;
        // if native_max gets rounded up during the conversion, we might end up with a situation where 'native_max < max <= FLOAT(native_max)'. 
        // This would result in undefined behavior when casting values equal to 'max' to Native_. 
        //
        // So instead, we compare the number of bits in Native_ with that required to store our (truncated) 'max'.
        // We ignore negative or zero values of 'max' as required_bits_for_float() expects positive values.
        // (Non-positive values would always be less than any 'native_max', so we can always return true in such cases.)
        constexpr auto digits = std::numeric_limits<Native_>::digits;
        assert(max == std::trunc(max));
        return (max <= 0 || sanisizer::required_bits_for_float(max) <= digits);
    }
}

template<typename Native_, typename Min_>
bool fits_lower_limit(Min_ min) {
    static_assert(std::is_integral<Native_>::value);

    if constexpr(std::is_integral<Min_>::value) {
        constexpr auto native_min = std::numeric_limits<Native_>::min();
        return is_less_than_or_equal(native_min, min);
    } else {
        assert(min == std::trunc(min));
        if constexpr(std::is_unsigned<Native_>::value) {
            return min >= 0;
        } else {
            // Pretty much the same logic as fits_upper_limit() but we reverse the sign.
            // We add 1 before reversing to account for the sign bit, i.e., -128 becomes 127 for 7 bits.
            constexpr auto digits = std::numeric_limits<Native_>::digits;
            return (min >= -1 || sanisizer::required_bits_for_float(-(min + 1)) <= digits); 
        }
    }
}

template<typename Value_>
void check_integer_range_fit(const WriteStorageType data_type, Value_ lower_data, Value_ upper_data) {
    bool okay = false;

    switch (data_type) {
        case WriteStorageType::INT8:
            okay = fits_lower_limit<std::int8_t  >(lower_data) && fits_upper_limit<std::int8_t  >(upper_data);
            break;
        case WriteStorageType::UINT8:
            okay = fits_lower_limit<std::uint8_t >(lower_data) && fits_upper_limit<std::uint8_t >(upper_data);
            break;
        case WriteStorageType::INT16:
            okay = fits_lower_limit<std::int16_t >(lower_data) && fits_upper_limit<std::int16_t >(upper_data);
            break;
        case WriteStorageType::UINT16:
            okay = fits_lower_limit<std::uint16_t>(lower_data) && fits_upper_limit<std::uint16_t>(upper_data);
            break;
        case WriteStorageType::INT32:
            okay = fits_lower_limit<std::int32_t >(lower_data) && fits_upper_limit<std::int32_t >(upper_data);
            break;
        case WriteStorageType::UINT32:
            okay = fits_lower_limit<std::uint32_t>(lower_data) && fits_upper_limit<std::uint32_t>(upper_data);
            break;
        case WriteStorageType::INT64:
            okay = fits_lower_limit<std::int64_t >(lower_data) && fits_upper_limit<std::int64_t >(upper_data);
            break;
        case WriteStorageType::UINT64:
            okay = fits_lower_limit<std::uint64_t>(lower_data) && fits_upper_limit<std::uint64_t>(upper_data);
            break;
        default:
            ;  // we should never get to this point as everything should already be truncated.
    }

    if (!okay) {
        throw std::runtime_error("no integer type can store the matrix values");
    }
}

template<typename Value_>
void check_data_value_fit(const WriteStorageType data_type, Value_ val) {
    if (data_type != WriteStorageType::DOUBLE && data_type != WriteStorageType::FLOAT) {
        if constexpr(std::is_floating_point<Value_>::value) {
            val = std::trunc(val);
            if (!std::isfinite(val)) {
                throw std::runtime_error("cannot store non-finite floating-point values as integers");
            }
        }
        check_integer_range_fit(data_type, val, val);
    }
}

template<typename Value_>
WriteStorageType choose_data_type(
    const std::optional<WriteStorageType>& data_type,
    Value_ lower_data,
    Value_ upper_data,
    bool has_decimal,
    bool force_integer,
    bool has_nonfinite
) {
    if (!data_type.has_value()) {
        if ((has_decimal && !force_integer) || has_nonfinite) {
            if constexpr(std::is_same<Value_, float>::value) {
                return WriteStorageType::FLOAT;
            } else {
                return WriteStorageType::DOUBLE;
            }
        }

        if constexpr(std::is_floating_point<Value_>::value) {
            lower_data = std::trunc(lower_data);
            upper_data = std::trunc(upper_data);
        }

        if (lower_data < 0) {
            if (fits_lower_limit<std::int8_t>(lower_data) && fits_upper_limit<std::int8_t>(upper_data)) {
                return WriteStorageType::INT8;
            } else if (fits_lower_limit<std::int16_t>(lower_data) && fits_upper_limit<std::int16_t>(upper_data)) {
                return WriteStorageType::INT16;
            } else if (fits_lower_limit<std::int32_t>(lower_data) && fits_upper_limit<std::int32_t>(upper_data)) {
                return WriteStorageType::INT32;
            } else if (fits_lower_limit<std::int64_t>(lower_data) && fits_upper_limit<std::int64_t>(upper_data)) {
                return WriteStorageType::INT64;
            }

        } else {
            if (fits_upper_limit<std::uint8_t>(upper_data)) {
                return WriteStorageType::UINT8;
            } else if (fits_upper_limit<std::uint16_t>(upper_data)) {
                return WriteStorageType::UINT16;
            } else if (fits_upper_limit<std::uint32_t>(upper_data)) {
                return WriteStorageType::UINT32;
            } else if (fits_upper_limit<std::uint64_t>(upper_data)) {
                return WriteStorageType::UINT64;
            }
        }

        throw std::runtime_error("no type can store the matrix values");
    }

    const auto dtype = *data_type;
    if (data_type != WriteStorageType::DOUBLE && data_type != WriteStorageType::FLOAT) {
        if constexpr(std::is_floating_point<Value_>::value) {
            lower_data = std::trunc(lower_data);
            upper_data = std::trunc(upper_data);
            if (has_nonfinite) {
                throw std::runtime_error("cannot store non-finite floating-point values as integers");
            }
        }
        check_integer_range_fit(dtype, lower_data, upper_data);
    }

    return dtype;
}

inline const H5::PredType* choose_pred_type(WriteStorageType type) {
    const H5::PredType* dtype = NULL;

    switch (type) {
        case WriteStorageType::INT8:
            dtype = &(H5::PredType::NATIVE_INT8);
            break;
        case WriteStorageType::UINT8:
            dtype = &(H5::PredType::NATIVE_UINT8);
            break;
        case WriteStorageType::INT16:
            dtype = &(H5::PredType::NATIVE_INT16);
            break;
        case WriteStorageType::UINT16:
            dtype = &(H5::PredType::NATIVE_UINT16);
            break;
        case WriteStorageType::INT32:
            dtype = &(H5::PredType::NATIVE_INT32);
            break;
        case WriteStorageType::UINT32:
            dtype = &(H5::PredType::NATIVE_UINT32);
            break;
        case WriteStorageType::INT64:
            dtype = &(H5::PredType::NATIVE_INT64);
            break;
        case WriteStorageType::UINT64:
            dtype = &(H5::PredType::NATIVE_UINT64);
            break;
        case WriteStorageType::FLOAT:
            dtype = &(H5::PredType::NATIVE_FLOAT);
            break;
        case WriteStorageType::DOUBLE:
            dtype = &(H5::PredType::NATIVE_DOUBLE);
            break;
        default:
            throw std::runtime_error("automatic HDF5 output type must be resolved before creating a HDF5 dataset");
    }

    return dtype;
}

template<typename T>
const H5::PredType& define_mem_type() {
    if constexpr(std::is_same<int, T>::value) {
        return H5::PredType::NATIVE_INT;
    } else if (std::is_same<unsigned int, T>::value) {
        return H5::PredType::NATIVE_UINT;
    } else if (std::is_same<long, T>::value) {
        return H5::PredType::NATIVE_LONG;
    } else if (std::is_same<unsigned long, T>::value) {
        return H5::PredType::NATIVE_ULONG;
    } else if (std::is_same<long long, T>::value) {
        return H5::PredType::NATIVE_LLONG;
    } else if (std::is_same<unsigned long long, T>::value) {
        return H5::PredType::NATIVE_ULLONG;
    } else if (std::is_same<short, T>::value) {
        return H5::PredType::NATIVE_SHORT;
    } else if (std::is_same<unsigned short, T>::value) {
        return H5::PredType::NATIVE_USHORT;
    } else if (std::is_same<char, T>::value) {
        return H5::PredType::NATIVE_CHAR;
    } else if (std::is_same<unsigned char, T>::value) {
        return H5::PredType::NATIVE_UCHAR;
    } else if (std::is_same<double, T>::value) {
        return H5::PredType::NATIVE_DOUBLE;
    } else if (std::is_same<float, T>::value) {
        return H5::PredType::NATIVE_FLOAT;
    } else if (std::is_same<long double, T>::value) {
        return H5::PredType::NATIVE_LDOUBLE;
    } else if (std::is_same<uint8_t, T>::value) {
        return H5::PredType::NATIVE_UINT8;
    } else if (std::is_same<int8_t, T>::value) {
        return H5::PredType::NATIVE_INT8;
    } else if (std::is_same<uint16_t, T>::value) {
        return H5::PredType::NATIVE_UINT16;
    } else if (std::is_same<int16_t, T>::value) {
        return H5::PredType::NATIVE_INT16;
    } else if (std::is_same<uint32_t, T>::value) {
        return H5::PredType::NATIVE_UINT32;
    } else if (std::is_same<int32_t, T>::value) {
        return H5::PredType::NATIVE_INT32;
    } else if (std::is_same<uint64_t, T>::value) {
        return H5::PredType::NATIVE_UINT64;
    } else if (std::is_same<int64_t, T>::value) {
        return H5::PredType::NATIVE_INT64;
    }
    static_assert("unsupported HDF5 type for template parameter 'T'");
}

template<bool integer_only, class GroupLike>
H5::DataSet open_and_check_dataset(const GroupLike& handle, const std::string& name) {
    // Avoid throwing H5 exceptions.
    if (!H5Lexists(handle.getId(), name.c_str(), H5P_DEFAULT) || handle.childObjType(name) != H5O_TYPE_DATASET) {
        throw std::runtime_error("no child dataset named '" + name + "'");
    }

    auto dhandle = handle.openDataSet(name);
    auto type = dhandle.getTypeClass();
    if constexpr(integer_only) {
        if (type != H5T_INTEGER) {
            throw std::runtime_error(std::string("expected integer values in the '") + name + "' dataset");
        }
    } else {
        if (type != H5T_INTEGER && type != H5T_FLOAT) { 
            throw std::runtime_error(std::string("expected numeric values in the '") + name + "' dataset");
        }
    }

    return dhandle;
}

template<int N>
std::array<hsize_t, N> get_array_dimensions(const H5::DataSet& handle, const std::string& name) {
    auto space = handle.getSpace();

    int ndim = space.getSimpleExtentNdims();
    if (ndim != N) {
        throw std::runtime_error(std::string("'") + name + "' should be a " + std::to_string(N) + "-dimensional array");
    }

    std::array<hsize_t, N> dims_out;
    space.getSimpleExtentDims(dims_out.data(), NULL);
    return dims_out;
}
/**
 * @endcond
 */

}

#endif
