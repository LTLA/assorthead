#ifndef TATAMI_TILEDB_UTILS_HPP
#define TATAMI_TILEDB_UTILS_HPP

#include <cstdint>
#include <stdexcept>
#include <limits>
#include <vector>
#include <cstddef>

#include <tiledb/tiledb>
#include "sanisizer/sanisizer.hpp"

namespace tatami_tiledb {

namespace internal {

// All TileDB-related members.
struct Components{
    Components(const std::string& location) : array(ctx, location, TILEDB_READ) {}
    Components(tiledb::Context context, const std::string& location) : ctx(std::move(context)), array(ctx, location, TILEDB_READ) {}
    tiledb::Context ctx;
    tiledb::Array array;
};

class VariablyTypedDimension {
public:
    VariablyTypedDimension() = default;

    VariablyTypedDimension(const tiledb::Dimension& dim) {
        reset(dim);
    }

public:
    void reset(const tiledb::Dimension& dim) {
        my_type = dim.type();
        switch (my_type) {
            case   TILEDB_INT8: populate(dim,  my_i8_start,  my_i8_end,  my_i8_tile); break;
            case  TILEDB_UINT8: populate(dim,  my_u8_start,  my_u8_end,  my_u8_tile); break;
            case  TILEDB_INT16: populate(dim, my_i16_start, my_i16_end, my_i16_tile); break;
            case TILEDB_UINT16: populate(dim, my_u16_start, my_u16_end, my_u16_tile); break;
            case  TILEDB_INT32: populate(dim, my_i32_start, my_i32_end, my_i32_tile); break;
            case TILEDB_UINT32: populate(dim, my_u32_start, my_u32_end, my_u32_tile); break;
            case  TILEDB_INT64: populate(dim, my_i64_start, my_i64_end, my_i64_tile); break;
            case TILEDB_UINT64: populate(dim, my_u64_start, my_u64_end, my_u64_tile); break;

            // I have no appetite to support the floating-point or date/time index types.
            default: throw std::runtime_error("unknown TileDB datatype '" + std::to_string(my_type) + "'");
        }
    }

    tiledb_datatype_t type() const {
        return my_type;
    }

public:
    template<typename Index_>
    Index_ extent() const {
        Index_ output = 0;
        switch (my_type) {
            case   TILEDB_INT8: compute_delta( my_i8_start,  &my_i8_end, 1, &output); break;
            case  TILEDB_UINT8: compute_delta( my_u8_start,  &my_u8_end, 1, &output); break;
            case  TILEDB_INT16: compute_delta(my_i16_start, &my_i16_end, 1, &output); break;
            case TILEDB_UINT16: compute_delta(my_u16_start, &my_u16_end, 1, &output); break;
            case  TILEDB_INT32: compute_delta(my_i32_start, &my_i32_end, 1, &output); break;
            case TILEDB_UINT32: compute_delta(my_u32_start, &my_u32_end, 1, &output); break;
            case  TILEDB_INT64: compute_delta(my_i64_start, &my_i64_end, 1, &output); break;
            case TILEDB_UINT64: compute_delta(my_u64_start, &my_u64_end, 1, &output); break;
            default: break;
        }
        return output + 1;
    }

    template<typename Index_>
    Index_ tile() const {
        Index_ output = 0;
        switch (my_type) {
            case   TILEDB_INT8: output =  my_i8_tile; break;
            case  TILEDB_UINT8: output =  my_u8_tile; break;
            case  TILEDB_INT16: output = my_i16_tile; break;
            case TILEDB_UINT16: output = my_u16_tile; break;
            case  TILEDB_INT32: output = my_i32_tile; break;
            case TILEDB_UINT32: output = my_u32_tile; break;
            case  TILEDB_INT64: output = my_i64_tile; break;
            case TILEDB_UINT64: output = my_u64_tile; break;
            default: break;
        }
        return output;
    }

    template<typename Index_>
    void add_range(tiledb::Subarray& subarray, int dim, Index_ start, Index_ length) const {
        switch (my_type) {
            case   TILEDB_INT8: add(subarray, dim,  my_i8_start, start, length); break;
            case  TILEDB_UINT8: add(subarray, dim,  my_u8_start, start, length); break;
            case  TILEDB_INT16: add(subarray, dim, my_i16_start, start, length); break;
            case TILEDB_UINT16: add(subarray, dim, my_u16_start, start, length); break;
            case  TILEDB_INT32: add(subarray, dim, my_i32_start, start, length); break;
            case TILEDB_UINT32: add(subarray, dim, my_u32_start, start, length); break;
            case  TILEDB_INT64: add(subarray, dim, my_i64_start, start, length); break;
            case TILEDB_UINT64: add(subarray, dim, my_u64_start, start, length); break;
            default: break;
        }
    }

    template<typename Index_, typename T>
    void correct_indices(const T* val, std::size_t len, Index_* output) const {
        if constexpr(std::is_same<T,   std::int8_t>::value) { compute_delta( my_i8_start, val, len, output); return; }
        if constexpr(std::is_same<T,  std::uint8_t>::value) { compute_delta( my_u8_start, val, len, output); return; }
        if constexpr(std::is_same<T,  std::int16_t>::value) { compute_delta(my_i16_start, val, len, output); return; }
        if constexpr(std::is_same<T, std::uint16_t>::value) { compute_delta(my_u16_start, val, len, output); return; }
        if constexpr(std::is_same<T,  std::int32_t>::value) { compute_delta(my_i32_start, val, len, output); return; }
        if constexpr(std::is_same<T, std::uint32_t>::value) { compute_delta(my_u32_start, val, len, output); return; }
        if constexpr(std::is_same<T,  std::int64_t>::value) { compute_delta(my_i64_start, val, len, output); return; }
        if constexpr(std::is_same<T, std::uint64_t>::value) { compute_delta(my_u64_start, val, len, output); return; }
        throw std::runtime_error("unsupported type for index correction");
    }

    template<typename Index_, typename T>
    Index_ correct_index(T val) const {
        Index_ output;
        correct_indices(&val, 1, &output);
        return output;
    }

private:
    tiledb_datatype_t my_type = TILEDB_INT32;
    std::int8_t    my_i8_start,  my_i8_end,  my_i8_tile;
    std::uint8_t   my_u8_start,  my_u8_end,  my_u8_tile;
    std::int16_t  my_i16_start, my_i16_end, my_i16_tile;
    std::uint16_t my_u16_start, my_u16_end, my_u16_tile;
    std::int32_t  my_i32_start, my_i32_end, my_i32_tile;
    std::uint32_t my_u32_start, my_u32_end, my_u32_tile;
    std::int64_t  my_i64_start, my_i64_end, my_i64_tile;
    std::uint64_t my_u64_start, my_u64_end, my_u64_tile;

private:
    template<typename T>
    static void populate(const tiledb::Dimension& dim, T& start, T& end, T& tile) {
        auto dom = dim.domain<T>();
        start = dom.first;
        end = dom.second;
        tile = dim.tile_extent<T>();
    }

    template<typename Index_, typename T>
    void compute_delta(T start, const T* pos, std::size_t len, Index_* output) const {
        if constexpr(std::is_integral<T>::value && std::is_signed<T>::value) {
            if (start < 0) {
                for (decltype(len) i = 0; i < len; ++i) {
                    auto curpos = pos[i];
                    if (curpos >= 0) {
                        // Protect against overflow as the difference between a non-negative and negative value might exceed 'T' (or its automatically promoted equivalent).
                        // We expect that the theoretical value of 'curpos - start' could fit in an Index_, as it must be storable in 'output'.
                        // This implies that 'curpos' could fit in an Index_, as 'curpos' is smaller than 'curpos - start' for 'curpos >= 0, start < 0'.
                        // Similarly, '-start' could fit in an Index_, though we compute '-(start + 1)' to avoid overflow, e.g., -128 => 127 for a signed 8-bit integer.
                        // Note that we need to +1 the sum to counteract the effect of adding 1 to 'start' before negation.
                        output[i] = static_cast<Index_>(curpos) + static_cast<Index_>(-(start + 1)) + 1;
                    } else {
                        output[i] = curpos - start;
                    }
                }
                return;
            }
        }

        // All elements of pos should be greater than or equal to 'start', so if 'start >= 0',
        // then 'pos[i] >= 0' and we can freely subtract without worrying about overflow.
        for (decltype(len) i = 0; i < len; ++i) {
            output[i] = pos[i] - start;
        }
    }

    template<typename T, typename Index_>
    void add(tiledb::Subarray& subarray, int dim, T domain_start, Index_ range_start, Index_ range_length) const {
        // range_length had better be positive!
        Index_ range_end = range_start + range_length - 1;

        if constexpr(!std::is_signed<T>::value && std::is_signed<Index_>::value) {
            // Forcibly cast to an unsigned type to ensure that we get predictable integer promotion to the larger unsigned type during arithmetic.
            // This is allowed as range_start and range_length are guaranteed to be non-negative.
            typedef typename std::make_unsigned<Index_>::type UIndex;
            subarray.add_range<T>(dim, domain_start + static_cast<UIndex>(range_start), domain_start + static_cast<UIndex>(range_end));
            return;
        }

        if constexpr(std::is_signed<T>::value && !std::is_signed<Index_>::value) {
            if (domain_start < 0) {
                // Here, some care is required as we can't coerce domain_start to an unsigned Index_,
                // nor can we coerce range_end to T as the latter might be too small.
                T new_end = safe_negative_add(domain_start, range_end);
                T new_start = safe_negative_add(domain_start, range_start);
                subarray.add_range<T>(dim, new_start, new_end);
                return;
            } else {
                // If domain_start is non-negative, the range of valid range_start and range_end is never to the right of domain_end.
                // So if T was large enough to fit domain_end, it is damn well large enough to fit range_end.
                subarray.add_range<T>(dim, domain_start + static_cast<T>(range_start), domain_start + static_cast<T>(range_end));
                return;
            }
        }

        // At this point, we're dealing with types of the same signedness, so we just let auto-promotion pick the larger type.
        // This can't overflow for valid range_* because any addition cannot exceed domain_end that is known to fit into T.
        subarray.add_range<T>(dim, domain_start + range_start, domain_start + range_end);
    }

    template<typename T, typename Index_>
    static T safe_negative_add(T l, Index_ r) {
        // The general principle here is to get us to the point where we're dealing with two unsigned integers that can be used in comparisons and arithmetic without surprises.
        typename std::make_unsigned<T>::type ul = -(l + 1); // +1 to avoid overflow when flipping the sign of a negative value (e.g., -128 to 127 for an 8-bit signed integer).
        if (ul < r) {
            return r - ul - 1;
        } else {
            // Coercing it back to a signed value before negation.
            return -static_cast<T>(ul - r + 1);
        }
    }
};

inline std::size_t determine_type_size(tiledb_datatype_t type) {
    std::size_t size = 0;
    switch (type) {
        case    TILEDB_INT8: size = 1; break;
        case   TILEDB_UINT8: size = 1; break;  
        case   TILEDB_INT16: size = 2; break; 
        case  TILEDB_UINT16: size = 2; break; 
        case   TILEDB_INT32: size = 4; break; 
        case  TILEDB_UINT32: size = 4; break; 
        case   TILEDB_INT64: size = 8; break; 
        case  TILEDB_UINT64: size = 8; break; 
        case TILEDB_FLOAT32: size = 4; break; 
        case TILEDB_FLOAT64: size = 8; break; 
        default: throw std::runtime_error("unknown TileDB datatype '" + std::to_string(type) + "'");
    }
    return size;
}

// Handling conversion from TileDB's storage type to our desired in-memory
// type - unlike HDF5, TileDB doesn't do this for us. 
class VariablyTypedVector {
public:
    VariablyTypedVector() = default;

    VariablyTypedVector(tiledb_datatype_t type, std::size_t len) {
        reset(type, len);
    }
    
public:
    void reset(tiledb_datatype_t type, std::size_t len) {
        my_type = type;
        switch (my_type) {
            case TILEDB_INT8:     my_i8.resize(sanisizer::cast<decltype( my_i8.size())>(len));   break;
            case TILEDB_UINT8:    my_u8.resize(sanisizer::cast<decltype( my_u8.size())>(len));   break;
            case TILEDB_INT16:   my_i16.resize(sanisizer::cast<decltype(my_i16.size())>(len));  break;
            case TILEDB_UINT16:  my_u16.resize(sanisizer::cast<decltype(my_u16.size())>(len));  break;
            case TILEDB_INT32:   my_i32.resize(sanisizer::cast<decltype(my_i32.size())>(len));  break;
            case TILEDB_UINT32:  my_u32.resize(sanisizer::cast<decltype(my_u32.size())>(len));  break;
            case TILEDB_INT64:   my_i64.resize(sanisizer::cast<decltype(my_i64.size())>(len));  break;
            case TILEDB_UINT64:  my_u64.resize(sanisizer::cast<decltype(my_u64.size())>(len));  break;
            case TILEDB_FLOAT32: my_f32.resize(sanisizer::cast<decltype(my_f32.size())>(len));  break;
            case TILEDB_FLOAT64: my_f64.resize(sanisizer::cast<decltype(my_f64.size())>(len));  break;
            default: throw std::runtime_error("unknown TileDB datatype '" + std::to_string(type) + "'");
        }
    }

public:
    void set_data_buffer(tiledb::Query& query, const std::string& name, std::size_t offset, std::size_t len) {
        switch (my_type) {
            case TILEDB_INT8:    query.set_data_buffer(name,  my_i8.data() + offset, len); break;
            case TILEDB_UINT8:   query.set_data_buffer(name,  my_u8.data() + offset, len); break;
            case TILEDB_INT16:   query.set_data_buffer(name, my_i16.data() + offset, len); break;
            case TILEDB_UINT16:  query.set_data_buffer(name, my_u16.data() + offset, len); break;
            case TILEDB_INT32:   query.set_data_buffer(name, my_i32.data() + offset, len); break;
            case TILEDB_UINT32:  query.set_data_buffer(name, my_u32.data() + offset, len); break;
            case TILEDB_INT64:   query.set_data_buffer(name, my_i64.data() + offset, len); break;
            case TILEDB_UINT64:  query.set_data_buffer(name, my_u64.data() + offset, len); break;
            case TILEDB_FLOAT32: query.set_data_buffer(name, my_f32.data() + offset, len); break;
            case TILEDB_FLOAT64: query.set_data_buffer(name, my_f64.data() + offset, len); break;
            default: break;
        }
    }

    template<typename Value_>
    void copy(std::size_t offset, std::size_t len, Value_* dest) const {
        switch (my_type) {
            case TILEDB_INT8:    std::copy_n( my_i8.begin() + offset, len, dest); break;
            case TILEDB_UINT8:   std::copy_n( my_u8.begin() + offset, len, dest); break;
            case TILEDB_INT16:   std::copy_n(my_i16.begin() + offset, len, dest); break;
            case TILEDB_UINT16:  std::copy_n(my_u16.begin() + offset, len, dest); break;
            case TILEDB_INT32:   std::copy_n(my_i32.begin() + offset, len, dest); break;
            case TILEDB_UINT32:  std::copy_n(my_u32.begin() + offset, len, dest); break;
            case TILEDB_INT64:   std::copy_n(my_i64.begin() + offset, len, dest); break;
            case TILEDB_UINT64:  std::copy_n(my_u64.begin() + offset, len, dest); break;
            case TILEDB_FLOAT32: std::copy_n(my_f32.begin() + offset, len, dest); break;
            case TILEDB_FLOAT64: std::copy_n(my_f64.begin() + offset, len, dest); break;
            default: break;
        }
    }

    void shift(std::size_t from, std::size_t len, std::size_t to) {
        switch (my_type) {
            case TILEDB_INT8:    std::copy_n( my_i8.begin() + from, len,  my_i8.begin() + to); break;
            case TILEDB_UINT8:   std::copy_n( my_u8.begin() + from, len,  my_u8.begin() + to); break;
            case TILEDB_INT16:   std::copy_n(my_i16.begin() + from, len, my_i16.begin() + to); break;
            case TILEDB_UINT16:  std::copy_n(my_u16.begin() + from, len, my_u16.begin() + to); break;
            case TILEDB_INT32:   std::copy_n(my_i32.begin() + from, len, my_i32.begin() + to); break;
            case TILEDB_UINT32:  std::copy_n(my_u32.begin() + from, len, my_u32.begin() + to); break;
            case TILEDB_INT64:   std::copy_n(my_i64.begin() + from, len, my_i64.begin() + to); break;
            case TILEDB_UINT64:  std::copy_n(my_u64.begin() + from, len, my_u64.begin() + to); break;
            case TILEDB_FLOAT32: std::copy_n(my_f32.begin() + from, len, my_f32.begin() + to); break;
            case TILEDB_FLOAT64: std::copy_n(my_f64.begin() + from, len, my_f64.begin() + to); break;
            default: break;
        }
    }

    template<typename Value_>
    void copy(std::size_t offset, std::size_t len, const VariablyTypedDimension& dim, Value_* dest) const {
        switch (my_type) {
            case   TILEDB_INT8: dim.correct_indices( my_i8.data() + offset, len, dest); break;
            case  TILEDB_UINT8: dim.correct_indices( my_u8.data() + offset, len, dest); break;
            case  TILEDB_INT16: dim.correct_indices(my_i16.data() + offset, len, dest); break;
            case TILEDB_UINT16: dim.correct_indices(my_u16.data() + offset, len, dest); break;
            case  TILEDB_INT32: dim.correct_indices(my_i32.data() + offset, len, dest); break;
            case TILEDB_UINT32: dim.correct_indices(my_u32.data() + offset, len, dest); break;
            case  TILEDB_INT64: dim.correct_indices(my_i64.data() + offset, len, dest); break;
            case TILEDB_UINT64: dim.correct_indices(my_u64.data() + offset, len, dest); break;
            default: throw std::runtime_error("unsupported type for copying with index correction");
        }
    }

    template<typename Index_>
    void compact(std::size_t from, std::size_t len, const VariablyTypedDimension& dim, std::vector<std::pair<Index_, Index_> >& counts) const {
        switch (my_type) {
            case   TILEDB_INT8: compact_internal( my_i8, from, len, dim, counts); break;
            case  TILEDB_UINT8: compact_internal( my_u8, from, len, dim, counts); break;
            case  TILEDB_INT16: compact_internal(my_i16, from, len, dim, counts); break;
            case TILEDB_UINT16: compact_internal(my_u16, from, len, dim, counts); break;
            case  TILEDB_INT32: compact_internal(my_i32, from, len, dim, counts); break;
            case TILEDB_UINT32: compact_internal(my_u32, from, len, dim, counts); break;
            case  TILEDB_INT64: compact_internal(my_i64, from, len, dim, counts); break;
            case TILEDB_UINT64: compact_internal(my_u64, from, len, dim, counts); break;
            default: throw std::runtime_error("unsupported type for index compaction");
        }
    }

private:
    tiledb_datatype_t my_type = TILEDB_INT32;

    std::vector<  std::int8_t> my_i8;
    std::vector< std::uint8_t> my_u8;
    std::vector< std::int16_t> my_i16;
    std::vector<std::uint16_t> my_u16;
    std::vector< std::int32_t> my_i32;
    std::vector<std::uint32_t> my_u32;
    std::vector< std::int64_t> my_i64;
    std::vector<std::uint64_t> my_u64;
    std::vector<        float> my_f32;
    std::vector<       double> my_f64;

    template<typename T, typename Index_>
    void compact_internal(const std::vector<T>& vals, std::size_t from, std::size_t len, const VariablyTypedDimension& dim, std::vector<std::pair<Index_, Index_> >& counts) const {
        counts.clear();
        std::size_t end = from + len;
        while (from < end) {
            T last = vals[from];
            Index_ count = 1;
            ++from;
            for (; from < end && last == vals[from]; ++from) {
                ++count;
            }
            counts.emplace_back(dim.correct_index<Index_>(last), count);
        }
    }
};

}

}

#endif
