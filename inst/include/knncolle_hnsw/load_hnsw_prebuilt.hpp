#ifndef KNNCOLLE_HNSW_LOAD_PREBUILT_HPP
#define KNNCOLLE_HNSW_LOAD_PREBUILT_HPP

#include "utils.hpp"
#include "Hnsw.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "knncolle/knncolle.hpp"

/**
 * @file load_hnsw_prebuilt.hpp
 * @brief Load a prebuilt HNSW index.
 */

namespace knncolle_hnsw {

/**
 * @brief Template types of a saved HNSW index.
 *
 * Instances are typically created by `load_hnsw_prebuilt_types()`.
 */
struct HnswPrebuiltTypes {
    /**
     * Type of the data, i.e., `HNSWData_` in `HNSWBuilder()`.
     */
    knncolle::NumericType data;
};

/**
 * @param dir Path to a directory in which a prebuilt HNSW index was saved.
 * An HNSW index is typically saved by calling the `knncolle::Prebuilt::save()` method of the HNSW subclass instance.
 *
 * @return Template types of the saved instance of a `knncolle::Prebuilt` HNSW subclass.
 * This is typically used to choose template parameters for `load_hnsw_prebuilt()`.
 */
inline HnswPrebuiltTypes load_hnsw_prebuilt_types(const std::filesystem::path& dir) {
    knncolle::NumericType type;
    knncolle::quick_load(dir / "TYPE", &type, 1);
    HnswPrebuiltTypes config;
    config.data = type;
    return config;
}

/**
 * Helper function to define a `knncolle::LoadPrebuiltFunction` for HNSW in `knncolle::load_prebuilt_raw()`.
 *
 * In an HNSW-specific `knncolle::LoadPrebuiltFunction`,
 * users should first call `load_hnsw_prebuilt_types()` to figure out the saved index's `HNSWData_`.
 * Then, they can call `load_hnsw_prebuilt()` with the specified types to return a pointer to a `knncolle::Prebuilt` object.
 * This can be registered in `knncolle::load_prebuilt_registry()` with the key in `knncolle_hnsw::hnsw_prebuilt_save_name`.
 * 
 * We do not define a default function for loading HNSW indices as there are too many possible combinations of types.
 * Instead, the user is responsible for deciding which combinations of types should be handled.
 * This avoids binary bloat from repeated instantiations of the HNSW template classes, if an application only deals with a certain subset of combinations. 
 * For types or distances that are unknown to `knncolle::get_numeric_type()` or `get_distance_name()`, respectively,
 * users can store additional information on disk via `custom_save_for_hnsw_data()` for use in loading.
 * 
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam HnswData_ Floating-point type for data in the HNSW index.
 * This should be the same as the type reported by `HnswPrebuiltTypes::data`.
 *
 * @param dir Path to a directory in which a prebuilt HNSW index was saved.
 * An HNSW index is typically saved by calling the `knncolle::Prebuilt::save()` method of the HNSW subclass instance.
 *
 * @return Pointer to a `knncolle::Prebuilt` HNSW index.
 */
template<typename Index_, typename Data_, typename Distance_, typename HnswData_ = float>
auto load_hnsw_prebuilt(const std::filesystem::path& dir) {
    return new HnswPrebuilt<Index_, Data_, Distance_, HnswData_>(dir);
}

}

#endif
