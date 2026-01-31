#ifndef KNNCOLLE_KMKNN_LOAD_KMKNN_PREBUILT_HPP
#define KNNCOLLE_KMKNN_LOAD_KMKNN_PREBUILT_HPP

#include "knncolle/knncolle.hpp"

#include <string>

/**
 * @file load_kmknn_prebuilt.hpp
 * @brief Load a prebuilt KMKNN index from disk.
 */

namespace knncolle_kmknn {

/**
 * @brief Template types of a saved KMKNN index.
 *
 * Instances are typically created by `load_kmknn_prebuilt_types()`.
 */
struct KmknnPrebuiltTypes {
    /**
     * Type of the k-means centroids, i.e., `KmeansFloat_` in `KmknnBuilder()`.
     */
    knncolle::NumericType kmeansfloat;
};

/**
 * @param dir Path to a directory in which a prebuilt KMKNN index was saved.
 * An Kmknn index would typically be saved by calling the `knncolle::Prebuilt::save()` method of the Kmknn subclass instance.
 *
 * @return Template types of the saved instance of a `knncolle::Prebuilt` KMKNN subclass.
 * This is typically used to choose template parameters for `load_kmknn_prebuilt()`.
 */
inline KmknnPrebuiltTypes load_kmknn_prebuilt_types(const std::filesystem::path& dir) {
    knncolle::NumericType type;
    knncolle::quick_load(dir / "FLOAT_TYPE", &type, 1);

    KmknnPrebuiltTypes config;
    config.kmeansfloat = type;

    return config;
}

/**
 * Helper function to define a `knncolle::LoadPrebuiltFunction` for KMKNN in `knncolle::load_prebuilt_raw()`.
 *
 * To load an KMKNN index from disk, users are expected to define and register an KMKNN-specific `knncolle::LoadPrebuiltFunction`.
 * In this function, users should call `load_kmknn_prebuilt_types()` to figure out the saved index's `KmknnFloat_`.
 * Then, they should call `load_kmknn_prebuilt()` with the appropriate types to return a pointer to a `knncolle::Prebuilt` object.
 * This user-defined function should be registered in `load_prebuilt_registry()` with the key in `knncolle_kmknn::kmknn_prebuilt_save_name`.
 * 
 * For unknown types, users can set `custom_save_for_kmknn_kmeansfloat()`.
 * Each custom function saves additional information about its type to disk during a `knncolle::Prebuilt::save()` call.
 * That information can then be parsed in the user-defined `knncolle::LoadPrebuiltFunction` to recover an KMKNN index with the appropriate template types.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam DistanceMetricData_ Class implementing the calculation of distances between observations.
 * This should satisfy the `knncolle::DistanceMetric` interface.
 * @tparam KmeansFloat_ Floating-point type of the cluster centroids.
 * @tparam DistanceMetricCenter_ Class implementing the calculation of distances between an observation and a cluster centroid.
 * This should satisfy the `knncolle::DistanceMetric` interface.
 *
 * @param dir Path to a directory in which a prebuilt KMKNN index was saved.
 * An KMKNN index would typically be saved by calling the `knncolle::Prebuilt::save()` method of the KMKNN subclass instance.
 *
 * @return Pointer to a `knncolle::Prebuilt` KMKNN index.
 */
template<
    typename Index_,
    typename Data_,
    typename Distance_,
    class DistanceMetricData_ = knncolle::DistanceMetric<Data_, Distance_>,
    typename KmeansFloat_ = Distance_,
    class DistanceMetricCenter_ = knncolle::DistanceMetric<KmeansFloat_, Distance_>
>
auto load_kmknn_prebuilt(const std::filesystem::path& dir) {
    return new KmknnPrebuilt<
        Index_,
        Data_,
        Distance_,
        DistanceMetricData_,
        KmeansFloat_,
        DistanceMetricCenter_
    >(dir);
}

}

#endif
