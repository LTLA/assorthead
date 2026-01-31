#ifndef KNNCOLLE_HNSW_DISTANCES_HPP
#define KNNCOLLE_HNSW_DISTANCES_HPP

#include <functional>
#include <cstddef>
#include <type_traits>

/**
 * @file distances.hpp
 * @brief Distance classes for HNSW.
 */

namespace knncolle_hnsw {

/**
 * Methods for distance normalization.
 * 
 * - `SQRT` applies a square root, typically in conjunction with `hnswlib::L2Space` or `SquaredEuclideanDistance` to obtain the Euclidean distance.
 * - `CUSTOM` applies a custom normalization function in `DistanceConfig::custom_normalize`.
 * - `NONE` does not apply any normalization of the distances.
 */
enum class DistanceNormalizeMethod : char { SQRT, CUSTOM, NONE };

/**
 * @brief Distance configuration for the HNSW index.
 *
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam HnswData_ Floating-point type for data in the HNSW index.
 */
template<typename Distance_, typename HnswData_ = float>
struct DistanceConfig {
    /**
     * Function that returns a pointer to a `hnswlib::SpaceInterface` instance, given the number of dimensions.
     */
    std::function<hnswlib::SpaceInterface<HnswData_>*(std::size_t)> create;

    /**
     * Method for normalizing distances.
     */
    DistanceNormalizeMethod normalize_method;

    /**
     * Normalization function to convert the output of `hnswlib::SpaceInterface::get_dist_func()` into actual distances.
     * This must be provided if `normalize_method = DistanceNormalizeMethod::CUSTOM`, otherwise it is ignored.
     */
    std::function<Distance_(Distance_)> custom_normalize;
};

/**
 * @brief Manhattan distance. 
 *
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 */
template<typename HnswData_ = float>
class ManhattanDistance final : public hnswlib::SpaceInterface<HnswData_> {
private:
    std::size_t my_data_size;
    std::size_t my_dim;

public:
    /**
     * @param dim Number of dimensions over which to compute the distance.
     */
    ManhattanDistance(std::size_t dim) : my_data_size(dim * sizeof(HnswData_)), my_dim(dim) {}

    /**
     * @cond
     */
public:
    std::size_t get_data_size() {
        return my_data_size;
    }

    hnswlib::DISTFUNC<HnswData_> get_dist_func() {
        return L1;
    }

    void * get_dist_func_param() {
        return &my_dim;
    }

private:
    static HnswData_ L1(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const HnswData_* pVect1 = static_cast<const HnswData_*>(pVect1v);
        const HnswData_* pVect2 = static_cast<const HnswData_*>(pVect2v);
        std::size_t qty = *((size_t *) qty_ptr);
        HnswData_ res = 0;
        for (; qty > 0; --qty, ++pVect1, ++pVect2) {
            res += std::abs(*pVect1 - *pVect2);
        }
        return res;
    }
    /**
     * @endcond
     */
};

/**
 * @brief Squared Euclidean distance. 
 *
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 */
template<typename HnswData_ = float>
class SquaredEuclideanDistance final : public hnswlib::SpaceInterface<HnswData_> {
private:
    std::size_t my_data_size;
    std::size_t my_dim;

public:
    /**
     * @param dim Number of dimensions over which to compute the distance.
     */
    SquaredEuclideanDistance(std::size_t dim) : my_data_size(dim * sizeof(HnswData_)), my_dim(dim) {}

    /**
     * @cond
     */
public:
    std::size_t get_data_size() {
        return my_data_size;
    }

    hnswlib::DISTFUNC<HnswData_> get_dist_func() {
        return L2;
    }

    void * get_dist_func_param() {
        return &my_dim;
    }

private:
    static HnswData_ L2(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const HnswData_* pVect1 = static_cast<const HnswData_*>(pVect1v);
        const HnswData_* pVect2 = static_cast<const HnswData_*>(pVect2v);
        std::size_t qty = *((size_t *) qty_ptr);
        HnswData_ res = 0;
        for (; qty > 0; --qty, ++pVect1, ++pVect2) {
            auto delta = *pVect1 - *pVect2;
            res += delta * delta;
        }
        return res;
    }
    /**
     * @endcond
     */
};

/**
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 * @return Configuration for using Euclidean distances in the HNSW index.
 * `DistanceConfig::create` is set to `hnswlib::L2Space` if `HnswData_ = float`, otherwise it is set to `SquaredEuclideanDistance`.
 */
template<typename Distance_, typename HnswData_ = float>
DistanceConfig<Distance_, HnswData_> configure_euclidean_distance() {
    DistanceConfig<Distance_, HnswData_> output;
    output.create = [](std::size_t dim) -> hnswlib::SpaceInterface<HnswData_>* {
        if constexpr(std::is_same<HnswData_, float>::value) {
            return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new hnswlib::L2Space(dim));
        } else {
            return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new SquaredEuclideanDistance<HnswData_>(dim));
        }
    };
    output.normalize_method = DistanceNormalizeMethod::SQRT;
    return output;
}

/**
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 * @return Configuration for using Manhattan distances in the HNSW index.
 */
template<typename Distance_, typename HnswData_ = float>
DistanceConfig<Distance_, HnswData_> configure_manhattan_distance() {
    DistanceConfig<Distance_, HnswData_> output;
    output.create = [](std::size_t dim) -> hnswlib::SpaceInterface<HnswData_>* {
        return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new knncolle_hnsw::ManhattanDistance<HnswData_>(dim));
    };
    output.normalize_method = DistanceNormalizeMethod::NONE;
    return output;
}

/**
 * @cond
 */
// Preserved for back-compatibility.
template<typename HnswData_ = float, typename Distance_ = double>
DistanceConfig<Distance_, HnswData_> makeEuclideanDistanceConfig() {
    return configure_euclidean_distance<Distance_, HnswData_>();
}

template<typename HnswData_ = float, typename Distance_ = double>
DistanceConfig<Distance_, HnswData_> makeManhattanDistanceConfig() {
    return configure_manhattan_distance<Distance_, HnswData_>();
}
/**
 * @endcond
 */

/**
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 * @param distance Pointer to a distance metric.
 *
 * @return String containing a name for each known metric:
 * 
 * - `"l2"` for `hnswlib::L2Space`.
 * - `"squared_euclidean"` for `SquaredEuclideanDistance`.
 * - `"manhattan"` for `ManhattanDistance`.
 * - otherwise, `"unknown"`.
 */
template<typename HnswData_>
const char* get_distance_name(const hnswlib::SpaceInterface<HnswData_>* distance) {
    if (dynamic_cast<const hnswlib::L2Space*>(distance) != NULL) {
        return "l2";
    } else if (dynamic_cast<const SquaredEuclideanDistance<HnswData_>*>(distance) != NULL) {
        return "squared_euclidean";
    } else if (dynamic_cast<const ManhattanDistance<HnswData_>*>(distance) != NULL) {
        return "manhattan";
    } else {
        return "unknown";
    }
}

}

#endif
