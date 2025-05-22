#ifndef KNNCOLLE_HNSW_DISTANCES_HPP
#define KNNCOLLE_HNSW_DISTANCES_HPP

#include <cmath>
#include <functional>
#include <cstddef>

/**
 * @file distances.hpp
 * @brief Distance classes for HNSW.
 */

namespace knncolle_hnsw {

/**
 * @brief Distance configuration for the HNSW index.
 *
 * @tparam HnswData_ Floating-point type for data in the HNSW index.
 */
template<typename HnswData_ = float>
struct DistanceConfig {
    /**
     * Create a `hnswlib::SpaceInterface` object, given the number of dimensions.
     */
    std::function<hnswlib::SpaceInterface<HnswData_>*(std::size_t)> create;

    /**
     * Normalization function to convert distance measures from `hnswlib::SpaceInterface::get_dist_func()` into actual distances.
     * If not provided , this defaults to a no-op.
     */
    std::function<HnswData_(HnswData_)> normalize;
};

/**
 * @brief Manhattan distance. 
 *
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 */
template<typename HnswData_ = float>
class ManhattanDistance : public hnswlib::SpaceInterface<HnswData_> {
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
class SquaredEuclideanDistance : public hnswlib::SpaceInterface<HnswData_> {
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
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 * @return Configuration for using Euclidean distances in the HNSW index.
 * `DistanceConfig::create` is set to `hnswlib::L2Space` if `HnswData_ = float`, otherwise it is set to `SquaredEuclideanDistance`.
 */
template<typename HnswData_ = float>
DistanceConfig<HnswData_> makeEuclideanDistanceConfig() {
    DistanceConfig<HnswData_> output;
    output.create = [](std::size_t dim) -> hnswlib::SpaceInterface<HnswData_>* {
        if constexpr(std::is_same<HnswData_, float>::value) {
            return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new hnswlib::L2Space(dim));
        } else {
            return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new SquaredEuclideanDistance<HnswData_>(dim));
        }
    };
    output.normalize = [](HnswData_ x) -> HnswData_ {
        return std::sqrt(x);
    };
    return output;
}

/**
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 * @return Configuration for using Manhattan distances in the HNSW index.
 */
template<typename HnswData_ = float>
DistanceConfig<HnswData_> makeManhattanDistanceConfig() {
    DistanceConfig<HnswData_> output;
    output.create = [](std::size_t dim) -> hnswlib::SpaceInterface<HnswData_>* {
        return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new knncolle_hnsw::ManhattanDistance<HnswData_>(dim));
    };
    return output;
}

}

#endif
