#ifndef KNNCOLLE_KMKNN_HPP
#define KNNCOLLE_KMKNN_HPP

#include "knncolle/knncolle.hpp"
#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <cstddef>

/**
 * @file Kmknn.hpp
 *
 * @brief Implements the k-means with k-nearest neighbors (KMKNN) algorithm.
 */

/**
 * @namespace knncolle_kmknn
 * @brief Namespace for the **knncolle_kmeans** library.
 */
namespace knncolle_kmknn {

/**
 * Common type for promoting `Data_` and `Distance_` during subtraction.
 * If `Data_` is integer, the common type will be the floating-point type of `Distance_`;
 * otherwise if both types are floating-point, the common type will be the higher-precision type.
 * This is used to define the common datatype for storing both the input data and the centroids,
 * so that the same `DistanceMetric` can be used to compute the distances to a query datapoint.
 *
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 */
template<typename Data_, typename Distance_>
using Common = decltype(std::declval<Data_>() - std::declval<Distance_>()); 

/**
 * Convenient alias for `knncolle::DistanceMetric` with the first template argument set to the `Common` type.
 *
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 */
template<typename Data_, typename Distance_>
using DistanceMetric = knncolle::DistanceMetric<Common<Data_, Distance_>, Distance_>;

/**
 * Convenient alias for `knncolle::EuclideanDistance` with the first template argument set to the `Common` type.
 *
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 */
template<typename Data_, typename Distance_>
using EuclideanDistance = knncolle::EuclideanDistance<Common<Data_, Distance_>, Distance_>;

/**
 * Convenient alias for `knncolle::ManhattanDistance` with the first template argument set to the `Common` type.
 *
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 */
template<typename Data_, typename Distance_>
using ManhattanDistance = knncolle::ManhattanDistance<Common<Data_, Distance_>, Distance_>;

/**
 * Convenient alias for `kmeans::Initialize`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using Initialize = kmeans::Initialize<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::InitializeRandom`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using InitializeRandom = kmeans::InitializeRandom<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::InitializeKmeanspp`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using InitializeKmeanspp = kmeans::InitializeKmeanspp<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::InitializeNone`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using InitializeNone = kmeans::InitializeNone<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::InitializeVariancePartition`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using InitializeVariancePartition = kmeans::InitializeVariancePartition<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::Refine`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using Refine = kmeans::Refine<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::RefineLloyd`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using RefineLloyd = kmeans::RefineLloyd<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/**
 * Convenient alias for `kmeans::RefineHartiganWong`, using the `Common` type for the input data and the distances/centroids.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
using RefineHartiganWong = kmeans::RefineHartiganWong<Index_, Common<Data_, Distance_>, Index_, Common<Data_, Distance_>, KmeansMatrix_>;

/** 
 * @brief Options for `KmknnBuilder` construction. 
 *
 * This can also be created via the `KmknnBuilder::Options` typedef,
 * which ensures consistency with the template parameters used in `KmknnBuilder`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 */
template<typename Index_, typename Data_, typename Distance_, class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
struct KmknnOptions {
    /**
     * Power of the number of observations, to define the number of cluster centers.
     * By default, a square root is performed.
     */
    double power = 0.5;

    /**
     * Initialization method for k-means clustering.
     * If NULL, defaults to `InitializeKmeanspp`.
     */
    std::shared_ptr<Initialize<Index_, Data_, Distance_, KmeansMatrix_> > initialize_algorithm;

    /**
     * Refinement method for k-means clustering.
     * If NULL, defaults to `RefineHartiganWong`.
     */
    std::shared_ptr<Refine<Index_, Data_, Distance_, KmeansMatrix_> > refine_algorithm;
};

template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class KmknnPrebuilt;

/**
 * @brief KMKNN searcher.
 *
 * Instances of this class are usually constructed using `KmknnPrebuilt::initialize()`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `knncolle::DistanceMetric` interface.
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_ = DistanceMetric<Data_, Distance_> >
class KmknnSearcher final : public knncolle::Searcher<Index_, Data_, Distance_> {
public:
    /**
     * @cond
     */
    KmknnSearcher(const KmknnPrebuilt<Index_, Data_, Distance_, DistanceMetric_>& parent) : my_parent(parent) {
        my_center_order.reserve(my_parent.my_sizes.size());
        if constexpr(needs_conversion) {
            my_conversion_buffer.resize(my_parent.my_dim);
        }
    }
    /**
     * @endcond
     */

private:                
    const KmknnPrebuilt<Index_, Data_, Distance_, DistanceMetric_>& my_parent;
    knncolle::NeighborQueue<Index_, Distance_> my_nearest;
    std::vector<std::pair<Distance_, Index_> > my_all_neighbors;
    std::vector<std::pair<Distance_, Index_> > my_center_order;

    // Converting input data to 'Center_'.
    static constexpr bool needs_conversion = !std::is_same<Common<Data_, Distance_>, Data_>::value;
    typename std::conditional<needs_conversion, std::vector<Common<Data_, Distance_> >, bool>::type my_conversion_buffer;

    const Common<Data_, Distance_> * sanitize(const Data_* query) {
        if constexpr(needs_conversion) {
            auto conv_buffer = my_conversion_buffer.data();
            std::copy_n(query, my_parent.my_dim, conv_buffer);
            return conv_buffer;
        } else {
            return query;
        }
    }

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_nearest.reset(k + 1);
        auto new_i = my_parent.my_new_location[i];
        auto iptr = my_parent.my_data.data() + static_cast<std::size_t>(new_i) * my_parent.my_dim; // cast to avoid overflow.
        my_parent.search_nn(iptr, my_nearest, my_center_order);
        my_nearest.report(output_indices, output_distances, new_i);
        my_parent.normalize(output_indices, output_distances);
    }

    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            if (output_indices) {
                output_indices->clear();
            }
            if (output_distances) {
                output_distances->clear();
            }
        } else {
            my_nearest.reset(k);
            my_parent.search_nn(sanitize(query), my_nearest, my_center_order);
            my_nearest.report(output_indices, output_distances);
            my_parent.normalize(output_indices, output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto new_i = my_parent.my_new_location[i];
        auto iptr = my_parent.my_data.data() + static_cast<std::size_t>(new_i) * my_parent.my_dim; // cast to avoid overflow.

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(iptr, d, count);
            return knncolle::count_all_neighbors_without_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(iptr, d, my_all_neighbors);
            knncolle::report_all_neighbors(my_all_neighbors, output_indices, output_distances, new_i);
            my_parent.normalize(output_indices, output_distances);
            return knncolle::count_all_neighbors_without_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Data_* query, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(sanitize(query), d, my_all_neighbors);
            knncolle::report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            my_parent.normalize(output_indices, output_distances);
            return my_all_neighbors.size();
        }
    }
};

/**
 * @brief Index for a KMKNN search.
 *
 * Instances of this class are usually constructed using `KmknnBuilder::build_raw()`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `knncolle::DistanceMetric` interface.
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_ = DistanceMetric<Data_, Distance_> >
class KmknnPrebuilt final : public knncolle::Prebuilt<Index_, Data_, Distance_> {
private:
    std::size_t my_dim;
    Index_ my_obs;

public:
    Index_ num_observations() const {
        return my_obs;
    }

    std::size_t num_dimensions() const {
        return my_dim;
    }

private:
    std::vector<Common<Data_, Distance_> > my_data;
    std::shared_ptr<const DistanceMetric_> my_metric;
    
    std::vector<Index_> my_sizes;
    std::vector<Index_> my_offsets;

    std::vector<Common<Data_, Distance_> > my_centers;

    std::vector<Index_> my_observation_id, my_new_location;
    std::vector<Distance_> my_dist_to_centroid;

public:
    /**
     * @cond
     */
    template<class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_> >
    KmknnPrebuilt(
        std::size_t num_dim,
        Index_ num_obs,
        std::vector<Common<Data_, Distance_> > data,
        std::shared_ptr<const DistanceMetric_> metric,
        const KmknnOptions<Index_, Data_, Distance_, KmeansMatrix_>& options
    ) :
        my_dim(num_dim),
        my_obs(num_obs),
        my_data(std::move(data)),
        my_metric(std::move(metric))
    { 
        auto init = options.initialize_algorithm;
        if (init == nullptr) {
            init.reset(new InitializeKmeanspp<Index_, Data_, Distance_, KmeansMatrix_>);
        }
        auto refine = options.refine_algorithm;
        if (refine == nullptr) {
            refine.reset(new RefineHartiganWong<Index_, Data_, Distance_, KmeansMatrix_>);
        }

        Index_ ncenters = std::ceil(std::pow(my_obs, options.power));
        my_centers.resize(static_cast<std::size_t>(ncenters) * my_dim); // cast to avoid overflow problems.

        kmeans::SimpleMatrix<Index_, Data_> mat(my_dim, my_obs, my_data.data());
        std::vector<Index_> clusters(my_obs);
        auto output = kmeans::compute(mat, *init, *refine, ncenters, my_centers.data(), clusters.data());

        // Removing empty clusters, e.g., due to duplicate points.
        {
            my_sizes.resize(ncenters);
            std::vector<Index_> remap(ncenters);
            Index_ survivors = 0;
            for (Index_ c = 0; c < ncenters; ++c) {
                if (output.sizes[c]) {
                    if (c > survivors) {
                        auto src = my_centers.begin() + static_cast<std::size_t>(c) * my_dim; // cast to avoid overflow.
                        auto dest = my_centers.begin() + static_cast<std::size_t>(survivors) * my_dim;
                        std::copy_n(src, my_dim, dest);
                    }
                    remap[c] = survivors;
                    my_sizes[survivors] = output.sizes[c];
                    ++survivors;
                }
            }

            if (survivors < ncenters) {
                for (auto& c : clusters) {
                    c = remap[c];
                }
                ncenters = survivors;
                my_centers.resize(static_cast<std::size_t>(ncenters) * my_dim);
                my_sizes.resize(ncenters);
            }
        }

        my_offsets.resize(ncenters);
        for (Index_ i = 1; i < ncenters; ++i) {
            my_offsets[i] = my_offsets[i - 1] + my_sizes[i - 1];
        }

        // Organize points correctly; firstly, sorting by distance from the assigned center.
        std::vector<std::pair<Distance_, Index_> > by_distance(my_obs);
        {
            auto sofar = my_offsets;
            auto host = my_data.data();
            for (Index_ o = 0; o < my_obs; ++o) {
                auto optr = host + static_cast<std::size_t>(o) * my_dim;
                auto clustid = clusters[o];
                auto cptr = my_centers.data() + static_cast<std::size_t>(clustid) * my_dim;

                auto& counter = sofar[clustid];
                auto& current = by_distance[counter];
                current.first = my_metric->normalize(my_metric->raw(my_dim, optr, cptr));
                current.second = o;

                ++counter;
            }

            for (Index_ c = 0; c < ncenters; ++c) {
                auto begin = by_distance.begin() + my_offsets[c];
                std::sort(begin, begin + my_sizes[c]);
            }
        }

        // Permuting in-place to mirror the reordered distances, so that the search is more cache-friendly.
        {
            auto host = my_data.data();
            std::vector<uint8_t> used(my_obs);
            std::vector<Data_> buffer(my_dim);
            my_observation_id.resize(my_obs);
            my_dist_to_centroid.resize(my_obs);
            my_new_location.resize(my_obs);

            for (Index_ o = 0; o < my_obs; ++o) {
                if (used[o]) {
                    continue;
                }

                const auto& current = by_distance[o];
                my_observation_id[o] = current.second;
                my_dist_to_centroid[o] = current.first;
                my_new_location[current.second] = o;
                if (current.second == o) {
                    continue;
                }

                // We recursively perform a "thread" of replacements until we
                // are able to find the home of the originally replaced 'o'.
                auto optr = host + static_cast<std::size_t>(o) * my_dim;
                std::copy_n(optr, my_dim, buffer.begin());
                Index_ replacement = current.second;
                do {
                    auto rptr = host + static_cast<std::size_t>(replacement) * my_dim;
                    std::copy_n(rptr, my_dim, optr);
                    used[replacement] = 1;

                    const auto& next = by_distance[replacement];
                    my_observation_id[replacement] = next.second;
                    my_dist_to_centroid[replacement] = next.first;
                    my_new_location[next.second] = replacement;

                    optr = rptr;
                    replacement = next.second;
                } while (replacement != o);

                std::copy(buffer.begin(), buffer.end(), optr);
            }
        }

        return;
    }
    /**
     * @endcond
     */

private:
    void search_nn(const Common<Data_, Distance_>* target, knncolle::NeighborQueue<Index_, Distance_>& nearest, std::vector<std::pair<Distance_, Index_> >& center_order) const { 
        /* Computing distances to all centers and sorting them. The aim is to
         * go through the nearest centers first, to try to get the shortest
         * threshold (i.e., 'nearest.limit()') possible at the start;
         * this allows us to skip searches of the later clusters.
         */
        center_order.clear();
        std::size_t ncenters = my_sizes.size();
        center_order.reserve(ncenters);
        auto clust_ptr = my_centers.data();
        for (std::size_t c = 0; c < ncenters; ++c, clust_ptr += my_dim) {
            center_order.emplace_back(my_metric->raw(my_dim, target, clust_ptr), c);
        }
        std::sort(center_order.begin(), center_order.end());

        // Computing the distance to each center, and deciding whether to proceed for each cluster.
        Distance_ threshold_raw = std::numeric_limits<Distance_>::infinity();
        for (const auto& curcent : center_order) {
            const Index_ center = curcent.second;
            const Distance_ dist2center = my_metric->normalize(curcent.first);

            const auto cur_nobs = my_sizes[center];
            const Distance_* dIt = my_dist_to_centroid.data() + my_offsets[center];
            const Distance_ maxdist = *(dIt + cur_nobs - 1);

            Index_ firstcell = 0;
#if KNNCOLLE_KMKNN_USE_UPPER
            Distance_ upper_bd = std::numeric_limits<Distance_>::max();
#endif

            if (!std::isinf(threshold_raw)) {
                const Distance_ threshold = my_metric->normalize(threshold_raw);

                /* The conditional expression below exploits the triangle inequality; it is equivalent to asking whether:
                 *     threshold + maxdist < dist2center
                 * All points (if any) within this cluster with distances above 'lower_bd' are potentially countable.
                 */
                const Distance_ lower_bd = dist2center - threshold;
                if (maxdist < lower_bd) {
                    continue;
                }

                firstcell = std::lower_bound(dIt, dIt + cur_nobs, lower_bd) - dIt;

#if KNNCOLLE_KMKNN_USE_UPPER
                /* This exploits the reverse triangle inequality, to ignore points where:
                 *     threshold + dist2center < point-to-center distance
                 */
                upper_bd = threshold + dist2center;
#endif
            }

            const auto cur_start = my_offsets[center];
            const auto* other_cell = my_data.data() + my_dim * static_cast<std::size_t>(cur_start + firstcell); // cast to avoid overflow issues.
            for (auto celldex = firstcell; celldex < cur_nobs; ++celldex, other_cell += my_dim) {
#if KNNCOLLE_KMKNN_USE_UPPER
                if (*(dIt + celldex) > upper_bd) {
                    break;
                }
#endif

                auto dist2cell_raw = my_metric->raw(my_dim, target, other_cell);
                if (dist2cell_raw <= threshold_raw) {
                    nearest.add(cur_start + celldex, dist2cell_raw);
                    if (nearest.is_full()) {
                        threshold_raw = nearest.limit(); // Shrinking the threshold, if an earlier NN has been found.
#if KNNCOLLE_KMKNN_USE_UPPER
                        upper_bd = my_metric->normalize(threshold_raw) + dist2center; 
#endif
                    }
                }
            }
        }
    }

    template<bool count_only_, typename Output_>
    void search_all(const Data_* target, Distance_ threshold, Output_& all_neighbors) const {
        Distance_ threshold_raw = my_metric->denormalize(threshold);

        /* Computing distances to all centers. We don't sort them here 
         * because the threshold is constant so there's no point.
         */
        Index_ ncenters = my_sizes.size();
        auto center_ptr = my_centers.data(); 
        for (Index_ center = 0; center < ncenters; ++center, center_ptr += my_dim) {
            const Distance_ dist2center = my_metric->normalize(my_metric->raw(my_dim, target, center_ptr));

            auto cur_nobs = my_sizes[center];
            const Distance_* dIt = my_dist_to_centroid.data() + my_offsets[center];
            const Distance_ maxdist = *(dIt + cur_nobs - 1);

            /* The conditional expression below exploits the triangle inequality; it is equivalent to asking whether:
             *     threshold + maxdist < dist2center
             * All points (if any) within this cluster with distances above 'lower_bd' are potentially countable.
             */
            const Distance_ lower_bd = dist2center - threshold;
            if (maxdist < lower_bd) {
                continue;
            }

            Index_ firstcell = std::lower_bound(dIt, dIt + cur_nobs, lower_bd) - dIt;
#if KNNCOLLE_KMKNN_USE_UPPER
            /* This exploits the reverse triangle inequality, to ignore points where:
             *     threshold + dist2center < point-to-center distance
             */
            Distance_ upper_bd = threshold + dist2center;
#endif

            const auto cur_start = my_offsets[center];
            auto other_ptr = my_data.data() + my_dim * static_cast<std::size_t>(cur_start + firstcell); // cast to avoid overflow issues.
            for (auto celldex = firstcell; celldex < cur_nobs; ++celldex, other_ptr += my_dim) {
#if KNNCOLLE_KMKNN_USE_UPPER
                if (*(dIt + celldex) > upper_bd) {
                    break;
                }
#endif

                auto dist2cell_raw = my_metric->raw(my_dim, target, other_ptr);
                if (dist2cell_raw <= threshold_raw) {
                    if constexpr(count_only_) {
                        ++all_neighbors;
                    } else {
                        all_neighbors.emplace_back(dist2cell_raw, cur_start + celldex);
                    }
                }
            }
        }
    }

    void normalize(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) const {
        if (output_indices) {
            for (auto& s : *output_indices) {
                s = my_observation_id[s];
            }
        }
        if (output_distances) {
            for (auto& d : *output_distances) {
                d = my_metric->normalize(d);
            }
        }
    }

    friend class KmknnSearcher<Index_, Data_, Distance_, DistanceMetric_>;

public:
    /**
     * Creates a `KmknnSearcher` instance.
     */
    std::unique_ptr<knncolle::Searcher<Index_, Data_, Distance_> > initialize() const {
        return std::make_unique<KmknnSearcher<Index_, Data_, Distance_, DistanceMetric_> >(*this);
    }
};

/**
 * @brief Perform a nearest neighbor search based on k-means clustering.
 *
 * In the k-means with k-nearest neighbors (KMKNN) algorithm (Wang, 2012), k-means clustering is first applied to the data points,
 * with the number of cluster centers defined as the square root of the number of points.
 * The cluster assignment and distance to the assigned cluster center for each point represent the KMKNN indexing information, 
 * allowing us to speed up the nearest neighbor search by exploiting the triangle inequality between cluster centers, the query point and each point in the cluster to narrow the search space.
 * The advantage of the KMKNN approach is its simplicity and minimal overhead,
 * resulting in performance improvements over conventional tree-based methods for high-dimensional data where most points need to be searched anyway.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * This is also used for the cluster centers.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `knncolle::Matrix` interface.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `knncolle::DistanceMetric` interface.
 * Note that the input data type is set to `Distance_` as this class must be able to compute distances to cluster centers.
 * @tparam KmeansMatrix_ Class of the input data matrix for **kmeans**.
 * This should satisfy the `kmeans::Matrix` interface, most typically a `kmeans::SimpleMatrix`.
 * (Note that this is a different class from the `knncolle::Matrix` interface!)
 *
 * @see
 * Wang X (2012). 
 * A fast exact k-nearest neighbors algorithm for high dimensional search using k-means clustering and triangle inequality. 
 * _Proc Int Jt Conf Neural Netw_, 43, 6:2351-2358.
 */
template<
    typename Index_,
    typename Data_,
    typename Distance_,
    class Matrix_ = knncolle::Matrix<Index_, Data_>,
    class DistanceMetric_ = DistanceMetric<Data_, Distance_>,
    class KmeansMatrix_ = kmeans::SimpleMatrix<Index_, Data_>
>
class KmknnBuilder final : public knncolle::Builder<Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * Convenient name for the `KmknnOptions` class that ensures consistent template parametrization.
     */
    typedef KmknnOptions<Index_, Data_, Distance_, KmeansMatrix_> Options;

private:
    std::shared_ptr<const DistanceMetric_> my_metric;
    Options my_options;

public:
    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     * @param options Further options for the KMKNN algorithm.
     */
    KmknnBuilder(std::shared_ptr<const DistanceMetric_> metric, Options options) : my_metric(std::move(metric)), my_options(std::move(options)) {}

    /**
     * Overloaded constructor using the default options.
     *
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    KmknnBuilder(std::shared_ptr<const DistanceMetric_> metric) : KmknnBuilder(std::move(metric), {}) {}

    // Don't provide an overload that accepts a raw metric pointer and the options,
    // as it's possible for the raw pointer to be constructed first, and then the options
    // is constructed but throws an error somewhere (e.g., in an IIFE), causing a memory leak.
    // as the raw pointer is never passed to the shared_ptr for management.

    /**
     * @return Options for the KMKNN algorithm.
     * These can be modified prior to running `build_raw()` and friends.
     */
    Options& get_options() {
        return my_options;
    }

public:
    /**
     * Creates a `KmknnPrebuilt` instance.
     */
    knncolle::Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        std::size_t ndim = data.num_dimensions();
        auto nobs = data.num_observations();

        std::vector<Common<Data_, Distance_> > store(ndim * static_cast<std::size_t>(nobs)); // cast to size_t to avoid overflow problems.
        auto work = data.new_extractor();
        for (Index_ o = 0; o < nobs; ++o) {
            auto ptr = work->next();
            std::copy_n(ptr, ndim, store.begin() + static_cast<std::size_t>(o) * ndim); // cast to size_t to avoid overflow.
        }

        return new KmknnPrebuilt<Index_, Data_, Distance_, DistanceMetric_>(ndim, nobs, std::move(store), my_metric, my_options);
    }
};

}

#endif
