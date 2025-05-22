#ifndef KNNCOLLE_HNSW_HPP
#define KNNCOLLE_HNSW_HPP

#include <vector>
#include <type_traits>
#include <queue>
#include <algorithm>
#include <memory>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "hnswlib/hnswalg.h"

#include "distances.hpp"

/**
 * @file knncolle_hnsw.hpp
 *
 * @brief Approximate nearest neighbor search with HNSW.
 */

/**
 * @namespace knncolle_hnsw
 * @brief knncolle bindings for HNSW search.
 */
namespace knncolle_hnsw {

/**
 * @brief Options for `HnswBuilder` and `HnswPrebuilt`.
 *
 * This can also be created via using the `HnswBuilder::Options` definition,
 * which ensures consistency of template parameters with `HnswBuilder`.
 */
struct HnswOptions {
    /**
     * Number of bidirectional links for each node.
     * This is equivalent to the `M` parameter in the underlying **hnswlib** library, 
     * see [here](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md#construction-parameters) for details.
     */
    int num_links = 16;

    /**
     * Size of the dynamic list of nearest neighbors during index construction.
     * This controls the trade-off between indexing time and accuracy and is equivalent to the `ef_construct` parameter in the underlying **hnswlib** library,
     * see [here](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md#construction-parameters) for details.
     */
    int ef_construction = 200;

    /**
     * Size of the dynamic list of nearest neighbors during searching.
     * This controls the trade-off between search speed and accuracy and is equivalent to the `ef` parameter in the underlying **hnswlib** library,
     * see [here](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md#search-parameters) for details.
     */
    int ef_search = 10;
};

template<typename Index_, typename Data_, typename Distance_, typename HnswData_>
class HnswPrebuilt;

/**
 * @brief Searcher on an Hnsw index.
 *
 * Instances of this class are usually constructed using `HnswPrebuilt::initialize()`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 */
template<typename Index_, typename Data_, typename Distance_, typename HnswData_>
class HnswSearcher final : public knncolle::Searcher<Index_, Data_, Distance_> {
private:
    const HnswPrebuilt<Index_, Data_, Distance_, HnswData_>& my_parent;

    std::priority_queue<std::pair<HnswData_, hnswlib::labeltype> > my_queue;

    static constexpr bool same_internal_data = std::is_same<Data_, HnswData_>::value;
    std::vector<HnswData_> my_buffer;

public:
    /**
     * @cond
     */
    HnswSearcher(const HnswPrebuilt<Index_, Data_, Distance_, HnswData_>& parent) : my_parent(parent) {
        if constexpr(!same_internal_data) {
            my_buffer.resize(my_parent.my_dim);
        }
    }
    /**
     * @endcond
     */

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_buffer = my_parent.my_index.template getDataByLabel<HnswData_>(i);
        Index_ kp1 = k + 1;
        my_queue = my_parent.my_index.searchKnn(my_buffer.data(), kp1); // +1, as it forgets to discard 'self'.

        if (output_indices) {
            output_indices->clear();
            output_indices->reserve(kp1);
        }
        if (output_distances) {
            output_distances->clear();
            output_distances->reserve(kp1);
        }

        bool self_found = false;
        hnswlib::labeltype icopy = i;
        while (!my_queue.empty()) {
            const auto& top = my_queue.top();
            if (!self_found && top.second == icopy) {
                self_found = true;
            } else {
                if (output_indices) {
                    output_indices->push_back(top.second);
                }
                if (output_distances) {
                    output_distances->push_back(top.first);
                }
            }
            my_queue.pop();
        }

        if (output_indices) {
            std::reverse(output_indices->begin(), output_indices->end());
        }
        if (output_distances) {
            std::reverse(output_distances->begin(), output_distances->end());
        }

        // Just in case we're full of ties at duplicate points, such that 'c'
        // is not in the set.  Note that, if self_found=false, we must have at
        // least 'K+2' points for 'c' to not be detected as its own neighbor.
        // Thus there is no need to worry whether we are popping off a non-'c'
        // element and then returning fewer elements than expected.
        if (!self_found) {
            if (output_indices) {
                output_indices->pop_back();
            }
            if (output_distances) {
                output_distances->pop_back();
            }
        }

        if (output_distances && my_parent.my_normalize) {
            for (auto& d : *output_distances) {
                d = my_parent.my_normalize(d);
            }
        }
    }

private:
    void search_raw(const HnswData_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        k = std::min(k, my_parent.my_obs);
        my_queue = my_parent.my_index.searchKnn(query, k); 

        if (output_indices) {
            output_indices->resize(k);
        }
        if (output_distances) {
            output_distances->resize(k);
        }

        auto position = k;
        while (!my_queue.empty()) {
            const auto& top = my_queue.top();
            --position;
            if (output_indices) {
                (*output_indices)[position] = top.second;
            }
            if (output_distances) {
                (*output_distances)[position] = top.first;
            }
            my_queue.pop();
        }

        if (output_distances && my_parent.my_normalize) {
            for (auto& d : *output_distances) {
                d = my_parent.my_normalize(d);
            }
        }
    }

public:
    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if constexpr(same_internal_data) {
            my_queue = my_parent.my_index.searchKnn(query, k);
            search_raw(query, k, output_indices, output_distances);
        } else {
            std::copy_n(query, my_parent.my_dim, my_buffer.begin());
            search_raw(my_buffer.data(), k, output_indices, output_distances);
        }
    }
};

/**
 * @brief Prebuilt index for an Hnsw search.
 *
 * Instances of this class are usually constructed using `HnswBuilder::build_raw()`.
 * The `initialize()` method will create an instance of the `HnswSearcher` class.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 */
template<typename Index_, typename Data_, typename Distance_, typename HnswData_>
class HnswPrebuilt : public knncolle::Prebuilt<Index_, Data_, Distance_> {
public:
    /**
     * @cond
     */
    template<class Matrix_>
    HnswPrebuilt(const Matrix_& data, const DistanceConfig<HnswData_>& distance_config, const HnswOptions& options) :
        my_dim(data.num_dimensions()),
        my_obs(data.num_observations()),
        my_space(distance_config.create(my_dim)),
        my_normalize(distance_config.normalize),
        my_index(my_space.get(), my_obs, options.num_links, options.ef_construction)
    {
        auto work = data.new_extractor();
        if constexpr(std::is_same<Data_, HnswData_>::value) {
            for (Index_ i = 0; i < my_obs; ++i) {
                auto ptr = work->next();
                my_index.addPoint(ptr, i);
            }
        } else {
            std::vector<HnswData_> incoming(my_dim);
            for (Index_ i = 0; i < my_obs; ++i) {
                auto ptr = work->next(); 
                std::copy_n(ptr, my_dim, incoming.begin());
                my_index.addPoint(incoming.data(), i);
            }
        }

        my_index.setEf(options.ef_search);
        return;
    }
    /**
     * @endcond
     */

private:
    std::size_t my_dim;
    Index_ my_obs;

    // The following must be a pointer for polymorphism, but also so that
    // references to the object in my_index are still valid after copying.
    std::shared_ptr<hnswlib::SpaceInterface<HnswData_> > my_space;

    std::function<HnswData_(HnswData_)> my_normalize;
    hnswlib::HierarchicalNSW<HnswData_> my_index;

    friend class HnswSearcher<Index_, Data_, Distance_, HnswData_>;

public:
    std::size_t num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

    /**
     * Creates a `HnswSearcher` instance.
     */
    std::unique_ptr<knncolle::Searcher<Index_, Data_, Distance_> > initialize() const {
        return std::make_unique<HnswSearcher<Index_, Data_, Distance_, HnswData_> >(*this);
    }
};

/**
 * @brief Perform an approximate nearest neighbor search with HNSW.
 *
 * In the HNSW algorithm (Malkov and Yashunin, 2016), each point is a node in a "nagivable small world" graph.
 * The nearest neighbor search proceeds by starting at a node and walking through the graph to obtain closer neighbors to a given query point.
 * Nagivable small world graphs are used to maintain connectivity across the data set by creating links between distant points.
 * This speeds up the search by ensuring that the algorithm does not need to take many small steps to move from one cluster to another.
 * The HNSW algorithm extends this idea by using a hierarchy of such graphs containing links of different lengths,
 * which avoids wasting time on small steps in the early stages of the search where the current node position is far from the query.
 *
 * @see
 * Malkov YA, Yashunin DA (2016).
 * Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.
 * _arXiv_.
 * https://arxiv.org/abs/1603.09320
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `knncolle::Matrix` interface.
 * @tparam HnswData_ Type of data in the HNSW index, usually floating-point.
 * This defaults to a `float` instead of a `double` to sacrifice some accuracy for performance.
 */
template<
    typename Index_,
    typename Data_,
    typename Distance_,
    class Matrix_ = knncolle::Matrix<Index_, Data_>, 
    typename HnswData_ = float
>
class HnswBuilder : public knncolle::Builder<Index_, Data_, Distance_, Matrix_> {
private:
    DistanceConfig<HnswData_> my_distance_config;
    HnswOptions my_options;

public:
    /**
     * @param distance_config Configuration for computing distances in the HNSW index, e.g., `makeEuclideanDistanceConfig()`.
     * @param options Further options for HNSW index construction and searching.
     */
    HnswBuilder(DistanceConfig<HnswData_> distance_config, HnswOptions options) : my_distance_config(std::move(distance_config)), my_options(std::move(options)) {
        if (!my_distance_config.create) {
            throw std::runtime_error("'distance_config.create' was not provided");
        }
    }

    /**
     * Overload that uses the default `Options`.
     * @param distance_config Configuration for computing distances in the HNSW index, e.g., `makeEuclideanDistanceConfig()`.
     */
    HnswBuilder(DistanceConfig<HnswData_> distance_config) : HnswBuilder(std::move(distance_config), {}) {}

    /**
     * @return Options for HNSW, to be modified prior to calling `build_raw()` and friends.
     */
    HnswOptions& get_options() {
        return my_options;
    }

public:
    /**
     * Creates a `HnswPrebuilt` instance.
     */
    knncolle::Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        return new HnswPrebuilt<Index_, Data_, Distance_, HnswData_>(data, my_distance_config, my_options);
    }
};

}

#endif
