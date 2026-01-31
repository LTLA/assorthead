#ifndef KNNCOLLE_HNSW_HNSW_HPP
#define KNNCOLLE_HNSW_HNSW_HPP

#include <vector>
#include <type_traits>
#include <queue>
#include <algorithm>
#include <memory>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <filesystem>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"
#include "hnswlib/hnswalg.h"

#include "distances.hpp"
#include "utils.hpp"

/**
 * @file knncolle_hnsw.hpp
 *
 * @brief Approximate nearest neighbor search with HNSW.
 */

namespace knncolle_hnsw {

/**
 * Name of the HNSW algorithm when registering a loading function to `knncolle::load_prebuilt_registry()`.
 */
inline static constexpr const char* hnsw_prebuilt_save_name = "knncolle_hnsw::Hnsw";

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

/**
 * @cond
 */
template<typename Index_, typename Data_, typename Distance_, typename HnswData_>
class HnswPrebuilt;

template<typename Index_, typename Data_, typename Distance_, typename HnswData_>
class HnswSearcher final : public knncolle::Searcher<Index_, Data_, Distance_> {
private:
    const HnswPrebuilt<Index_, Data_, Distance_, HnswData_>& my_parent;

    std::priority_queue<std::pair<HnswData_, hnswlib::labeltype> > my_queue;

    static constexpr bool same_internal_data = std::is_same<Data_, HnswData_>::value;
    std::vector<HnswData_> my_buffer;

public:
    HnswSearcher(const HnswPrebuilt<Index_, Data_, Distance_, HnswData_>& parent) : my_parent(parent) {
        if constexpr(!same_internal_data) {
            sanisizer::resize(my_buffer, my_parent.my_dim);
        }
    }

private:
    void normalize_distances(std::vector<Distance_>& output_distances) const {
        switch(my_parent.my_normalize_method) {
            case DistanceNormalizeMethod::SQRT:
                for (auto& d : output_distances) {
                    d = std::sqrt(d);
                }
                break;
            case DistanceNormalizeMethod::CUSTOM:
                for (auto& d : output_distances) {
                    d = my_parent.my_custom_normalize(d);
                }
                break;
            case DistanceNormalizeMethod::NONE:
                break;
        }
    }

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

        if (output_distances) {
            normalize_distances(*output_distances);
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

        if (output_distances) {
            normalize_distances(*output_distances);
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

template<typename Index_, typename Data_, typename Distance_, typename HnswData_>
class HnswPrebuilt final : public knncolle::Prebuilt<Index_, Data_, Distance_> {
public:
    template<class Matrix_>
    HnswPrebuilt(const Matrix_& data, const DistanceConfig<Distance_, HnswData_>& distance_config, const HnswOptions& options) :
        my_dim(data.num_dimensions()),
        my_obs(data.num_observations()),
        my_space(distance_config.create(my_dim)),
        my_normalize_method(distance_config.normalize_method),
        my_custom_normalize(distance_config.custom_normalize),
        my_index(my_space.get(), my_obs, options.num_links, options.ef_construction)
    {
        auto work = data.new_known_extractor();
        if constexpr(std::is_same<Data_, HnswData_>::value) {
            for (Index_ i = 0; i < my_obs; ++i) {
                auto ptr = work->next();
                my_index.addPoint(ptr, i);
            }
        } else {
            auto incoming = sanisizer::create<std::vector<HnswData_> >(my_dim);
            for (Index_ i = 0; i < my_obs; ++i) {
                auto ptr = work->next(); 
                std::copy_n(ptr, my_dim, incoming.begin());
                my_index.addPoint(incoming.data(), i);
            }
        }

        my_index.setEf(options.ef_search);
        return;
    }

private:
    std::size_t my_dim;
    Index_ my_obs;

    // The following must be a pointer for polymorphism, but also so that
    // references to the object in my_index are still valid after copying.
    std::shared_ptr<hnswlib::SpaceInterface<HnswData_> > my_space;

    DistanceNormalizeMethod my_normalize_method;
    std::function<Distance_(Distance_)> my_custom_normalize;

    hnswlib::HierarchicalNSW<HnswData_> my_index;

    friend class HnswSearcher<Index_, Data_, Distance_, HnswData_>;

public:
    std::size_t num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

public:
    std::unique_ptr<knncolle::Searcher<Index_, Data_, Distance_> > initialize() const {
        return initialize_known();
    }

    auto initialize_known() const {
        return std::make_unique<HnswSearcher<Index_, Data_, Distance_, HnswData_> >(*this);
    }

public:
    void save(const std::filesystem::path& dir) const {
        knncolle::quick_save(dir / "ALGORITHM", hnsw_prebuilt_save_name, std::strlen(hnsw_prebuilt_save_name));
        knncolle::quick_save(dir / "NUM_OBS", &my_obs, 1);
        knncolle::quick_save(dir / "NUM_DIM", &my_dim, 1);

        auto type = knncolle::get_numeric_type<HnswData_>();
        knncolle::quick_save(dir / "TYPE", &type, 1);

        const char* distname = get_distance_name(my_space.get());;
        knncolle::quick_save(dir / "DISTANCE", distname, std::strlen(distname));
        knncolle::quick_save(dir / "NORMALIZE", &my_normalize_method, 1);

        // Custom normalization functions.
        auto& datafunc = custom_save_for_hnsw_data<HnswData_>();
        if (datafunc) {
            datafunc(dir);
        }

        auto& distfunc = custom_save_for_hnsw_distance<HnswData_>();
        if (std::strcmp(distname, "unknown") == 0 && distfunc) {
            distfunc(dir, my_space.get());
        }

        auto& normfunc = custom_save_for_hnsw_normalize<Distance_>();
        if (my_normalize_method == DistanceNormalizeMethod::CUSTOM && normfunc) {
            normfunc(dir, my_custom_normalize);
        }

        // Dear God, make saveIndex() const.
        auto index_ptr = const_cast<hnswlib::HierarchicalNSW<HnswData_>*>(&my_index);
        index_ptr->saveIndex(dir / "INDEX");
    }

    HnswPrebuilt(const std::filesystem::path& dir) : 
        my_dim([&]() {
            std::size_t dim;
            knncolle::quick_load(dir / "NUM_DIM", &dim, 1);
            return dim;
        }()),

        my_obs([&]() {
            Index_ obs;
            knncolle::quick_load(dir / "NUM_OBS", &obs, 1);
            return obs;
        }()),

        my_space([&]() {
            std::string method = knncolle::quick_load_as_string(dir / "DISTANCE");

            if constexpr(std::is_same<HnswData_, float>::value) {
                if (method == "l2") {
                    return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new hnswlib::L2Space(my_dim));
                }
            }
            if (method == "squared_euclidean") {
                return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new SquaredEuclideanDistance<HnswData_>(my_dim));
            } else if (method == "manhattan") {
                return static_cast<hnswlib::SpaceInterface<HnswData_>*>(new ManhattanDistance<HnswData_>(my_dim));
            }

            auto& loadfun = custom_load_for_hnsw_distance<HnswData_>();
            if (!loadfun) {
                throw std::runtime_error("no loader provided for an unknown distance");
            }
            return static_cast<hnswlib::SpaceInterface<HnswData_>*>(loadfun(dir, my_dim));
        }()),

        my_normalize_method([&]() {
            DistanceNormalizeMethod norm;
            knncolle::quick_load(dir / "NORMALIZE", &norm, 1);
            return norm;
        }()),

        my_index(my_space.get(), dir / "INDEX")

    {
        if (my_normalize_method == DistanceNormalizeMethod::CUSTOM) {
            auto& normfun = custom_load_for_hnsw_normalize<Distance_>();
            if (!normfun) {
                throw std::runtime_error("no loader provided for an unknown normalization");
            }
            my_custom_normalize = normfun(dir);
        }
    }
};
/**
 * @endcond
 */

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
class HnswBuilder final : public knncolle::Builder<Index_, Data_, Distance_, Matrix_> {
private:
    DistanceConfig<Distance_, HnswData_> my_distance_config;
    HnswOptions my_options;

public:
    /**
     * @param distance_config Configuration for computing distances in the HNSW index, e.g., `configure_euclidean_distance()`.
     * @param options Further options for HNSW index construction and searching.
     */
    HnswBuilder(DistanceConfig<Distance_, HnswData_> distance_config, HnswOptions options) :
        my_distance_config(std::move(distance_config)),
        my_options(std::move(options))
    {
        if (!my_distance_config.create) {
            throw std::runtime_error("'distance_config.create' was not provided");
        }
        if (my_distance_config.normalize_method == DistanceNormalizeMethod::CUSTOM && !my_distance_config.custom_normalize) {
            throw std::runtime_error("'distance_config.custom_normalize' was not provided");
        }
    }

    /**
     * Overload that uses the default `Options`.
     * @param distance_config Configuration for computing distances in the HNSW index, e.g., `configure_euclidean_distance()`.
     */
    HnswBuilder(DistanceConfig<Distance_, HnswData_> distance_config) : HnswBuilder(std::move(distance_config), {}) {}

    /**
     * @return Options for HNSW, to be modified prior to calling `knncolle::Builder::build_raw()` and friends.
     */
    HnswOptions& get_options() {
        return my_options;
    }

public:
    /**
     * @cond
     */
    knncolle::Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        return build_known_raw(data);
    }
    /**
     * @endcond
     */

public:
    /**
     * Override to assist devirtualization.
     */
    auto build_known_raw(const Matrix_& data) const {
        return new HnswPrebuilt<Index_, Data_, Distance_, HnswData_>(data, my_distance_config, my_options);
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_unique(const Matrix_& data) const {
        return std::unique_ptr<I<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_shared(const Matrix_& data) const {
        return std::shared_ptr<I<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }
};

}

#endif
