#ifndef KNNCOLLE_ANNOY_HPP
#define KNNCOLLE_ANNOY_HPP

#include <vector>
#include <type_traits>
#include <algorithm>
#include <memory>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

/**
 * @file knncolle_annoy.hpp
 * @brief Approximate nearest neighbor search with Annoy.
 */

/**
 * @namespace knncolle_annoy
 * @brief Approximate nearest neighbor search with Annoy.
 */
namespace knncolle_annoy {

/**
 * @brief Options for `AnnoyBuilder()`. 
 */
struct AnnoyOptions {
    /**
     * Number of trees to construct.
     * Larger values improve accuracy at the cost of index size (i.e., memory usage), see [here](https://github.com/spotify/annoy#tradeoffs) for details.
     */
    int num_trees = 50;

    /**
     * Factor that is multiplied by the number of neighbors `k` to determine the number of nodes to search in `AnnoySearcher::search()`.
     * Larger values improve accuracy at the cost of runtime, see [here](https://github.com/spotify/annoy#tradeoffs) for details.
     * This should be greater than or equal to 1 - any other value will be replaced by `num_trees`.
     */
    double search_mult = -1;
};

/**
 * @cond
 */
template<typename Index_, typename Data_, typename Distance_, typename AnnoyDistance_, typename AnnoyIndex_, typename AnnoyData_, class AnnoyRng_, class AnnoyThreadPolicy_>
class AnnoyPrebuilt;

template<
    typename Index_,
    typename Data_,
    typename Distance_, 
    class AnnoyDistance_,
    typename AnnoyIndex_ = Index_,
    typename AnnoyData_ = float,
    class AnnoyRng_ = Annoy::Kiss64Random,
    class AnnoyThreadPolicy_ = Annoy::AnnoyIndexSingleThreadedBuildPolicy
>
class AnnoySearcher final : public knncolle::Searcher<Index_, Data_, Distance_> {
private:
    const AnnoyPrebuilt<Index_, Data_, Distance_, AnnoyDistance_, AnnoyIndex_, AnnoyData_, AnnoyRng_, AnnoyThreadPolicy_>& my_parent;

    static constexpr bool same_internal_data = std::is_same<Data_, AnnoyData_>::value;
    typename std::conditional<!same_internal_data, std::vector<AnnoyData_>, bool>::type my_buffer;

    static constexpr bool same_internal_index = std::is_same<Index_, AnnoyIndex_>::value;
    std::vector<AnnoyIndex_> my_indices;

    static constexpr bool same_internal_distance = std::is_same<Distance_, AnnoyData_>::value;
    typename std::conditional<!same_internal_distance, std::vector<AnnoyData_>, bool>::type my_distances;

    int get_search_k(int k) const {
        if (my_parent.my_search_mult < 1) {
            return -1; // instructs Annoy to use k * num_trees. 
        } else {
            return my_parent.my_search_mult * static_cast<double>(k) + 0.5; // rounded.
        }
    }

public:
    AnnoySearcher(const AnnoyPrebuilt<Index_, Data_, Distance_, AnnoyDistance_, AnnoyIndex_, AnnoyData_, AnnoyRng_, AnnoyThreadPolicy_>& parent) : my_parent(parent) {
        if constexpr(!same_internal_data) {
            my_buffer.resize(my_parent.my_dim);
        }
    }

private:
    std::pair<std::vector<AnnoyIndex_>*, std::vector<AnnoyData_>*> obtain_pointers(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ k) {
        std::vector<AnnoyIndex_>* icopy_ptr = &my_indices;
        if (output_indices) {
            if constexpr(same_internal_index) {
                icopy_ptr = output_indices;
            }
        }
        icopy_ptr->clear();
        icopy_ptr->reserve(k);

        std::vector<AnnoyData_>* dcopy_ptr = NULL;
        if (output_distances) {
            if constexpr(same_internal_distance) {
                dcopy_ptr = output_distances;
            } else {
                dcopy_ptr = &my_distances;
            }
            dcopy_ptr->clear();
            dcopy_ptr->reserve(k);
        }

        return std::make_pair(icopy_ptr, dcopy_ptr);
    }

    template<typename Type_>
    static void remove_self(std::vector<Type_>& vec, std::size_t at) {
        if (at < vec.size()) {
            vec.erase(vec.begin() + at);
        } else {
            vec.pop_back();
        }
    }

    template<typename Source_, typename Dest_>
    static void copy_skip_self(const std::vector<Source_>& source, std::vector<Dest_>& dest, std::size_t at) {
        auto sIt = source.begin();
        auto end = source.size();
        dest.clear();
        dest.reserve(end - 1);

        if (at < end) {
            dest.insert(dest.end(), sIt, sIt + at);
            dest.insert(dest.end(), sIt + at + 1, source.end());
        } else {
            // Just in case we're full of ties at duplicate points, such that 'c'
            // is not in the set.  Note that, if self_found=false, we must have at
            // least 'k+2' points for 'c' to not be detected as its own neighbor.
            // Thus there is no need to worry whether 'end - 1 != k'; we
            // are guaranteed to return 'k' elements in this case.
            dest.insert(dest.end(), sIt, sIt + end - 1);
        }
    }

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        Index_ kp1 = k + 1; // +1, as it forgets to discard 'self'.
        auto ptrs = obtain_pointers(output_indices, output_distances, kp1);
        auto icopy_ptr = ptrs.first;
        auto dcopy_ptr = ptrs.second;

        my_parent.my_index.get_nns_by_item(i, kp1, get_search_k(kp1), icopy_ptr, dcopy_ptr);

        std::size_t at;
        {
            const auto& cur_i = *icopy_ptr;
            at = cur_i.size();
            AnnoyIndex_ icopy = i;
            for (std::size_t x = 0, end = cur_i.size(); x < end; ++x) {
                if (cur_i[x] == icopy) {
                    at = x;
                    break;
                }
            }
        }

        if (output_indices) {
            if constexpr(same_internal_index) {
                remove_self(*output_indices, at);
            } else {
                copy_skip_self(my_indices, *output_indices, at);
            }
        }

        if (output_distances) {
            if constexpr(same_internal_distance) {
                remove_self(*output_distances, at);
            } else {
                copy_skip_self(my_distances, *output_distances, at);
            }
        }
    }

private:
    void search_raw(const AnnoyData_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto ptrs = obtain_pointers(output_indices, output_distances, k);
        auto icopy_ptr = ptrs.first;
        auto dcopy_ptr = ptrs.second;

        my_parent.my_index.get_nns_by_vector(query, k, get_search_k(k), icopy_ptr, dcopy_ptr);

        if (output_indices) {
            if constexpr(!same_internal_index) {
                output_indices->clear();
                output_indices->insert(output_indices->end(), my_indices.begin(), my_indices.end());
            }
        }

        if (output_distances) {
            if constexpr(!same_internal_distance) {
                output_distances->clear();
                output_distances->insert(output_distances->end(), my_distances.begin(), my_distances.end());
            }
        }
    }

public:
    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if constexpr(same_internal_data) {
            search_raw(query, k, output_indices, output_distances);
        } else {
            std::copy_n(query, my_parent.my_dim, my_buffer.begin());
            search_raw(my_buffer.data(), k, output_indices, output_distances);
        }
    }
};

template<
    typename Index_,
    typename Data_,
    typename Distance_, 
    class AnnoyDistance_,
    typename AnnoyIndex_ = Index_,
    typename AnnoyData_ = float,
    class AnnoyRng_ = Annoy::Kiss64Random,
    class AnnoyThreadPolicy_ = Annoy::AnnoyIndexSingleThreadedBuildPolicy
>
class AnnoyPrebuilt final : public knncolle::Prebuilt<Index_, Data_, Distance_> {
public:
    template<class Matrix_>
    AnnoyPrebuilt(const Matrix_& data, const AnnoyOptions& options) :
        my_dim(data.num_dimensions()),
        my_obs(data.num_observations()),
        my_search_mult(options.search_mult),
        my_index(my_dim)
    {
        auto work = data.new_known_extractor();
        if constexpr(std::is_same<Data_, AnnoyData_>::value) {
            for (Index_ i = 0; i < my_obs; ++i) {
                auto ptr = work->next();
                my_index.add_item(i, ptr);
            }
        } else {
            std::vector<AnnoyData_> incoming(my_dim);
            for (Index_ i = 0; i < my_obs; ++i) {
                auto ptr = work->next();
                std::copy_n(ptr, my_dim, incoming.begin());
                my_index.add_item(i, incoming.data());
            }
        }

        my_index.build(options.num_trees);
        return;
    }

private:
    std::size_t my_dim;
    Index_ my_obs;
    double my_search_mult;
    Annoy::AnnoyIndex<AnnoyIndex_, AnnoyData_, AnnoyDistance_, AnnoyRng_, AnnoyThreadPolicy_> my_index;

    friend class AnnoySearcher<Index_, Data_, Distance_, AnnoyDistance_, AnnoyIndex_, AnnoyData_, AnnoyRng_, AnnoyThreadPolicy_>;

public:
    std::size_t num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

    std::unique_ptr<knncolle::Searcher<Index_, Data_, Distance_> > initialize() const {
        return initialize_known();
    }

    auto initialize_known() const {
        return std::make_unique<AnnoySearcher<Index_, Data_, Distance_, AnnoyDistance_, AnnoyIndex_, AnnoyData_, AnnoyRng_, AnnoyThreadPolicy_> >(*this);
    }
};
/**
 * @endcond
 */

/**
 * @brief Perform an approximate nearest neighbor search with Annoy.
 *
 * In the Approximate Nearest Neighbors Oh Yeah (Annoy) algorithm, a tree is constructed where a random hyperplane splits the points into two subsets at each internal node.
 * Leaf nodes are defined when the number of points in a subset falls below a threshold (close to twice the number of dimensions for the settings used here).
 * Multiple trees are constructed in this manner, each of which is different due to the random choice of hyperplanes.
 * For a given query point, each tree is searched to identify the subset of all points in the same leaf node as the query point. 
 * The union of these subsets across all trees is exhaustively searched to identify the actual nearest neighbors to the query.
 *
 * @see
 * Bernhardsson E (2018).
 * Annoy.
 * https://github.com/spotify/annoy
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam AnnoyDistance_ An **Annoy**-compatible class to compute the distance between vectors, e.g., `Annoy::Euclidean`, `Annoy::Manhattan`.
 * Note that this is not the same as `knncolle::DistanceMetric`.
 * @tparam AnnoyIndex_ Integer type for the observation indices in the Annoy index.
 * @tparam AnnoyData_ Floating-point type for data in the Annoy index.
 * This defaults to a `float` instead of a `double` to sacrifice some accuracy for performance.
 * @tparam AnnoyRng_ An **Annoy** class for random number generation.
 * @tparam AnnoyThreadPolicy_ An **Annoy** class for the threadedness of Annoy index building.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `knncolle::Matrix` interface.
 */
template<
    typename Index_,
    typename Data_,
    typename Distance_, 
    class AnnoyDistance_,
    typename AnnoyIndex_ = Index_,
    typename AnnoyData_ = float,
    class AnnoyRng_ = Annoy::Kiss64Random,
    class AnnoyThreadPolicy_ = Annoy::AnnoyIndexSingleThreadedBuildPolicy,
    class Matrix_ = knncolle::Matrix<Index_, Data_>
>
class AnnoyBuilder : public knncolle::Builder<Index_, Data_, Distance_, Matrix_> {
private:
    AnnoyOptions my_options;

public:
    /**
     * @param options Further options for Annoy index construction and searching.
     */
    AnnoyBuilder(AnnoyOptions options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    AnnoyBuilder() = default;

    /**
     * @return Options to the Annoy algorithm,
     * to be modified prior to calling `build_raw()` and friends.
     */
    AnnoyOptions& get_options() {
        return my_options;
    }

public:
    knncolle::Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        return build_known_raw(data);
    }

public:
    /**
     * Override to assist devirtualization.
     */
    auto build_known_raw(const Matrix_& data) const {
        return new AnnoyPrebuilt<Index_, Data_, Distance_, AnnoyDistance_, AnnoyIndex_, AnnoyData_, AnnoyRng_, AnnoyThreadPolicy_>(data, my_options);
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_unique(const Matrix_& data) const {
        return std::unique_ptr<std::remove_reference_t<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_shared(const Matrix_& data) const {
        return std::shared_ptr<std::remove_reference_t<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }
};

}

#endif
