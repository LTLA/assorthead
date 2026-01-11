#ifndef KMEANS_TATAMI_HPP
#define KMEANS_TATAMI_HPP

#include <memory>
#include <type_traits>
#include <cstddef>

#include "tatami/tatami.hpp"
#include "kmeans/kmeans.hpp"

/**
 * @file kmeans_tatami.hpp
 * @brief Use a **tatami** matrix with the **kmeans** library.
 */

/**
 * @namespace kmeans_tatami
 * @brief Wrapper around a **tatami** matrix.
 */
namespace kmeans_tatami {

/**
 * @cond
 */
template<typename KIndex_, typename KData_, typename TValue_, typename TIndex_, class Extractor_ = tatami::MyopicDenseExtractor<TValue_, TIndex_> >
class Random final : public kmeans::RandomAccessExtractor<KIndex_, KData_> {
public:
    Random(std::unique_ptr<Extractor_> ext, TIndex_ ndim) : my_ext(std::move(ext)) {
        tatami::resize_container_to_Index_size(my_buffer, ndim); 
        if constexpr(!same_type) {
            tatami::resize_container_to_Index_size(my_output, ndim); 
        }
    }

private:
    std::unique_ptr<Extractor_> my_ext;
    std::vector<TValue_> my_buffer;
    static constexpr bool same_type = std::is_same<TValue_, KData_>::value;
    typename std::conditional<same_type, bool, std::vector<KData_> > my_output;

public:
    const KData_* get_observation(KIndex_ i) {
        auto ptr = my_ext->fetch(i, my_buffer.data());
        if constexpr(same_type) {
            return ptr;
        } else {
            std::copy_n(ptr, my_buffer.size(), my_output.data());
            return my_output.data();
        }
    }
};

template<typename KIndex_, typename KData_, typename TValue_, typename TIndex_, class Extractor_ = tatami::OracularDenseExtractor<TValue_, TIndex_> >
class Consecutive final : public kmeans::ConsecutiveAccessExtractor<KIndex_, KData_> {
public:
    Consecutive(std::unique_ptr<Extractor_> ext, TIndex_ ndim) : my_ext(std::move(ext)) {
        tatami::resize_container_to_Index_size(my_buffer, ndim); 
        if constexpr(!same_type) {
            tatami::resize_container_to_Index_size(my_output, ndim); 
        }
    }

private:
    std::unique_ptr<Extractor_> my_ext;
    std::vector<TValue_> my_buffer;
    static constexpr bool same_type = std::is_same<TValue_, KData_>::value;
    typename std::conditional<same_type, bool, std::vector<KData_> > my_output;

public:
    const KData_* get_observation() {
        auto ptr = my_ext->fetch(my_buffer.data());
        if constexpr(same_type) {
            return ptr;
        } else {
            std::copy_n(ptr, my_buffer.size(), my_output.data());
            return my_output.data();
        }
    }
};

template<typename KIndex_, typename KData_, typename TValue_, typename TIndex_, class Extractor_ = tatami::OracularDenseExtractor<TValue_, TIndex_> >
class Indexed final : public kmeans::IndexedAccessExtractor<KIndex_, KData_> {
public:
    Indexed(std::unique_ptr<Extractor_> ext, TIndex_ ndim) : my_ext(std::move(ext)) {
        tatami::resize_container_to_Index_size(my_buffer, ndim); 
        if constexpr(!same_type) {
            tatami::resize_container_to_Index_size(my_output, ndim); 
        }
    }

private:
    std::unique_ptr<Extractor_> my_ext;
    std::vector<TValue_> my_buffer;
    static constexpr bool same_type = std::is_same<TValue_, KData_>::value;
    typename std::conditional<same_type, bool, std::vector<KData_> > my_output;

public:
    const KData_* get_observation() {
        auto ptr = my_ext->fetch(my_buffer.data());
        if constexpr(same_type) {
            return ptr;
        } else {
            std::copy_n(ptr, my_buffer.size(), my_output.data());
            return my_output.data();
        }
    }
};
/**
 * @endcond
 */

/**
 * @tparam KIndex_ Integer type of observation indices for **kmeans**.
 * @tparam KData_ Numeric type of the data for **kmeans**.
 * @tparam TValue_ Numeric type of matrix values for **tatami**.
 * @tparam TIndex_ Integer type of the row/column indices for **tatami**.
 * @tparam Matrix_ Class of the **tatami** matrix.
 *
 * @brief **kmeans**-compatible wrapper around a **tatami** matrix.
 *
 * Pretty much as it says on the tin - implements a `kmeans::Matrix` subclass to wrap a `tatami::Matrix`.
 * The idea is to enable the use of arbitrary **tatami** matrix representations in **kmeans** functions,
 * e.g., to support clustering from a file-backed matrix.
 */
template<typename KIndex_, typename KData_, typename TValue_, typename TIndex_, class Matrix_ = tatami::Matrix<TValue_, TIndex_> >
class Matrix final : public kmeans::Matrix<KIndex_, KData_> {
private:
    std::shared_ptr<tatami::Matrix<TValue_, TIndex_> > my_matrix;

public:
    Matrix(std::shared_ptr<tatami::Matrix<TValue_, TIndex_> > matrix) : my_matrix(std::move(matrix)) {
        // Making sure that we can cast to KIndex_.
        // tatami extents are guaranteed to be positive and fit in a size_t, so we attest that.
        sanisizer::cast<KIndex_>(sanisizer::attest_gez(sanisizer::attest_max_by_type<std::size_t>(my_matrix->ncol())));
    }

    KIndex_ num_observations() const {
        return my_matrix->ncol();
    }

    std::size_t num_dimensions() const {
        // tatami extents are guaranteed to fit in a size_t, so it's okay.
        return my_matrix->nrow();
    }

public:
    std::unique_ptr<kmeans::RandomAccessExtractor<KIndex_, KData_> > new_extractor() const {
        return std::make_unique<Random<KIndex_, KData_, TValue_, TIndex_> >(my_matrix->dense_column(), my_matrix->ncol());
    }

    std::unique_ptr<kmeans::ConsecutiveAccessExtractor<KIndex_, KData_> > new_extractor(KIndex_ start, KIndex_ length) const {
        // Block should be castable from TIndex_ to KIndex_ as it should be less than num_observations(). 
        auto optr = std::make_shared<tatami::ConsecutiveOracle<TIndex_> >(start, length);
        return std::make_unique<Consecutive<KIndex_, KData_, TValue_, TIndex_> >(my_matrix->dense_column(std::move(optr)), my_matrix->ncol());
    }

    std::unique_ptr<kmeans::IndexedAccessExtractor<KIndex_, KData_> > new_extractor(const KIndex_* sequence, std::size_t length) const {
        auto optr = std::make_shared<tatami::FixedViewOracle<TIndex_, const KIndex_*> >(sequence, length);
        return std::make_unique<Indexed<KIndex_, KData_, TValue_, TIndex_> >(my_matrix->dense_column(std::move(optr)), my_matrix->ncol());
    }
};

}

#endif
