#ifndef KMEANS_TATAMI_HPP
#define KMEANS_TATAMI_HPP

#include <memory>
#include <type_traits>
#include <cstddef>
#include <vector>

#include "tatami/tatami.hpp"
#include "knncolle/knncolle.hpp"

/**
 * @file knncolle_tatami.hpp
 * @brief Use a **tatami** matrix with the **knncolle** library.
 */

/**
 * @namespace knncolle_tatami
 * @brief Wrapper around a **tatami** matrix.
 */
namespace knncolle_tatami {

/**
 * @cond
 */
template<typename KData_, typename TValue_, typename TIndex_, class Extractor_ = tatami::OracularDenseExtractor<TValue_, TIndex_> >
class Extractor final : public knncolle::MatrixExtractor<KData_> {
public:
    Extractor(std::unique_ptr<Extractor_> ext, TIndex_ ndim) : my_ext(std::move(ext)) {
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
    const KData_* next() {
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
 * @tparam KIndex_ Integer type of observation indices for **knncolle**.
 * @tparam KData_ Numeric type of the data for **knncolle**.
 * @tparam TValue_ Numeric type of matrix values for **tatami**.
 * @tparam TIndex_ Integer type of the row/column indices for **tatami**.
 * @tparam MatrixPointer_ Pointer to a `tatami::Matrix`.
 * This may be a smart or raw pointer class.
 *
 * @brief **knncolle**-compatible wrapper around a **tatami** matrix.
 *
 * Pretty much as it says on the tin - implements a `knncolle::Matrix` subclass to wrap a `tatami::Matrix`.
 * The idea is to enable the use of arbitrary **tatami** matrix representations in **knncolle** functions,
 * e.g., to create a search index from a file-backed matrix.
 */
template<typename KIndex_, typename KData_, typename TValue_, typename TIndex_, class MatrixPointer_ = std::shared_ptr<const tatami::Matrix<TValue_, TIndex_> > >
class Matrix final : public knncolle::Matrix<KIndex_, KData_> {
private:
    MatrixPointer_ my_matrix;
    KIndex_ my_nobs;
    std::size_t my_ndim;
    bool my_transposed;

public:
    /**
     * @param matrix Raw or smart pointer to a `tatami::Matrix`.
     * @param transposed Whether to transpose the matrix during extraction in **kmeans** functions.
     * If `true`, `new_extractor()` will extract rows instead of columns.
     */
    Matrix(MatrixPointer_  matrix, bool transposed) : my_matrix(std::move(matrix)), my_transposed(transposed) {
        TIndex_ cur_nobs;
        if (my_transposed) {
            cur_nobs = my_matrix->nrow();
            my_ndim = my_matrix->ncol(); // cast is guaranteed to be safe as tatami indices can always fit in a size_t.
        } else {
            cur_nobs = my_matrix->ncol();
            my_ndim = my_matrix->nrow();
        }

        // Making sure that we can cast to Index_.
        // tatami extents are guaranteed to be positive and fit in a size_t, so we attest that.
        my_nobs = sanisizer::cast<KIndex_>(sanisizer::attest_gez(sanisizer::attest_max_by_type<std::size_t>(cur_nobs)));
    }

    KIndex_ num_observations() const {
        return my_nobs;
    }

    std::size_t num_dimensions() const {
        return my_ndim;
    }

public:
    std::unique_ptr<knncolle::MatrixExtractor<KData_> > new_extractor() const {
        return new_known_extractor();
    }

    /**
     * Override to assist devirtualization.
     */
    auto new_known_extractor() const {
        auto ext = tatami::consecutive_extractor<false, TValue_, TIndex_>(*my_matrix, my_transposed, 0, my_nobs);
        return std::make_unique<Extractor<KData_, TValue_, TIndex_> >(std::move(ext), my_ndim);
    }
};

}

#endif
