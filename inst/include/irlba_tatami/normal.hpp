#ifndef IRLBA_TATAMI_NORMAL_HPP
#define IRLBA_TATAMI_NORMAL_HPP

#include "tatami/tatami.hpp"
#include "tatami_mult/tatami_mult.hpp"
#include "irlba/irlba.hpp"
#include "sanisizer/sanisizer.hpp"

#include <memory>
#include <type_traits>

/**
 * @file normal.hpp
 * @brief Wrap a **tatami** matrix. 
 */

namespace irlba_tatami {

/**
 * @cond
 */
template<class EigenVector_, typename TValue_, typename TIndex_, class TMatrix_ = tatami::Matrix<TValue_, TIndex_> >
class NormalWorkspace final : public irlba::Workspace<EigenVector_> {
public:
    NormalWorkspace(const TMatrix_& mat, int num_threads) : my_mat(mat) {
        opt.num_threads = num_threads;
    }

private:
    const TMatrix_& my_mat;
    tatami_mult::Options opt;

public:
    void multiply(const EigenVector_& right, EigenVector_& out) {
        tatami_mult::multiply(my_mat, right.data(), out.data(), opt);
    }
};

template<class EigenVector_, typename TValue_, typename TIndex_, class TMatrix_ = tatami::Matrix<TValue_, TIndex_> >
class NormalAdjointWorkspace final : public irlba::AdjointWorkspace<EigenVector_> {
public:
    NormalAdjointWorkspace(const TMatrix_& mat, int num_threads) : my_mat(mat) {
        opt.num_threads = num_threads;
    }

private:
    const TMatrix_& my_mat;
    tatami_mult::Options opt;

public:
    void multiply(const EigenVector_& right, EigenVector_& out) {
        tatami_mult::multiply(right.data(), my_mat, out.data(), opt);
    }
};

template<class EigenMatrix_, typename TValue_, typename TIndex_, class TMatrix_ = tatami::Matrix<TValue_, TIndex_> >
class NormalRealizeWorkspace final : public irlba::RealizeWorkspace<EigenMatrix_> {
public:
    NormalRealizeWorkspace(const TMatrix_& mat, int num_threads) : my_mat(mat), my_num_threads(num_threads) {}

private:
    const TMatrix_& my_mat;
    int my_num_threads;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        // Copying into a transposed matrix, hence the switch of the ncol/nrow order.
        // Both values can be cast to Eigen::Index, as we checked this in the Normal constructor.
        buffer.resize(my_mat.nrow(), my_mat.ncol());

        tatami::convert_to_dense(
            my_mat,
            buffer.IsRowMajor,
            buffer.data(),
            [&]{
                tatami::ConvertToDenseOptions opt;
                opt.num_threads = my_num_threads;
                return opt;
            }()
        );

        return buffer;
    }
};
/**
 * @endcond
 */

/**
 * @brief Wrap a **tatami** matrix for **irlba**.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector`.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix`.
 * @tparam TValue_ Numeric type of the **tatami** matrix value.
 * @tparam TIndex_ Integer type of the **tatami** matrix row/column indices.
 * @tparam TMatrixPointer_ Pointer to a **tatami** matrix class consistent with `TValue_` and `TIndex_`. 
 * This may be a raw or smart pointer.
 *
 * This class computes the matrix-vector product for a `tatami::Matrix` or one of its subclasses.
 * The aim is to avoid realizing a `tatami::Matrix` into an `Eigen::Matrix` or `irlba::ParallelSparseMatrix` for use in `irlba::compute()`.
 * Iteration over a `tatami::Matrix` is generally slower, as this effectively trades speed for memory efficiency.
 */
template<class EigenVector_, class EigenMatrix_, typename TValue_, typename TIndex_, class TMatrixPointer_ = std::shared_ptr<const tatami::Matrix<TValue_, TIndex_> > >
class Normal final : public irlba::Matrix<EigenVector_, EigenMatrix_> {
public:
    /**
     * @param mat Pointer to an instance of a **tatami** matrix.
     * The lifetime of `mat` should exceed that of the `Normal` instance constructed from it.
     * @param num_threads Number of threads for the various operations.
     */
    Normal(TMatrixPointer_ mat, int num_threads) : my_mat(std::move(mat)), my_num_threads(num_threads) {
        // Check that these casts are safe in rows(), cols() below.
        sanisizer::cast<Eigen::Index>(tatami::attest_for_Index(my_mat->nrow()));
        sanisizer::cast<Eigen::Index>(tatami::attest_for_Index(my_mat->ncol()));
    }

public:
    Eigen::Index rows() const {
        return my_mat->nrow();
    }

    Eigen::Index cols() const {
        return my_mat->ncol();
    }

private:
    TMatrixPointer_ my_mat;
    int my_num_threads;

    typedef std::remove_cv_t<std::remove_reference_t<decltype(*my_mat)> > TMatrix;

public:
    std::unique_ptr<irlba::Workspace<EigenVector_> > new_workspace() const {
        return new_known_workspace();
    }

    std::unique_ptr<irlba::AdjointWorkspace<EigenVector_> > new_adjoint_workspace() const {
        return new_known_adjoint_workspace();
    }

    std::unique_ptr<irlba::RealizeWorkspace<EigenMatrix_> > new_realize_workspace() const {
        return new_known_realize_workspace();
    }

public:
    /**
     * Overrides `irlba::Matrix::new_workspace()` to enable devirtualization.
     */
    auto new_known_workspace() const {
        return std::make_unique<NormalWorkspace<EigenVector_, TValue_, TIndex_, TMatrix> >(*my_mat, my_num_threads);
    }

    /**
     * Overrides `irlba::Matrix::new_adjoint_workspace()` to enable devirtualization.
     */
    auto new_known_adjoint_workspace() const {
        return std::make_unique<NormalAdjointWorkspace<EigenVector_, TValue_, TIndex_, TMatrix> >(*my_mat, my_num_threads);
    }

    /**
     * Overrides `irlba::Matrix::new_realize_workspace()` to enable devirtualization.
     */
    auto new_known_realize_workspace() const {
        return std::make_unique<NormalRealizeWorkspace<EigenMatrix_, TValue_, TIndex_, TMatrix> >(*my_mat, my_num_threads);
    }
};

}

#endif
