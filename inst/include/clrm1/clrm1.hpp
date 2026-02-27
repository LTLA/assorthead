#ifndef CLRM1_HPP
#define CLRM1_HPP

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include <cmath>

/**
 * @file clrm1.hpp
 * @brief CLRm1 method for ADT normalization.
 */

/**
 * @namespace clrm1
 * @brief CLRm1 method for ADT normalization.
 */
namespace clrm1 {

/**
 * @brief Options for `compute()`.
 */
struct Options {
    /**
     * Number of threads to use in the calculations.
     * This is used to set `tatami_stats::sums::Options::num_threads`.
     */
    int num_threads = 1;

    /**
     * Whether to check for and remove all-zero rows before size factor calculations.
     */
    bool remove_all_zero = true;
};

/**
 * Compute CLRm1 size factors for each cell in an ADT count matrix.
 *
 * The centered log-ratio (CLR) method is defined, in R notation, as `exp(mean(log1p(x)))` where `x` is the vector of counts across all tags for a cell.
 * This aims to capture any systematic fold-difference across most tags between two cells, while diluting the compositional biases introduced by the occasional upregulated tag.
 * In the CLRm1 method, we replace `exp()` with `expm1()`, where the extra subtraction provides some symmetry and seems to improve accuracy.
 *
 * Note that the size factors computed by this function are not actually centered, despite the name of the method.
 * Centering should be done manually by the caller (e.g., using `scran_norm::center_size_factors()`) to preserve the scale of the counts after scaling normalization.
 *
 * Tags with all-zero counts across all cells are uninformative and ignored in the calculations described above.
 *
 * @tparam Value_ Type of the matrix value.
 * @tparam Index_ Integer type for the row/column indices.
 * @tparam Output_ Floating-point type for the output size factors.
 *
 * @param matrix Matrix of ADT counts, where rows are tags and columns are cells.
 * @param options Further options.
 * @param[out] output Pointer to an array of length `matrix.ncol()`.
 * On output, this stores the CLRm1 size factor for each cell in `matrix`.
 */
template<typename Value_, typename Index_, typename Output_>
void compute(const tatami::Matrix<Value_, Index_>& matrix, const Options& options, Output_* output) {
    auto ptr = tatami::wrap_shared_ptr(&matrix);

    if (options.remove_all_zero) {
        const auto num_zeros = tatami_stats::counts::zero::by_row<Index_>(matrix, [&]{
            tatami_stats::counts::zero::Options czopt;
            czopt.num_threads = options.num_threads;
            return czopt;
        }());

        const Index_ NR = matrix.nrow(), NC = matrix.ncol();
        std::vector<Index_> keep;
        for (Index_ s = 0; s < NR; ++s) {
            if (num_zeros[s] < NC) {
                keep.push_back(s);
            }
        }

        if (static_cast<Index_>(keep.size()) < NR) {
            auto sub = tatami::make_DelayedSubset(std::move(ptr), std::move(keep), true);
            ptr = std::move(sub);
        }
    }

    tatami::DelayedUnaryIsometricOperation<Output_, Value_, Index_> logmat(std::move(ptr), std::make_shared<tatami::DelayedUnaryIsometricLog1p<Value_, Output_, Index_> >());
    tatami_stats::sums::apply(false, logmat, output, [&]{
        tatami_stats::sums::Options sopt;
        sopt.num_threads = options.num_threads;
        return sopt;
    }());

    const Output_ denom = 1.0/(logmat.nrow());
    const Index_ NC = matrix.ncol();
    for (Index_ c = 0; c < NC; ++c) {
        output[c] = std::expm1(output[c] * denom);
    }
}

}

#endif
