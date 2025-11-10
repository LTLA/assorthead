#ifndef GSDECON_UTILS_HPP
#define GSDECON_UTILS_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <numeric>

#include "Eigen/Dense"
#include "sanisizer/sanisizer.hpp"

#include "Results.hpp"

namespace gsdecon {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

template<typename Value_, typename Index_, typename Float_>
bool check_edge_cases(const tatami::Matrix<Value_, Index_>& matrix, const int rank, const Buffers<Float_>& output) {
    const auto NR = matrix.nrow();
    const auto NC = matrix.ncol();
    if (NR == 0) {
        std::fill_n(output.scores, NC, 0.0);
        return true;
    }

    if (NR == 1) {
        output.weights[0] = 1;
        auto ext = matrix.dense_row();
        if constexpr(std::is_same<Value_, Float_>::value) {
            const auto ptr = ext->fetch(0, output.scores);
            tatami::copy_n(ptr, NC, output.scores);
        } else {
            auto buffer = sanisizer::create<std::vector<Value_> >(NC);
            const auto ptr = ext->fetch(0, buffer.data());
            std::copy_n(ptr, NC, output.scores);
        }
        return true;
    }

    if (NC == 0) {
        std::fill_n(output.weights, NR, 0.0); 
        return true;
    }

    if (rank == 0) {
        std::fill_n(output.scores, NC, 0.0); 
        std::fill_n(output.weights, NR, 0.0); 
        return true;
    }

    return false;
}

template<typename Float_>
void process_output(const Eigen::MatrixXd& rotation, const Eigen::MatrixXd& components, bool scale, const Eigen::VectorXd& scale_v, const Buffers<Float_>& output) {
    const auto npcs = rotation.cols();
    const auto nfeat = rotation.rows();
    const auto ncells = components.cols();
    static_assert(!Eigen::MatrixXd::IsRowMajor); // just double-checking...

    /*
     * Consider a matrix of PC scores 'P' and a rotation matrix 'R', plus a centering vector 'C' and scaling vector 'S'.
     * The low-rank approximation is defined as (using R syntax):
     *
     *     L = outer(R, P) * S + C 
     *       = outer(R * S, P) + C
     *
     * Remember that we want the column means of the rank-1 approximation, so:
     *
     *     colMeans(L) = colMeans(R * S) * P + colMeans(C)
     *
     * When R and P only have 1 column, the above expression simplifies to:
     *
     *     colMeans(L) = mean(R * S) * P + colMeans(C)
     *
     * When R and P have multiple columns, we recognize that the outer product can be decomposed to the sum of the outer products of corresponding columns.
     * This allows us to easily loop over the PCs to compute each contribution to the column means.
     *
     *     colMeans(L) = outer(R_1, P_1) * S + outer(R_2 * S, P_2) * S + ... + outer(R_n, P_n) * S + colMeans(C)
     *                 = mean(R_1 * S) * P_1 + mean(R_2 * S) * P_2 + ... mean(R_n * S) * P_n + colMeans(C)
     *
     * If scale = false, then S can be dropped from the above expressions.
     */

    if (npcs > 1) {
        auto multipliers = sanisizer::create<std::vector<Float_> >(npcs);
        std::fill_n(output.weights, nfeat, 0);

        for (I<decltype(npcs)> pc = 0; pc < npcs; ++pc) {
            const auto rptr = rotation.data() + sanisizer::product_unsafe<std::size_t>(pc, nfeat); 

            for (I<decltype(nfeat)> i = 0; i < nfeat; ++i) {
                const auto val = rptr[i];
                output.weights[i] += val * val;
            }

            // Multipliers correspond to 'mean(R_x * S)' in the equations above.
            // We don't calculate the full 'mean(R_x * S) * P_x' as 'components' is column-major,
            // so it's more efficient to calculate it for each cell rather than for each PC.
            if (scale) {
                multipliers[pc] = std::inner_product(rptr, rptr + nfeat, scale_v.data(), static_cast<Float_>(0));
            } else {
                multipliers[pc] = std::accumulate(rptr, rptr + nfeat, static_cast<Float_>(0));
            }
            multipliers[pc] /= nfeat;
        }

        for (I<decltype(nfeat)> i = 0; i < nfeat; ++i) {
            output.weights[i] = std::sqrt(output.weights[i] / npcs);
        }

        // 'scores' should be filled with the (possibly block-specific) center means before this function is called.
        for (I<decltype(ncells)> c = 0; c < ncells; ++c) {
            const auto cptr = components.data() + sanisizer::product_unsafe<std::size_t>(c, npcs);
            output.scores[c] += std::inner_product(multipliers.begin(), multipliers.end(), cptr, static_cast<Float_>(0));
        }

    } else {
        const auto rptr = rotation.data();
        for (I<decltype(nfeat)> i = 0; i < nfeat; ++i) {
            output.weights[i] = std::abs(rptr[i]);
        }

        Float_ multiplier;
        if (scale) {
            multiplier = std::inner_product(rptr, rptr + nfeat, scale_v.data(), static_cast<Float_>(0));
        } else {
            multiplier = std::accumulate(rptr, rptr + nfeat, static_cast<Float_>(0));
        }
        multiplier /= nfeat;

        // 'scores' should be filled with the (possibly block-specific) center means before this function is called.
        const auto cptr = components.data();
        for (I<decltype(ncells)> c = 0; c < ncells; ++c) {
            output.scores[c] += cptr[c] * multiplier;
        }
    }
}

}

#endif
