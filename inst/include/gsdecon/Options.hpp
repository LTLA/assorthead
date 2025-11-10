#ifndef GSDECON_OPTIONS_HPP
#define GSDECON_OPTIONS_HPP

#include "scran_blocks/scran_blocks.hpp"
#include "irlba/irlba.hpp"
#include "Eigen/Dense"

/**
 * @file Options.hpp
 * @brief Options for the **gsdecon** algorithm.
 */

namespace gsdecon {

/**
 * @brief Options for `compute()` and `compute_blocked()`.
 */
struct Options {
    /**
     * @cond
     */
    Options() {
        irlba_options.cap_number = true;
    }
    /**
     * @endcond
     */

    /**
     * Rank of the low-rank approximation.
     * Higher values can capture more biological signal at the risk of including more noise.
     * The default value of 1 assumes that each gene set only describes a single coordinated biological function.
     */
    int rank = 1;

    /**
     * Should genes be scaled to unit variance?
     * Genes with zero variance are ignored.
     * This ensures that each gene contributes equally to the PCA, favoring consistent variation across many genes rather than large variation in a few genes.
     */
    bool scale = false;

    /**
     * Policy to use for weighting batches of different size, for `compute_blocked()`.
     */
    scran_blocks::WeightPolicy block_weight_policy = scran_blocks::WeightPolicy::VARIABLE;

    /**
     * Parameters for the variable block weights for `compute_blocked()`.
     * Only used when `Options::block_weight_policy = scran_blocks::WeightPolicy::VARIABLE`.
     */
    scran_blocks::VariableWeightParameters variable_block_weight_parameters;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()` and `irlba::parallelize()`.
     */
    int num_threads = 1;

    /**
     * Whether to realize `tatami::Matrix` objects into an appropriate in-memory format before PCA.
     * This is typically faster but increases memory usage.
     */
    bool realize_matrix = true;

    /**
     * Further options to pass to `irlba::compute()`.
     */
    irlba::Options<Eigen::VectorXd> irlba_options;
};

}

#endif
