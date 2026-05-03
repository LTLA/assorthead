#ifndef SCRAN_NORM_CENTER_SPIKE_IN_FACTORS_HPP
#define SCRAN_NORM_CENTER_SPIKE_IN_FACTORS_HPP

#include "center_size_factors.hpp"
#include "utils.hpp"

/**
 * @file center_spike_in_factors.hpp
 * @brief Center size factors for endogenous genes and spike-in transcripts.
 */

namespace scran_norm {

/**
 * @brief Options for `center_spike_in_factors()`.
 */
struct CenterSpikeInFactorsOptions {
    /**
     * See `CenterSizeFactorsOptions::ignore_invalid`.
     */
    bool ignore_invalid = true;

    /**
     * Diagnostics of the size factors for endogenous genes,
     * see `CenterSizeFactorsOptions::diagnostics` for more details.
     */
    SizeFactorDiagnostics* endogenous_diagnostics = NULL;

    /**
     * Vector of diagnostics of the size factors for spike-in transcripts.
     * This should be of the same length as the number of spike-in sets (i.e., `spike_ins`).
     * Individual pointers may be `NULL` to omit calculation of diagnostics for each set. 
     * See `CenterSizeFactorsOptions::diagnostics` for more details.
     */
    std::vector<SizeFactorDiagnostics*> spike_in_diagnostics; 
};

/**
 * Center size factors for both endogenous genes and spike-in transcripts.
 * This ensures that the average normalized abundances of genes and spike-ins are comparable in downstream analyses.
 *
 * In practice, this function is no different to calling `center_size_factors()` separately for `endogenous` and each of `spike_ins`.
 * This function is provided for convenience and for consistency with `center_spike_in_factors_blocked()`.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 *
 * @param num Number of cells.
 * @param[in] endogenous Pointer to an array of length `num`, containing the size factor for endogenous genes for each cell.
 * On output, this contains size factors that are centered at 1.
 * @param[in] spike_ins Vector of length equal to the number of spike-in sets (e.g., ERCCs, SIRV).
 * Each entry should be a pointer to an array of length `num`, containing the size factor for its corresponding spike-in set for each cell.
 * On output, this contains size factors that are centered at 1.
 * @param options Further options.
 */
template<typename SizeFactor_>
void center_spike_in_factors(
    const std::size_t num,
    SizeFactor_* const endogenous,
    const std::vector<SizeFactor_*>& spike_ins,
    const CenterSpikeInFactorsOptions& options
) {
    CenterSizeFactorsOptions cent_opt;
    cent_opt.ignore_invalid = options.ignore_invalid;
    cent_opt.diagnostics = options.endogenous_diagnostics;
    center_size_factors(num, endogenous, cent_opt);

    const auto num_spikes = spike_ins.size();
    const bool use_diagnostics = (num_spikes == options.spike_in_diagnostics.size());
    for (I<decltype(num_spikes)> i = 0; i < num_spikes; ++i) {
        if (use_diagnostics) {
            cent_opt.diagnostics = options.spike_in_diagnostics[i];
        }
        center_size_factors(num, spike_ins[i], cent_opt);
    }
}

/**
 * @brief Options for `center_spike_in_factors_blocked()`.
 */
struct CenterSpikeInFactorsBlockedOptions {
    /**
     * See `CenterSizeFactorsBlockedOptions::ignore_invalid`.
     */
    bool ignore_invalid = true;

    /**
     * Diagnostics of the size factors for endogenous genes,
     * see `CenterSizeFactorsBlockedOptions::diagnostics` for more details.
     */
    SizeFactorDiagnostics* endogenous_diagnostics = NULL;

    /**
     * Vector of diagnostics of the size factors for spike-in transcripts.
     * This should be of the same length as the number of spike-in sets (i.e., `spike_ins`).
     * Individual pointers may be `NULL` to omit calculation of diagnostics for each set. 
     * See `CenterSizeFactorsBlockedOptions::diagnostics` for more details.
     */
    std::vector<SizeFactorDiagnostics*> spike_in_diagnostics; 

    /**
     * Strategy for handling blocks, see `CenterSizeFactorsBlockedOptions::block_mode` for more details.
     */
    CenterBlockMode block_mode = CenterBlockMode::LOWEST;
};

/**
 * Center size factors for both endogenous genes and spike-in transcripts while accounting for the block structure.
 * Specifically, the size factors of each block are scaled so that the mean size factor for each spike-in set is the same as the mean for the endogenous genes.
 * This ensures that the average normalized abundances of genes and spike-ins are comparable within each block.
 *
 * @tparam SizeFactor_ Floating-point type of the size factors.
 * @tparam Block_ Integer type for the block assignments.
 *
 * @param num Number of cells.
 * @param[in,out] endogenous Pointer to an array of length `num`, containing the size factor for endogenous genes for each cell.
 * On output, this contains size factors that are centered according to `CenterSpikeInFactorsBlockedOptions::block_mode`.
 * @param[in,out] spike_ins Vector of length equal to the number of spike-in sets (e.g., ERCCs, SIRV).
 * Each entry should be a pointer to an array of length `num`, containing the size factor for its corresponding spike-in set for each cell.
 * On output, the mean size factor within each block for each spike-in set is equal to the mean of the corresponding entries of `endogenous`.
 * @param[in] block Pointer to an array of length `num`, containing the block assignment for each cell.
 * Each assignment should be an integer in \f$[0, N)\f$ where \f$N\f$ is the total number of blocks.
 * @param options Further options.
 *
 * @return Vector of length equal to the number of blocks.
 * Each entry contains the mean of the output size factors (endogenous or spike-in) in each block.
 */
template<typename SizeFactor_, typename Block_>
std::vector<SizeFactor_> center_spike_in_factors_blocked(
    const std::size_t num,
    SizeFactor_* const endogenous,
    const std::vector<SizeFactor_*>& spike_ins,
    const Block_* const block,
    const CenterSpikeInFactorsBlockedOptions& options
) {
    CenterSizeFactorsBlockedOptions cent_opt;
    cent_opt.ignore_invalid = options.ignore_invalid;
    cent_opt.block_mode = options.block_mode;
    cent_opt.diagnostics = options.endogenous_diagnostics;
    cent_opt.report_final = true;
    auto output = center_size_factors_blocked(num, endogenous, block, cent_opt);

    cent_opt.block_mode = CenterBlockMode::CUSTOM;
    cent_opt.custom_centers = std::move(output);
    const auto num_spikes = spike_ins.size();
    const bool use_diagnostics = (num_spikes == options.spike_in_diagnostics.size());

    for (I<decltype(num_spikes)> i = 0; i < num_spikes; ++i) {
        if (use_diagnostics) {
            cent_opt.diagnostics = options.spike_in_diagnostics[i];
        }
        center_size_factors_blocked(num, spike_ins[i], block, cent_opt);
    }

    output.swap(*(cent_opt.custom_centers));
    return output;
}

}

#endif
