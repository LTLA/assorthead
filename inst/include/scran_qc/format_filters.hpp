#ifndef SCRAN_QC_FORMAT_FILTERS_HPP
#define SCRAN_QC_FORMAT_FILTERS_HPP

#include <vector>
#include <algorithm>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file format_filters.hpp
 * @brief Format QC filters for downstream analysis.
 */

namespace scran_qc {

/**
 * Convert the filtering vectors produced by `compute_rna_qc_filters()` and friends into formats that can be used for downstream analysis.
 * In particular, we want to slice the original feature-by-cell matrix so only the high-quality subset of cells are retained.
 * This is most easily done by using `tatami::make_DelayedSubset()` to subset the `tatami::Matrix` with the indices of the high-quality cells.
 * For this purpose, we can use `filter_index()` to convert the boolean filtering vector into a vector of sorted and unique column indices.
 *
 * @tparam Keep_ Boolean type of the filter.
 * @tparam Index_ Integer type of array indices.
 *
 * @param num_cells Number of cells in the dataset.
 * @param[in] filter Pointer to an array of length `num_cells`, indicating whether a cell is of high quality.
 * @param[out] output On output, a vector of sorted and unique indices of the cells considered to be high quality.
 */
template<typename Keep_, typename Index_>
void filter_index(const std::size_t num_cells, const Keep_* const filter, std::vector<Index_>& output) {
    output.clear();
    for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
        if (filter[i]) {
            output.push_back(i);
        }
    }
}

/**
 * Overload of `filter_index()` that returns a vector directly.
 *
 * @tparam Index_ Integer type of array indices.
 * @tparam Keep_ Boolean type of each filter modality.
 *
 * @param num_cells Number of cells in the dataset.
 * @param[in] filter Pointer to an array of length `num_cells`, indicating whether a cell is of high quality.
 *
 * @return Vector of sorted and unique indices of the cells considered to be high quality.
 */
template<typename Index_, typename Keep_>
std::vector<Index_> filter_index(const std::size_t num_cells, const Keep_* const filter) {
    std::vector<Index_> output;
#ifdef SCRAN_QC_TEST_INIT
    output.resize(10, SCRAN_QC_TEST_INIT);
#endif
    filter_index(num_cells, filter, output);
    return output;
}

/**
 * When dealing with multiple filters from different modalities (e.g., `CrisprQcFilters::filter()`, `AdtQcFilters::filter()`),
 * our default strategy is to take the intersection, i.e., we only retain cells that are considered to be high quality in all modalities.
 * This ensures that downstream analyses can be safely performed on each modality in the filtered dataset. 
 *
 * @tparam Keep_ Boolean type of each filter modality.
 * @tparam Output_ Boolean type of the output.
 *
 * @param num_cells Number of cells in the dataset.
 * @param[in] filters Vector of pointers to arrays of length `num_cells`.
 * Each array corresponds to a modality and indicates whether each cell is high quality (truthy) or not (falsey) for that modality.
 * @param[out] output Pointer to an array of length `num_cells`.
 * On output, this is filled with truthy values only for cells that are high quality in all modalities.
 */
template<typename Keep_, typename Output_>
void combine_filters(const std::size_t num_cells, const std::vector<Keep_*>& filters, Output_* const output) {
    std::copy_n(filters.front(), num_cells, output);
    const auto nfilters = filters.size();
    for (I<decltype(nfilters)> f = 1; f < nfilters; ++f) {
        const auto filt = filters[f];
        for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
            output[i] = output[i] && filt[i];
        }
    }
}

/**
 * Overload of `combine_filters()` that returns a vector directly.
 *
 * @tparam Output_ Boolean type of the output.
 * @tparam Keep_ Boolean type of each filter modality.
 *
 * @param num_cells Number of cells in the dataset.
 * @param[in] filters Vector of pointers to arrays of length `num_cells`.
 * Each array corresponds to a modality and indicates whether each cell is high quality (truthy) or not (falsey) for that modality.
 *
 * @return Vector of length `num_cells`, indicating which cells are high quality in all modalities.
 */
template<typename Output_ = unsigned char, typename Keep_>
std::vector<Output_> combine_filters(const std::size_t num_cells, const std::vector<const Keep_*>& filters) {
    auto output = sanisizer::create<std::vector<Output_> >(num_cells
#ifdef SCRAN_QC_TEST_INIT
        , SCRAN_QC_TEST_INIT
#endif
    );
    combine_filters(num_cells, filters, output.data());
    return output;
}

/**
 * This has the same behavior as `combine_filters()` followed by `filter_index()`.
 *
 * @tparam Keep_ Boolean type of each filter modality.
 *
 * @param num_cells Number of cells in the dataset.
 * @param[in] filters Vector of pointers to arrays of length `num_cells`.
 * Each array corresponds to a modality and indicates whether each cell is high quality (truthy) or not (falsey) for that modality.
 * @param[out] output On output, a vector of sorted and unique indices of the cells considered to be high quality in all modalities.
 */
template<typename Index_, typename Keep_>
void combine_filters_index(const Index_ num_cells, const std::vector<const Keep_*>& filters, std::vector<Index_>& output) {
    output.clear();

    const auto nfilters = filters.size();
    for (I<decltype(num_cells)> i = 0; i < num_cells; ++i) {
        bool keep = true;
        for (I<decltype(nfilters)> f = 0; f < nfilters; ++f) {
            if (!filters[f][i]) {
                keep = false;
                break;
            }
        }
        if (keep) {
            output.push_back(i);
        }
    }
}

/**
 * Overload of `combine_filters_index()` that returns a vector directly.
 *
 * @tparam Index_ Integer type of array indices.
 * @tparam Keep_ Boolean type of each filter modality.
 *
 * @param num_cells Number of cells in the dataset.
 * @param[in] filters Vector of pointers to arrays of length `num_cells`.
 * Each array corresponds to a modality and indicates whether each cell is high quality (truthy) or not (falsey) for that modality.
 *
 * @return Vector of sorted and unique indices of the cells considered to be high quality in all modalities.
 */
template<typename Index_, typename Keep_>
std::vector<Index_> combine_filters_index(const Index_ num_cells, const std::vector<const Keep_*>& filters) {
    std::vector<Index_> output;
#ifdef SCRAN_QC_TEST_INIT
    output.resize(10, SCRAN_QC_TEST_INIT);
#endif
    combine_filters_index(num_cells, filters, output);
    return output;
}

}

#endif
