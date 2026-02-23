#ifndef SINGLEPP_INTERSECTION_HPP
#define SINGLEPP_INTERSECTION_HPP

#include "defs.hpp"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>

/**
 * @file Intersection.hpp
 * @brief Create an intersection of genes.
 */

namespace singlepp {

/**
 * Intersection of genes between two datasets.
 * Each pair represents a gene that is present in both datasets.
 * The two elements of the pair represent the row indices of that gene in the respective matrices.
 *
 * Typically, the first element of the pair contains the row index of a gene in the test dataset,
 * while the second element of the pair contains the row index of the same gene in the reference dataset.
 * This convention is used by `intersect_genes()` and the relevant overloads of `train_single()` and `prepare_integrated_input()`.
 *
 * A row index for a matrix should occur no more than once in the `Intersection` object.
 * That is, all the first elements should be unique and all of the second elements should be unique.
 * Pairs may be arbitrarily ordered within the object.
 *
 * @tparam Index_ Integer type for the gene (row) indices.
 */
template<typename Index_ = DefaultIndex>
using Intersection = std::vector<std::pair<Index_, Index_> >;

/**
 * Compute the intersection of genes in the test and reference datasets.
 *
 * @tparam Index_ Integer type for the row indices of genes in each dataset.
 * Also used as the type for the number of genes.
 * @tparam Id_ Type of the gene identifier, typically an integer or string.
 *
 * @param test_nrow Number of genes (i.e., rows) in the test dataset.
 * @param[in] test_id Pointer to an array of length `test_nrow`, containing the gene identifiers for each row in the test dataset.
 * @param ref_nrow Number of genes (i.e., rows) in the reference dataset.
 * @param[in] ref_id Pointer to an array of length `ref_nrow`, containing the gene identifiers for each row in the reference dataset.
 * 
 * @return Intersection of genes between the two datasets.
 * The first entry of each pair contains the row index in the test dataset while the second entry contains the row index in the reference.
 * If duplicated identifiers are present in either of `test_id` or `ref_id`, only the first occurrence is considered in the intersection.
 */
template<typename Index_, typename Id_>
Intersection<Index_> intersect_genes(Index_ test_nrow, const Id_* test_id, Index_ ref_nrow, const Id_* ref_id) {
    std::unordered_map<Id_, Index_> ref_found;
    for (Index_ i = 0; i < ref_nrow; ++i) {
        auto current = ref_id[i];
        auto tfIt = ref_found.find(current);
        if (tfIt == ref_found.end()) { // only using the first occurrence of each ID in ref_id.
            ref_found[current] = i;
        }
    }

    Intersection<Index_> output;
    for (Index_ i = 0; i < test_nrow; ++i) {
        auto current = test_id[i];
        auto tfIt = ref_found.find(current);
        if (tfIt != ref_found.end()) {
            output.emplace_back(i, tfIt->second);
            ref_found.erase(tfIt); // only using the first occurrence of each ID in test_id; the next will not enter this clause.
        }
    }

    // This is implicitly sorted by the test indices... not that it really
    // matters, as subset_to_markers() doesn't care that it's unsorted.
    return output;
}

}

#endif
