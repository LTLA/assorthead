#ifndef SINGLEPP_MARKERS_HPP
#define SINGLEPP_MARKERS_HPP

#include "defs.hpp"

#include <vector>

/**
 * @file Markers.hpp
 *
 * @brief Define type aliases for marker lists.
 */

namespace singlepp {

/**
 * A vector of vectors of marker lists, with one list for each pairwise comparison between labels in the reference dataset.
 * This is used to determine which genes should be used to compute correlations in `train_single()`. 
 *
 * For a given reference dataset, the corresponding `PairwiseMarkers` object should have length equal to the number of labels in that reference.
 * Each middle vector (`markers[i]`) should also have length equal to the number of labels.
 * Each innermost off-diagonal vector (`markers[i][j]` for `i != j`) contains a list of marker genes for label `i` compared to label `j`.
 * Each gene is represented as the row index of the reference expression matrix, i.e., `ref` in the call to `train_single()`. 
 * Each innermost off-diagonal vector may be of any positive length and should contain unique row indices.
 * The order of indices within this vector is ignored in **singlepp** though most applications will sort markers by the strength of differential expression.
 *
 * Any innermost vector along the "diagonal" (i.e., `markers[i][i]`) is typically of zero length.
 * This is because it makes little sense to identify upregulated markers in a label compared to itself.
 * That said, any genes stored on the diagonal vectors will still be respected and used in all gene subsets for the corresponding label.
 * This can be exploited by advanced users to store "universal" markers for a label, i.e., markers that are applicable in all comparisons to other labels.
 * (See also `PerLabelMarkers` to store label-specific markers.)
 *
 * Typically, each `markers[i][j]` vector is created by identifying the top genes that are upregulated in label `i` compared to `j`. 
 * We suggest using functions like `score_markers_best()` from the [**scran_markers**](https://github.com/libscran/scran_markers) library to obtain a suitable set of top markers.
 *
 * @tparam Index_ Integer type for the gene (row) indices.
 */
template<typename Index_ = DefaultIndex>
using PairwiseMarkers = std::vector<std::vector<std::vector<Index_> > >;

/**
 * Vector of marker lists, with one list for each label in the reference dataset.
 * This is used to determine which genes should be used to compute correlations in `train_integrated()`. 
 *
 * For a `PerLabelMarkers` object `markers`, the inner vector at `markers[i]` will contain the markers for label `i`.
 * Each marker is represented as a row index into the reference matrix, i.e., `ref` in the call to `prepare_integrated_input()`.
 * Each `markers[i]` entry should have non-zero length and should contain unique row indices.
 * The order of indices within the inner vector is ignored in **singlepp** though most applications will sort markers by the strength of differential expression.
 *
 * The combination of markers in `markers[i]` is expected to distinguish `i` from all other labels.
 * We suggest using functions like `score_markers_summary()` from the [**scran_markers**](https://github.com/libscran/scran_markers) library to obtain a suitable set of markers.
 * Alternatively, if a corresponding `PairwiseMarkers` object was already constructed for the same reference (denoted `pmarkers`),
 * we can just set `markers[i]` to the union of the top marker genes from `pmarkers[i][j]` for all `j` for a given `i`. 
 */
template<typename Index_ = DefaultIndex>
using PerLabelMarkers = std::vector<std::vector<Index_> >;

/**
 * @cond
 */
// For back-compatibility.
template<typename Index_ = DefaultIndex>
using Markers = PairwiseMarkers<Index_>;
/**
 * @endcond
 */

}

#endif
