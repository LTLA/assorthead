#ifndef SINGLEPP_MARKERS_HPP
#define SINGLEPP_MARKERS_HPP

#include "defs.hpp"

#include <vector>

/**
 * @file Markers.hpp
 *
 * @brief Define the `Markers` typedef.
 */

namespace singlepp {

/**
 * A vector of vectors of marker lists, with one list for each pairwise comparison between labels in the reference dataset.
 * This is used to determine which genes should be used to compute correlations in `train_single()`. 
 *
 * For a `Markers` object `markers`, let us consider the vector at `markers[0][1]`.
 * This vector should contain a list of marker genes for label 0 compared to label 1.
 * Each gene is represented as the row index of the reference expression matrix, i.e., `ref` in `train_single()`. 
 * The vector should also be sorted by the "strength" of the markers such that the earliest entries are the strongest markers for that pairwise comparison.
 * Typically, this vector is created by identifying the genes that are upregulated in label 0 compared to 1 and sorting by decreasing effect size.
 * So, for example, `markers[0][1][0]` should contain the row index of the most upregulated gene in this comparison.
 *
 * For a given reference dataset, the corresponding `Markers` object should have length equal to the number of labels in that reference.
 * Each middle vector (i.e., `markers[i]` for non-negative `i` less than the number of labels) should also have length equal to the number of labels.
 * Any innermost vector along the "diagonal" (i.e., `markers[i][i]`) is typically of zero length.
 * The innermost vectors that are not on the diagonal (i.e., `markers[i][j]` for `i != j`) may be of any positive length and should contain unique row indices.
 * Note that the length of all innermost vectors will be be capped by any non-negative `TrainSingleOptions::top` in `train_single()` and friends.
 *
 * As mentioned above, the diagonal innermost vectors are typically empty, given that it makes little sense to identify upregulated markers in a label compared to itself.
 * That said, any genes stored on the diagonal will be respected and used in all gene subsets for the corresponding label.
 * This can be exploited by advanced users to efficiently store "universal" markers for a label, i.e., markers that are applicable in all comparisons to other labels.
 *
 * @tparam Index_ Integer type for the gene (row) indices.
 */
template<typename Index_ = DefaultIndex>
using Markers = std::vector<std::vector<std::vector<Index_> > >;

}

#endif
