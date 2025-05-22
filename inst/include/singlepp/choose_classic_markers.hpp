#ifndef SINGLEPP_CHOOSE_CLASSIC_MARKERS_HPP
#define SINGLEPP_CHOOSE_CLASSIC_MARKERS_HPP

#include "defs.hpp"

#include "tatami/tatami.hpp"

#include "Markers.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <set>
#include <cstddef>

/**
 * @file choose_classic_markers.hpp
 * @brief Classic method for choosing markers.
 */

namespace singlepp {

/**
 * Choose the number of markers in `choose_classic_markers()`.
 * The exact expression is defined as \f$500 (\frac{2}{3})^{\log_2{L}}\f$ for \f$L\f$ labels,
 * which steadily decreases the markers per comparison as the number of labels increases.
 * This aims to avoid an excessive number of features when dealing with references with many labels.
 *
 * @param num_labels Number of labels in the reference(s).
 *
 * @return An appropriate number of markers for each pairwise comparison.
 */
inline std::size_t number_of_classic_markers(std::size_t num_labels) {
    return std::round(500.0 * std::pow(2.0/3.0, std::log(static_cast<double>(num_labels)) / std::log(2.0)));
}

/**
 * @brief Options for `choose_classic_markers()`.
 */
struct ChooseClassicMarkersOptions {
    /**
     * Number of top genes to use as the marker set in each pairwise comparison.
     * If -1, this is automatically determined from the number of labels, see `number_of_classic_markers()`.
     */
    int number = -1;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `tatami::parallelize()`.
     */
    int num_threads = 1;
};

/**
 * Overload of `choose_classic_markers()` that handles multiple references. 
 * When choosing markers for label \f$A\f$ against \f$B\f$, we only consider those references with both labels \f$A\f$ and \f$B\f$.
 * For each gene, we compute the log-fold change within each reference, and then sum the log-fold changes across references;
 * the ordering of the top genes is then performed using this sum.
 * It is assumed that all references have the same number and ordering of features in their rows.
 *
 * @tparam Value_ Numeric type of matrix values.
 * @tparam Index_ Integer type of matrix row/column indices.
 * @tparam Label_ Integer type for the label identity.
 *
 * @param representatives Vector of pointers to representative **tatami** matrices.
 * Each matrix should contain no more than one column per label.
 * Each column should contain a "representative" log-expression profile for that label,
 * e.g., using the per-gene median expression across all samples assigned to that label.
 * All matrices should have the same number of rows, corresponding to the same features.
 * @param labels Vector of pointers of length equal to `representatives`.
 * Each array should be of length equal to the number of columns of the corresponding entry of `representatives`.
 * Each value of the array should specify the label for the corresponding column in its matrix.
 * Values should lie in \f$[0, L)\f$ for \f$L\f$ unique labels across all entries of `labels`.
 * @param options Further options.
 *
 * @return Top markers for each pairwise comparison between labels.
 */
template<typename Value_, typename Index_, typename Label_>
Markers<Index_> choose_classic_markers(
    const std::vector<const tatami::Matrix<Value_, Index_>*>& representatives,
    const std::vector<const Label_*>& labels,
    const ChooseClassicMarkersOptions& options)
{
    auto nrefs = representatives.size();
    if (nrefs != labels.size()) {
        throw std::runtime_error("'representatives' and 'labels' should have the same length");
    }
    if (nrefs == 0) {
        throw std::runtime_error("'representatives' should contain at least one entry");
    }
    auto ngenes = representatives.front()->nrow();

    std::size_t nlabels = 0;
    for (decltype(nrefs) r = 0; r < nrefs; ++r) {
        const auto& current = *representatives[r];

        auto nrows = current.nrow();
        if (nrows != ngenes) {
            throw std::runtime_error("all entries of 'representatives' should have the same number of rows");
        }

        auto ncols = current.ncol();
        if (ncols) {
            auto curlab = labels[r];
            nlabels = std::max(nlabels, static_cast<std::size_t>(*std::max_element(curlab, curlab + ncols)) + 1);
        }
    }

    // Generating mappings.
    std::vector<std::vector<std::pair<bool, Index_> > > labels_to_index(nrefs);
    for (decltype(nrefs) r = 0; r < nrefs; ++r) {
        auto& current = labels_to_index[r];
        current.resize(nlabels);
        auto ncols = representatives[r]->ncol();
        auto curlab = labels[r];

        for (decltype(ncols) c = 0; c < ncols; ++c) {
            auto& info = current[curlab[c]];
            if (info.first) {
                throw std::runtime_error("each label should correspond to no more than one column in each reference");
            }
            info.first = 1;
            info.second = c;
        }
    }

    Markers<Index_> output(nlabels);
    for (auto& x : output) {
        x.resize(nlabels);
    }

    int actual_number = options.number;
    if (actual_number < 0) {
        actual_number = number_of_classic_markers(nlabels);
    } 
    if (actual_number > static_cast<int>(ngenes)) {
        actual_number = ngenes;
    }

    // Generating pairs for compute; this sacrifices some memory for convenience.
    std::vector<std::pair<Label_, Label_> > pairs;
    {
        std::set<std::pair<Label_, Label_> > pairs0;
        for (decltype(nrefs) r = 0; r < nrefs; ++r) {
            auto ncols = representatives[r]->ncol();
            auto curlab = labels[r];
            for (decltype(ncols) c1 = 0; c1 < ncols; ++c1) {
                for (decltype(c1) c2 = 0; c2 < c1; ++c2) {
                    pairs0.emplace(curlab[c1], curlab[c2]);
                }
            }
        }
        pairs.insert(pairs.end(), pairs0.begin(), pairs0.end()); // already sorted by the std::set.
    }

    auto npairs = pairs.size();
    tatami::parallelize([&](int, decltype(npairs) start, decltype(npairs) len) {
        std::vector<std::pair<Value_, Index_> > sorter(ngenes);
        std::vector<Value_> rbuffer(ngenes), lbuffer(ngenes);
        std::vector<std::shared_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > > rextractors(nrefs), lextractors(nrefs);

        for (decltype(npairs) p = start, end = start + len; p < end; ++p) {
            auto curleft = pairs[p].first;
            auto curright = pairs[p].second;

            for (decltype(ngenes) g = 0; g < ngenes; ++g) {
                sorter[g].first = 0;
                sorter[g].second = g;
            }

            for (decltype(nrefs) i = 0; i < nrefs; ++i) {
                const auto& curavail = labels_to_index[i];
                auto lcol = curavail[curleft];
                auto rcol = curavail[curright];
                if (!lcol.first || !rcol.first) {
                    continue;                            
                }

                // Initialize extractors as needed.
                auto& lext = lextractors[i];
                if (!lext) {
                    lext = representatives[i]->dense_column();
                }
                auto lptr = lext->fetch(lcol.second, lbuffer.data());

                auto& rext = rextractors[i];
                if (!rext) {
                    rext = representatives[i]->dense_column();
                }
                auto rptr = rext->fetch(rcol.second, rbuffer.data());

                for (decltype(ngenes) g = 0; g < ngenes; ++g) {
                    sorter[g].first += lptr[g] - rptr[g]; 
                }
            }

            // At flip = 0, we're looking for genes upregulated in right over left,
            // as we're sorting on left - right in increasing order. At flip = 1,
            // we reverse the signs to we sort on right - left.
            for (int flip = 0; flip < 2; ++flip) {
                if (flip) {
                    for (auto& s : sorter) {
                        s.first *= -1;
                    }
                }

                // partial sort is guaranteed to be stable due to the second index resolving ties.
                std::partial_sort(sorter.begin(), sorter.begin() + actual_number, sorter.end());

                std::vector<Index_> stuff;
                stuff.reserve(actual_number);
                for (int g = 0; g < actual_number && sorter[g].first < 0; ++g) { // only negative values (i.e., positive log-fold changes for our comparison).
                    stuff.push_back(sorter[g].second); 
                }

                if (flip) {
                    output[curleft][curright] = std::move(stuff);
                } else {
                    output[curright][curleft] = std::move(stuff);
                }
            }
        }
    }, npairs, options.num_threads);

    return output;
}

/**
 * @cond
 */
// Overload for the non-const case.
template<typename Value_, typename Index_, typename Label_>
Markers<Index_> choose_classic_markers(
    const std::vector<tatami::Matrix<Value_, Index_>*>& representatives,
    const std::vector<const Label_*>& labels,
    const ChooseClassicMarkersOptions& options)
{
    std::vector<const tatami::Matrix<Value_, Index_>*> rep2(representatives.begin(), representatives.end());
    return choose_classic_markers(rep2, labels, options);
}
/**
 * @endcond
 */

/**
 * Implements the classic **SingleR** method for choosing markers from (typically bulk) reference datasets.
 * We assume that we have a matrix of representative expression profiles for each label, typically computed by averaging across all reference profiles for that label.
 * For the comparison between labels \f$A\f$ and \f$B\f$, we define the marker set as the top genes with the largest positive differences in \f$A\f$'s profile over \f$B\f$.
 * This difference can be interpreted as the log-fold change if the input matrix contains log-expression values.
 * If multiple genes have the same difference, ties are broken by favoring genes in earlier rows of the input matrix.
 * The number of top genes can either be explicitly specified or it can be automatically determined from the number of labels.
 *
 * @tparam Value_ Numeric type of matrix values.
 * @tparam Index_ Integer type of matrix row/column indices.
 * @tparam Label_ Integer type for the label identity.
 *
 * @param representative A representative matrix, containing one column per label.
 * Each column should have a representative log-expression profile for that label.
 * @param labels Pointer to an array of length equal to the number of columns in `representative`.
 * Each value of the array should specify the label for the corresponding column. 
 * Values should lie in \f$[0, L)\f$ for \f$L\f$ unique labels. 
 * @param options Further options.
 *
 * @return Top markers for each pairwise comparison between labels.
 */
template<typename Value_, typename Index_, typename Label_>
Markers<Index_> choose_classic_markers(const tatami::Matrix<Value_, Index_>& representative, const Label_* labels, const ChooseClassicMarkersOptions& options) {
    return choose_classic_markers(std::vector<const tatami::Matrix<Value_, Index_>*>{ &representative }, std::vector<const Label_*>{ labels }, options);
}

}

#endif
