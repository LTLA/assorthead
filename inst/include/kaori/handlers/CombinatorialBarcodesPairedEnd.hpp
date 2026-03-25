#ifndef KAORI_COMBINATORIAL_BARCODES_PAIRED_END_HPP
#define KAORI_COMBINATORIAL_BARCODES_PAIRED_END_HPP

#include "../SimpleSingleMatch.hpp"
#include "../utils.hpp"

#include <array>
#include <vector>
#include <unordered_map>

/**
 * @file CombinatorialBarcodesPairedEnd.hpp
 *
 * @brief Process paired-end combinatorial barcodes.
 */

namespace kaori {

/**
 * @brief Handler for paired-end combinatorial barcodes.
 *
 * In this design, each read contains a vector sequence created from a template with a single variable region.
 * For one read, the barcode is drawn from one pool of options, while the other read contains a barcode from another pool.
 * Combinations are assembled randomly during library construction, where the large number of combinations provide many unique identifiers for cell-tracing applications.
 * This handler will capture the frequencies of each barcode combination. 
 *
 * @tparam max_size_ Maximum length of the template sequences on both reads.
 */
template<SeqLength max_size_>
class CombinatorialBarcodesPairedEnd {
public:
    struct Options {
        /**
         * Whether to search only for the first match.
         * If `false`, the handler will search for the best match (i.e., fewest mismatches) instead.
         */
        bool use_first = true;

        /** 
         * Maximum number of mismatches allowed across the first vector sequence.
         */
        int max_mismatches1 = 0;

        /**
         * Strand(s) of the read sequence to search for the first vector sequence.
         */
        SearchStrand strand1 = SearchStrand::FORWARD;

        /** 
         * Maximum number of mismatches allowed across the second vector sequence.
         */
        int max_mismatches2 = 0;

        /**
         * Strand(s) of the read sequence to search for the second vector sequence.
         */
        SearchStrand strand2 = SearchStrand::FORWARD;

        /** 
         * How duplicated barcode sequences should be handled.
         */
        DuplicateAction duplicates = DuplicateAction::ERROR;

        /**
         * Whether the reads are randomized with respect to the first/second vector sequences.
         * If `false`, the first read is searched for the first vector sequence only, and the second read is searched for the second vector sequence only.
         * If `true`, an additional search will be performed in the opposite orientation.
         */
        bool random = false;
    };

public:
    /**
     * @param[in] template_seq1 Pointer to a character array containing the first template sequence. 
     * This should contain exactly one variable region.
     * @param template_length1 Length of the first template.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pool1 Pool of known barcode sequences for the variable region in the first template.
     * @param[in] template_seq2 Pointer to a character array containing the second template sequence. 
     * This should contain exactly one variable region.
     * @param template_length2 Length of the second template.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pool2 Pool of known barcode sequences for the variable region in the second template.
     * @param options Optional parameters.
     */
    CombinatorialBarcodesPairedEnd(
        const char* template_seq1, SeqLength template_length1, const BarcodePool& barcode_pool1,
        const char* template_seq2, SeqLength template_length2, const BarcodePool& barcode_pool2,
        const Options& options
    ) :
        my_matcher1(
            template_seq1, 
            template_length1, 
            barcode_pool1,
            [&]{
                typename SimpleSingleMatch<max_size_>::Options opt;
                opt.strand = options.strand1;
                opt.max_mismatches = options.max_mismatches1;
                opt.duplicates = options.duplicates;
                return opt;
            }()
        ),
        my_matcher2(
            template_seq2, 
            template_length2, 
            barcode_pool2, 
            [&]{
                typename SimpleSingleMatch<max_size_>::Options opt;
                opt.strand = options.strand2;
                opt.max_mismatches = options.max_mismatches2;
                opt.duplicates = options.duplicates;
                return opt;
            }()
        ),
        my_randomized(options.random),
        my_use_first(options.use_first)
    {
        my_pool_size[0] = barcode_pool1.size();
        my_pool_size[1] = barcode_pool2.size();
    }

private:
    SimpleSingleMatch<max_size_> my_matcher1, my_matcher2;
    std::array<BarcodeIndex, 2> my_pool_size;

    bool my_randomized;
    bool my_use_first = true;

    std::unordered_map<std::array<BarcodeIndex, 2>, Count, CombinationHash<2> > my_combinations;
    Count my_total = 0;
    Count my_barcode1_only = 0;
    Count my_barcode2_only = 0;

public:
    /**
     *@cond
     */
    struct State {
        State() = default;
        State(typename SimpleSingleMatch<max_size_>::State s1, typename SimpleSingleMatch<max_size_>::State s2) : search1(std::move(s1)), search2(std::move(s2)) {}

        std::unordered_map<std::array<BarcodeIndex, 2>, Count, CombinationHash<2> >collected;
        Count barcode1_only = 0;
        Count barcode2_only = 0;
        Count total = 0;

        /**
         * @cond
         */
        typename SimpleSingleMatch<max_size_>::State search1;
        typename SimpleSingleMatch<max_size_>::State search2;
        /**
         * @endcond
         */
    };

    State initialize() const {
        return State(my_matcher1.initialize(), my_matcher2.initialize());
    }

    void reduce(State& s) {
        my_matcher1.reduce(s.search1);
        my_matcher2.reduce(s.search2);
        for (const auto& col : s.collected) {
            my_combinations[col.first] += col.second;
        }
        my_total += s.total;
        my_barcode1_only += s.barcode1_only;
        my_barcode2_only += s.barcode2_only;
    }

    constexpr static bool use_names = false;
    /**
     *@endcond
     */

public:
    /**
     *@cond
     */
    void process(State& state, const std::pair<const char*, const char*>& r1, const std::pair<const char*, const char*>& r2) const {
        if (my_use_first) {
            bool m1 = my_matcher1.search_first(r1.first, r1.second - r1.first, state.search1);
            bool m2 = my_matcher2.search_first(r2.first, r2.second - r2.first, state.search2);

            if (m1 && m2) {
                std::array<BarcodeIndex, 2> key{ state.search1.index, state.search2.index };
                ++state.collected[std::move(key)];
            } else if (my_randomized) {
                bool n1 = my_matcher1.search_first(r2.first, r2.second - r2.first, state.search1);
                bool n2 = my_matcher2.search_first(r1.first, r1.second - r1.first, state.search2);
                if (n1 && n2) {
                    std::array<BarcodeIndex, 2> key{ state.search1.index, state.search2.index };
                    ++state.collected[std::move(key)];
                } else {
                    if (m1 || n1) {
                        ++state.barcode1_only;
                    } else if (m2 || n2) {
                        ++state.barcode2_only;
                    }
                }
            } else {
                if (m1) {
                    ++state.barcode1_only;
                } else if (m2) {
                    ++state.barcode2_only;
                }
            }

        } else {
            bool m1 = my_matcher1.search_best(r1.first, r1.second - r1.first, state.search1);
            bool m2 = my_matcher2.search_best(r2.first, r2.second - r2.first, state.search2);

            if (!my_randomized) {
                if (m1 && m2) {
                    std::array<BarcodeIndex, 2> key{ state.search1.index, state.search2.index };
                    ++state.collected[std::move(key)];
                } else if (m1) {
                    ++state.barcode1_only;
                } else if (m2) {
                    ++state.barcode2_only;
                }
            } else if (m1 && m2) {
                std::array<BarcodeIndex, 2> candidate{ state.search1.index, state.search2.index };
                int mismatches = state.search1.mismatches + state.search2.mismatches;

                bool n1 = my_matcher1.search_best(r2.first, r2.second - r2.first, state.search1);
                bool n2 = my_matcher2.search_best(r1.first, r1.second - r1.first, state.search2);

                if (n1 && n2) {
                    int rmismatches = state.search1.mismatches + state.search2.mismatches;
                    if (mismatches > rmismatches) {
                        std::array<BarcodeIndex, 2> key{ state.search1.index, state.search2.index };
                        ++state.collected[std::move(key)];
                    } else if (mismatches < rmismatches) {
                        ++state.collected[candidate];
                    } else if (candidate[0] == state.search1.index && candidate[1] == state.search2.index) {
                        // If the mismatches are the same, it may not be ambiguous
                        // if the indices would be the same anyway.
                        ++state.collected[candidate];
                    }
                } else {
                    ++state.collected[candidate];
                }
            } else {
                bool n1 = my_matcher1.search_best(r2.first, r2.second - r2.first, state.search1);
                bool n2 = my_matcher2.search_best(r1.first, r1.second - r1.first, state.search2);

                if (n1 && n2) {
                    std::array<BarcodeIndex, 2> key{ state.search1.index, state.search2.index };
                    ++state.collected[std::move(key)];
                } else if (m1 || n1) {
                    ++state.barcode1_only;
                } else if (m2 || n2) {
                    ++state.barcode2_only;
                }
            }
        }

        ++state.total;
    }
    /**
     *@endcond
     */

public:
    /**
     * @return Combinations encountered by the handler, along with their frequencies.
     * In each array, the first and second element contains the indices of known barcodes in the first and second pools, respectively.
     */
    const std::unordered_map<std::array<BarcodeIndex, 2>, Count, CombinationHash<2> >& get_combinations() const {
        return my_combinations;
    }

    /**
     * @return Total number of read pairs processed by the handler.
     */
    BarcodeIndex get_total() const {
        return my_total;
    }

    /**
     * @return Number of read pairs with a valid match to the first barcode but no valid match to the second barcode.
     */
    BarcodeIndex get_barcode1_only() const {
        return my_barcode1_only;
    }

    /**
     * @return Number of read pairs with a valid match to the second barcode but no valid match to the first barcode.
     */
    BarcodeIndex get_barcode2_only() const {
        return my_barcode2_only;
    }
};

}

#endif
