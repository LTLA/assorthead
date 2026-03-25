#ifndef KAORI_SINGLE_BARCODE_SINGLE_END_HPP
#define KAORI_SINGLE_BARCODE_SINGLE_END_HPP

#include "../SimpleSingleMatch.hpp"
#include <vector>

/**
 * @file SingleBarcodeSingleEnd.hpp
 *
 * @brief Process single-end single barcodes.
 */

namespace kaori {

/**
 * @brief Handler for single-end single barcodes.
 *
 * In this design, the vector sequence is constructed from a template with a single variable region drawn from a pool of barcode sequences.
 * The construct containing the vector sequence is then subjected to single-end sequencing.
 * This handler will search the read for the vector sequence and count the frequency of each barcode.
 *
 * @tparam max_size_ Maximum length of the template sequence.
 */
template<SeqLength max_size_>
class SingleBarcodeSingleEnd {
public:
    /**
     * @brief Optional parameters for `SingleBarcodeSingleEnd`.
     */
    struct Options {
        /** 
         * Maximum number of mismatches allowed across the vector sequence.
         */
        int max_mismatches = 0;

        /** 
         * Whether to search only for the first match.
         * If `false`, the handler will search for the best match (i.e., fewest mismatches) instead.
         */
        bool use_first = true;

        /** 
         * Strand(s) of the read sequence to search.
         */
        SearchStrand strand = SearchStrand::FORWARD;

        /** 
         * How duplicated barcode sequences should be handled.
         */
        DuplicateAction duplicates = DuplicateAction::ERROR;
    };

public:
    /**
     * @param[in] template_seq Pointer to a character array containing the template sequence.
     * This should contain exactly one variable region.
     * @param template_length Length of the array pointed to by `template_seq`.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pool Known barcode sequences for the variable region.
     * @param options Optional parameters.
     */
    SingleBarcodeSingleEnd(const char* template_seq, SeqLength template_length, const BarcodePool& barcode_pool, const Options& options) :
        my_matcher(
            template_seq, 
            template_length,
            barcode_pool, 
            [&]{
                typename SimpleSingleMatch<max_size_>::Options ssopt;
                ssopt.strand = options.strand;
                ssopt.max_mismatches = options.max_mismatches;
                ssopt.duplicates = options.duplicates;
                return ssopt;
            }()
        ),
        my_counts(barcode_pool.size()),
        my_use_first(options.use_first) 
    {}

public:
    /**
     * @cond
     */
    struct State {
        State() = default;
        State(typename SimpleSingleMatch<max_size_>::State s, typename std::vector<Count>::size_type nvar) : search(std::move(s)), counts(nvar) {}

        typename SimpleSingleMatch<max_size_>::State search;
        std::vector<Count> counts;
        Count total = 0;
    };

    void process(State& state, const std::pair<const char*, const char*>& x) const {
        bool found = false;
        if (my_use_first) {
            found = my_matcher.search_first(x.first, x.second - x.first, state.search);
        } else {
            found = my_matcher.search_best(x.first, x.second - x.first, state.search);
        }
        if (found) {
            ++(state.counts[state.search.index]);
        }
        ++state.total;
    }

    static constexpr bool use_names = false;
    /**
     * @endcond
     */

public:
    /**
     * @cond
     */
    State initialize() const {
        return State(my_matcher.initialize(), my_counts.size());
    }

    void reduce(State& s) {
        my_matcher.reduce(s.search);
        for (decltype(my_counts.size()) i = 0, end = my_counts.size(); i < end; ++i) {
            my_counts[i] += s.counts[i];
        }
        my_total += s.total;
    }
    /**
     * @endcond
     */

private:
    SimpleSingleMatch<max_size_> my_matcher;
    std::vector<Count> my_counts;
    Count my_total = 0;
    bool my_use_first;

public:
    /**
     * @return Vector containing the frequency of each barcode.
     * This has length equal to the number of valid barcodes (i.e., the length of `barcode_pool` in the constructor).
     */
    const std::vector<Count>& get_counts() const {
        return my_counts;        
    }

    /**
     * @return Total number of reads processed by the handler.
     */
    Count get_total() const {
        return my_total;
    }
};

}

#endif
