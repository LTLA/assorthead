#ifndef KAORI_COMBINATORIAL_BARCODES_SINGLE_END_HPP
#define KAORI_COMBINATORIAL_BARCODES_SINGLE_END_HPP

#include "../ScanTemplate.hpp"
#include "../BarcodeSearch.hpp"
#include "../utils.hpp"

#include <array>
#include <vector>
#include <unordered_map>

/**
 * @file CombinatorialBarcodesSingleEnd.hpp
 *
 * @brief Process single-end combinatorial barcodes.
 */

namespace kaori {

/**
 * @brief Handler for single-end combinatorial barcodes.
 *
 * In this design, the vector sequence is created from a template with multiple variable regions.
 * Each region contains a barcode from a different pool of options, where combinations are assembled randomly by library construction.
 * The idea is to use the large number of combinations to provide many unique identifiers, e.g., for cell-tracing applications.
 * This handler will capture the frequencies of each barcode combination. 
 *
 * @tparam max_size_ Maximum length of the template sequences on both reads.
 * @tparam num_variable_ Number of variable regions in the construct.
 */
template<SeqLength max_size_, int num_variable_>
class CombinatorialBarcodesSingleEnd {
public:
    /**
     * @brief Optional parameters for `CombinatorialBarcodeSingleEnd`.
     */
    struct Options {
        /**
         * Maximum number of mismatches allowed across the vector sequence.
         */
        int max_mismatches = 0;

        /** @param Whether to search only for the first match.
         * If `false`, the handler will search for the best match (i.e., fewest mismatches) instead.
         */
        bool use_first = true;

        /**
         * Strand(s) of the read sequence to search for the vector sequence.
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
     * This should contain exactly `num_variable_` variable regions.
     * @param template_length Length of the array pointed to by `template_seq`.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pools Array containing the barcode pools for each of the variable regions, in the order of their appearance in the template sequence.
     * @param options Optional parameters.
     *
     * @tparam BarcodePoolContainer Some iterable container of `BarcodePool` instances,
     * usually either a `std::vector` or a `std::array`.
     */
    template<class BarcodePoolContainer>
    CombinatorialBarcodesSingleEnd(const char* template_seq, SeqLength template_length, const BarcodePoolContainer& barcode_pools, const Options& options) :
        my_forward(search_forward(options.strand)),
        my_reverse(search_reverse(options.strand)),
        my_max_mm(options.max_mismatches),
        my_use_first(options.use_first),
        my_constant_matcher(template_seq, template_length, options.strand)
    {
        const auto& regions = my_constant_matcher.forward_variable_regions();
        if (regions.size() != num_variable_) { 
            throw std::runtime_error("expected " + std::to_string(num_variable_) + " variable regions in the constant template");
        }
        if (barcode_pools.size() != num_variable_) { 
            throw std::runtime_error("length of 'barcode_pools' should be equal to the number of variable regions");
        }

        for (int i = 0; i < num_variable_; ++i) {
            SeqLength rlen = regions[i].second - regions[i].first;
            SeqLength vlen = barcode_pools[i].length();
            if (vlen != rlen) {
                throw std::runtime_error("length of variable region " + std::to_string(i + 1) + " (" + std::to_string(rlen) + 
                    ") should be the same as its sequences (" + std::to_string(vlen) + ")");
            }
        }

        // We'll be using this later.
        for (int i = 0; i < num_variable_; ++i) {
            my_pool_size[i] = barcode_pools[i].pool().size();
        }

        SimpleBarcodeSearch::Options bopt;
        bopt.max_mismatches = options.max_mismatches;
        bopt.duplicates = options.duplicates;

        if (my_forward) {
            bopt.reverse = false;
            for (int i = 0; i < num_variable_; ++i) {
                my_forward_lib[i] = SimpleBarcodeSearch(barcode_pools[i], bopt);
            }
        }

        if (my_reverse) {
            bopt.reverse = true;
            for (int i = 0; i < num_variable_; ++i) {
                my_reverse_lib[i] = SimpleBarcodeSearch(barcode_pools[num_variable_ - i - 1], bopt);
            }
        }
    }
        
    /**
     * @param t Whether to search only for the first match.
     * If `false`, the handler will search for the best match (i.e., fewest mismatches) instead.
     *
     * @return A reference to this `CombinatorialBarcodesSingleEnd` instance.
     */
    CombinatorialBarcodesSingleEnd& set_first(bool t = true) {
        my_use_first = t;
        return *this;
    }

private:
    bool my_forward;
    bool my_reverse;
    int my_max_mm;
    bool my_use_first;

    ScanTemplate<max_size_> my_constant_matcher;
    std::array<SimpleBarcodeSearch, num_variable_> my_forward_lib, my_reverse_lib;
    std::array<BarcodeIndex, num_variable_> my_pool_size;

    std::unordered_map<std::array<BarcodeIndex, num_variable_>, Count, CombinationHash<num_variable_> > my_combinations;
    Count my_total = 0;

public:
    /**
     * @cond
     */
    struct State {
        std::unordered_map<std::array<BarcodeIndex, num_variable_>, Count, CombinationHash<num_variable_> > collected;
        Count total = 0;

        std::array<BarcodeIndex, num_variable_> temp;
        std::string buffer;

        // Default constructors should be called in this case, so it should be fine.
        std::array<typename SimpleBarcodeSearch::State, num_variable_> forward_details, reverse_details;
    };
    /**
     * @endcond
     */

private:
    std::pair<bool, int> find_match(
        bool reverse,
        const char* seq, 
        SeqLength position, 
        int obs_mismatches, 
        const std::array<SimpleBarcodeSearch, num_variable_>& libs, 
        std::array<typename SimpleBarcodeSearch::State, num_variable_>& states, 
        std::array<BarcodeIndex, num_variable_>& temp,
        std::string& buffer
    ) const {
        const auto& regions = my_constant_matcher.variable_regions(reverse); 

        for (int r = 0; r < num_variable_; ++r) {
            const auto& range = regions[r];
            auto start = seq + position;
            buffer.clear(); // clear and insert preserves buffer's existing heap allocation.
            buffer.insert(buffer.end(), start + range.first, start + range.second);

            auto& curstate = states[r];
            libs[r].search(buffer, curstate, my_max_mm - obs_mismatches);
            if (!is_barcode_index_ok(curstate.index)) {
                return std::make_pair(false, 0);
            }

            obs_mismatches += curstate.mismatches;
            if (obs_mismatches > my_max_mm) {
                return std::make_pair(false, 0);
            }

            if (reverse) {
                temp[num_variable_ - r - 1] = curstate.index;
            } else {
                temp[r] = curstate.index;
            }
        }

        return std::make_pair(true, obs_mismatches);
    }

    std::pair<bool, int> forward_match(const char* seq, const typename ScanTemplate<max_size_>::State& deets, State& state) const {
        return find_match(false, seq, deets.position, deets.forward_mismatches, my_forward_lib, state.forward_details, state.temp, state.buffer);
    }

    std::pair<bool, int> reverse_match(const char* seq, const typename ScanTemplate<max_size_>::State& deets, State& state) const {
        return find_match(true, seq, deets.position, deets.reverse_mismatches, my_reverse_lib, state.reverse_details, state.temp, state.buffer);
    }

private:
    void process_first(State& state, const std::pair<const char*, const char*>& x) const {
        auto deets = my_constant_matcher.initialize(x.first, x.second - x.first);

        while (!deets.finished) {
            my_constant_matcher.next(deets);

            if (my_forward && deets.forward_mismatches <= my_max_mm) {
                if (forward_match(x.first, deets, state).first) {
                    ++state.collected[state.temp];
                    return;
                }
            }

            if (my_reverse && deets.reverse_mismatches <= my_max_mm) {
                if (reverse_match(x.first, deets, state).first) {
                    ++state.collected[state.temp];
                    return;
                }
            }
        }
    }

    void process_best(State& state, const std::pair<const char*, const char*>& x) const {
        auto deets = my_constant_matcher.initialize(x.first, x.second - x.first);
        bool found = false;
        int best_mismatches = my_max_mm + 1;
        std::array<BarcodeIndex, num_variable_> best_id;

        auto update = [&](std::pair<bool, int> match) -> void {
            if (match.first && match.second <= best_mismatches) {
                if (match.second == best_mismatches) {
                    if (best_id != state.temp) { // ambiguous.
                        found = false;
                    }
                } else { 
                    // A further optimization at this point would be to narrow
                    // max_mm to the current 'best_mismatches'. But
                    // this probably isn't worth it.

                    found = true;
                    best_mismatches = match.second;
                    best_id = state.temp;
                }
            }
        };

        while (!deets.finished) {
            my_constant_matcher.next(deets);

            if (my_forward && deets.forward_mismatches <= my_max_mm) {
                update(forward_match(x.first, deets, state));
            }

            if (my_reverse && deets.reverse_mismatches <= my_max_mm) {
                update(reverse_match(x.first, deets, state));
            }
        }

        if (found) {
            ++state.collected[best_id];
        }
    }

public:
    /**
     * @cond
     */
    State initialize() const {
        return State();
    }

    void reduce(State& s) {
        if (my_forward) {
            for (int r = 0; r < num_variable_; ++r) {
                my_forward_lib[r].reduce(s.forward_details[r]);
            }
        }
        if (my_reverse) {
            for (int r = 0; r < num_variable_; ++r) {
                my_reverse_lib[r].reduce(s.reverse_details[r]);
            }
        }

        for (const auto& col : s.collected) {
            my_combinations[col.first] += col.second;
        }
        my_total += s.total;
        return;
    }

    void process(State& state, const std::pair<const char*, const char*>& x) const {
        if (my_use_first) {
            process_first(state, x);
        } else {
            process_best(state, x);
        }
        ++state.total;
    }

    static constexpr bool use_names = false;
    /**
     * @endcond
     */

public:
    /**
     * @return All combinations encountered by the handler, along with their frequencies.
     */
    const std::unordered_map<std::array<BarcodeIndex, num_variable_>, Count, CombinationHash<num_variable_> >& get_combinations() const {
        return my_combinations;
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
