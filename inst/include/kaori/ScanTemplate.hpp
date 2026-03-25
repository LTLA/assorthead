#ifndef KAORI_SCAN_TEMPLATE_HPP
#define KAORI_SCAN_TEMPLATE_HPP

#include <bitset>
#include <vector>
#include <stdexcept>
#include <string>

#include "utils.hpp"

/**
 * @file ScanTemplate.hpp
 *
 * @brief Defines the `ScanTemplate` class.
 */

namespace kaori {

/**
 * @brief Scan a read sequence for the template sequence.
 *
 * When searching for barcodes, **kaori** first searches for a "template sequence" in the read sequence.
 * The template sequence contains one or more variable regions flanked by constant regions.
 * The template is realized into the "vector sequence" (i.e., the final sequence on the vector construct)
 * by replacing each variable region with one sequence from a pool of barcodes.
 *
 * This class will scan the read sequence to find a location that matches the constant regions of the template, give or take any number of substitutions.
 * Multiple locations on the read may match the template, provided `next()` is called repeatedly.
 * For efficiency, the search itself is done after converting all base sequences into a bit encoding.
 * The maximum size of this encoding is determined at compile-time by the `max_length` template parameter.
 *
 * Once a match is found, the sequence of the read at each variable region can be matched against a pool of known barcode sequences.
 * See other classes like `SimpleBarcodeSearch` and `SegmentedBarcodeSearch` for details.
 *
 * @tparam max_size_ Maximum length of the template sequence.
 */
template<SeqLength max_size_>
class ScanTemplate { 
private:
    static constexpr SeqLength N = max_size_ * 4;

public:
    /**
     * Default constructor.
     * This is only provided to enable composition, the resulting object should not be used until it is copy-assigned to a properly constructed instance.
     */
    ScanTemplate() = default;

    /**
     * @param[in] template_seq Pointer to a character array containing the template sequence.
     * Constant sequences should only contain `A`, `C`, `G` or `T` (or their lower-case equivalents).
     * Variable regions should be marked with `-`.
     * @param template_length Length of the array pointed to by `template_seq`.
     * This should be positive and less than or equal to `max_size_`.
     * @param strand Strand(s) of the read sequence to search.
     */
    ScanTemplate(const char* template_seq, SeqLength template_length, SearchStrand strand) :
        my_length(template_length),
        my_forward(search_forward(strand)),
        my_reverse(search_reverse(strand))
    {
        if (my_length > max_size_) {
            throw std::runtime_error("maximum template size should be " + std::to_string(max_size_) + " bp");
        }

        if (my_forward) {
            for (SeqLength i = 0; i < my_length; ++i) {
                char b = template_seq[i];
                if (b != '-') {
                    add_base_to_hash(my_forward_ref, b);
                    add_mask_to_hash(my_forward_mask);
                    my_forward_mask_ambiguous.set(i);
                } else {
                    shift_hash(my_forward_ref);
                    shift_hash(my_forward_mask);
                    add_variable_base(my_forward_variables, i);
                }
            }
        } else {
            // Forward variable regions are always defined.
            for (SeqLength i = 0; i < my_length; ++i) {
                char b = template_seq[i];
                if (b == '-') {
                    add_variable_base(my_forward_variables, i);
                }
            }
        }

        if (my_reverse) {
            for (SeqLength i = 0; i < my_length; ++i) {
                char b = template_seq[my_length - i - 1];
                if (b != '-') {
                    add_base_to_hash(my_reverse_ref, complement_base(b));
                    add_mask_to_hash(my_reverse_mask);
                    my_reverse_mask_ambiguous.set(i);
                } else {
                    shift_hash(my_reverse_ref);
                    shift_hash(my_reverse_mask);
                    add_variable_base(my_reverse_variables, i);
                }
            }
        }
    }

public:
    /**
     * @brief Details on the current match to the read sequence.
     */
    struct State {
        /**
         * Position of the match to the template on the read sequence.
         * This should only be used after a call to `next()`.
         */
        SeqLength position = static_cast<SeqLength>(-1); // set to -1 so that the first call to next() overflows to 0.

        /**
         * Number of mismatches on the forward strand at the current position.
         * This should only be used after a call to `next()`, and only if the forward strand is searched.
         */
        int forward_mismatches = 0;

        /**
         * Number of mismatches on the reverse strand at the current position.
         * This should only be used after a call to `next()`, and only if the reverse strand is searched.
         */
        int reverse_mismatches = 0;

        /**
         * Whether the match is at the end of the read sequence.
         * If `true`, `next()` should not be called.
         */
        bool finished = false;

        /**
         * @cond
         */
        std::bitset<N> state;
        const char * seq;
        SeqLength len;

        std::bitset<N/4> ambiguous; // we only need a yes/no for the ambiguous state, so we can use a smaller bitset.
        SeqLength last_ambiguous; // contains the position of the most recent ambiguous base; should only be read if any_ambiguous = true.
        bool any_ambiguous = false; // indicates whether ambiguous.count() > 0.
        /**
         * @endcond
         */
    };

    /**
     * Begin a new search for the template in a read sequence.
     *
     * @param[in] read_seq Pointer to an array containing the read sequence.
     * @param read_length Length of the read sequence.
     *
     * @return An empty state object.
     * If its `finished` member is `false`, it should be passed to `next()` before accessing its other members.
     * If `true`, the read sequence was too short for any match to be found.
     */
    State initialize(const char* read_seq, SeqLength read_length) const {
        State out;
        out.seq = read_seq;
        out.len = read_length;

        if (my_length <= read_length) {
            for (SeqLength i = 0; i < my_length - 1; ++i) {
                char base = read_seq[i];

                if (is_standard_base(base)) {
                    add_base_to_hash(out.state, base);

                    if (out.any_ambiguous) {
                        out.ambiguous <<= 1;
                    }
                } else {
                    add_other_to_hash(out.state);

                    if (out.any_ambiguous) {
                        out.ambiguous <<= 1;
                    } else {
                        out.any_ambiguous = true;
                    }
                    out.ambiguous.set(0);
                    out.last_ambiguous = i;
                }
            }
        } else {
            out.finished = true;
        }

        return out;
    }

    /**
     * Find the next match in the read sequence.
     * The first invocation will search for a match at position 0;
     * this can be repeatedly called until `match.finished` is `true`.
     *
     * @param state A state object produced by `initialize()`.
     * On return, `state` is updated with the details of the current match at a particular position on the read sequence.
     */
    void next(State& state) const {
        SeqLength right = state.position + my_length;
        char base = state.seq[right];

        if (is_standard_base(base)) {
            add_base_to_hash(state.state, base); // no need to trim off the end, the mask will handle that.
            if (state.any_ambiguous) {
                state.ambiguous <<= 1;

                // If the last ambiguous position is equal to 'position', the
                // ensuing increment to the latter will shift it out of the
                // hash... at which point, we've got no ambiguity left.
                if (state.last_ambiguous == state.position) {
                    state.any_ambiguous = false;
                }
            }

        } else {
            add_other_to_hash(state.state);

            if (state.any_ambiguous) {
                state.ambiguous <<= 1;
            } else {
                state.any_ambiguous = true;
            }
            state.ambiguous.set(0);
            state.last_ambiguous = right;
        }

        ++state.position;
        full_match(state);
        if (right + 1 == state.len) {
            state.finished = true;
        }

        return;
    }

private:
    std::bitset<N> my_forward_ref, my_forward_mask;
    std::bitset<N> my_reverse_ref, my_reverse_mask;
    SeqLength my_length;
    bool my_forward, my_reverse;

    std::bitset<N/4> my_forward_mask_ambiguous; // we only need a yes/no for whether a position is an ambiguous base, so we can use a smaller bitset.
    std::bitset<N/4> my_reverse_mask_ambiguous;

    static void add_mask_to_hash(std::bitset<N>& current) {
        shift_hash(current);
        current.set(0);
        current.set(1);
        current.set(2);
        current.set(3);
        return;
    }

    static int strand_match(const State& match, const std::bitset<N>& ref, const std::bitset<N>& mask, const std::bitset<N/4>& mask_ambiguous) {
        // pop count here is equal to the number of non-ambiguous mismatches *
        // 2 + number of ambiguous mismatches * 3. This is because
        // non-ambiguous bases are encoded by 1 set bit per 4 bases (so 2 are
        // left after a XOR'd mismatch), while ambiguous mismatches are encoded
        // by all set bits per 4 bases (which means that 3 are left after XOR).
        int pcount = ((match.state & mask) ^ ref).count(); 

        if (match.any_ambiguous) {
            int acount = (match.ambiguous & mask_ambiguous).count();
            return (pcount - acount) / 2; // i.e., acount + (pcount - acount * 3) / 2;
        } else {
            return pcount / 2;
        }
    }

    void full_match(State& match) const {
        if (my_forward) {
            match.forward_mismatches = strand_match(match, my_forward_ref, my_forward_mask, my_forward_mask_ambiguous);
        }
        if (my_reverse) {
            match.reverse_mismatches = strand_match(match, my_reverse_ref, my_reverse_mask, my_reverse_mask_ambiguous);
        }
    }

private:
    std::vector<std::pair<SeqLength, SeqLength> > my_forward_variables, my_reverse_variables;

public:
    /**
     * @return A vector of pairs where each pair specifies the start and one-past-the-end position of each variable region in the template.
     * Coordinates are reported relative to the start of the template.
     * Pairs are ordered by the start positions.
     */
    const std::vector<std::pair<SeqLength, SeqLength> >& forward_variable_regions() const {
        return my_forward_variables;
    }

    /**
     * @return A vector of pairs where each pair specifies the start and one-past-the-end position of each variable region in the reverse-complented template.
     * Coordinates are reported relative to the start of the template.
     * Pairs are ordered by the start positions.
     */
    const std::vector<std::pair<SeqLength, SeqLength> >& reverse_variable_regions() const {
        return my_reverse_variables;
    } 

    /**
     * @param reverse Whether to return variable regions on the reverse-complemented template.
     * @return Convenient alias for `forward_variable_regions()` or `reverse_variable_regions()` depending on `reverse`.
     */
    const std::vector<std::pair<SeqLength, SeqLength> >& variable_regions(bool reverse) const {
        if (reverse) {
            return reverse_variable_regions();
        } else {
            return forward_variable_regions();
        }
    } 

private:
    static void add_variable_base(std::vector<std::pair<SeqLength, SeqLength> >& variables, SeqLength i) {
        if (!variables.empty()) {
            auto& last = variables.back().second;
            if (last == i) {
                ++last;
                return;
            }
        }
        variables.emplace_back(i, i + 1);
        return;
    }
};

}

#endif
