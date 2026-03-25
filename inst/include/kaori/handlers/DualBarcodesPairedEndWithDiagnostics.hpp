#ifndef KAORI_DUAL_BARCODES_PAIRED_END_WITH_DIAGNOSTICS_HPP
#define KAORI_DUAL_BARCODES_PAIRED_END_WITH_DIAGNOSTICS_HPP

#include "DualBarcodesPairedEnd.hpp"
#include "CombinatorialBarcodesPairedEnd.hpp"
#include "../utils.hpp"

/**
 * @file DualBarcodesPairedEndWithDiagnostics.hpp
 *
 * @brief Process dual barcodes with extra diagnostics.
 */

namespace kaori {

/**
 * @brief Handler for dual barcodes with extra diagnostics.
 *
 * This provides the same information as `DualBarcodesPairedEnd` but also captures the frequency of the invalid combinations.
 * These frequences can be helpful for diagnosing problems with library construction.
 * The handler also counts the number of reads where only one barcode construct matches to a read.
 *
 * @tparam max_size_ Maximum length of the template sequences on both reads.
 */
template<SeqLength max_size_>
class DualBarcodesPairedEndWithDiagnostics { 
public:
    /**
     * @param[in] template_seq1 Pointer to a character array containing the first template sequence. 
     * This should contain exactly one variable region.
     * @param template_length1 Length of the array pointed to by `template_seq1`.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pool1 Pool of known barcode sequences for the variable region in the first template.
     * @param[in] template_seq2 Pointer to a character array containing the second template sequence. 
     * This should contain exactly one variable region.
     * @param template_length2 Length of the array pointed to by `template_seq2`.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pool2 Pool of known barcode sequences for the variable region in the second template.
     * @param options Optional parameters.
     *
     * See the `DualBarcodesPairedEnd` constructor for details.
     */
    DualBarcodesPairedEndWithDiagnostics(
        const char* template_seq1, SeqLength template_length1, const BarcodePool& barcode_pool1,
        const char* template_seq2, SeqLength template_length2, const BarcodePool& barcode_pool2, 
        const typename DualBarcodesPairedEnd<max_size_>::Options& options
    ) :
        my_dual_handler(
            template_seq1,
            template_length1,
            barcode_pool1,
            template_seq2,
            template_length2,
            barcode_pool2,
            options
        ),
        my_combo_handler(
            template_seq1, 
            template_length1, 
            barcode_pool1, 
            template_seq2, 
            template_length2, 
            barcode_pool2, 
            [&]{
                typename CombinatorialBarcodesPairedEnd<max_size_>::Options combopt;
                combopt.use_first = options.use_first;

                combopt.max_mismatches1 = options.max_mismatches1;
                combopt.strand1 = options.strand1;
                combopt.max_mismatches2 = options.max_mismatches2;
                combopt.strand2 = options.strand2;

                // we allow duplicates in the trie for each individual barcode, as only the pairs are unique in the dual barcode setup.
                combopt.duplicates = DuplicateAction::FIRST; 
                combopt.random = options.random;
                return combopt;
            }()
        )
    {}

private:
    DualBarcodesPairedEnd<max_size_> my_dual_handler;
    CombinatorialBarcodesPairedEnd<max_size_> my_combo_handler;

public:
    /**
     *@cond
     */
    struct State {
        State() {}
        State(typename DualBarcodesPairedEnd<max_size_>::State ds, typename CombinatorialBarcodesPairedEnd<max_size_>::State cs) : dual_state(std::move(ds)), combo_state(std::move(cs)) {}

        /**
         * @cond
         */
        typename DualBarcodesPairedEnd<max_size_>::State dual_state;
        typename CombinatorialBarcodesPairedEnd<max_size_>::State combo_state;
        /**
         * @endcond
         */
    };

    State initialize() const {
        return State(my_dual_handler.initialize(), my_combo_handler.initialize());
    }

    void reduce(State& s) {
        my_dual_handler.reduce(s.dual_state);
        my_combo_handler.reduce(s.combo_state);
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
        // Only searching for combinations if we couldn't find a proper dual barcode match.
        if (!my_dual_handler.process(state.dual_state, r1, r2)) {
            my_combo_handler.process(state.combo_state, r1, r2);
        }
    }
    /**
     *@endcond
     */

public:
    /**
     * @return Vector containing the frequency of each valid combination.
     * This has length equal to the number of valid dual barcode combinations (i.e., the length of `barcode_pool1` and `barcode_pool2` in the constructor).
     * Each entry contains the count for the corresponding dual barcode combination.
     */
    const std::vector<Count>& get_counts() const {
        return my_dual_handler.get_counts();
    }

    /**
     * @return Invalid combinations encountered by the handler, along with their frequencies.
     * In each array, the first and second element contains the indices of known barcodes in the first and second pools, respectively.
     */
    const std::unordered_map<std::array<BarcodeIndex, 2>, Count, CombinationHash<2> >& get_combinations() const {
        return my_combo_handler.get_combinations();
    }

    /**
     * @return Total number of read pairs processed by the handler.
     */
    Count get_total() const {
        return my_dual_handler.get_total();
    }

    /**
     * @return Number of read pairs with a valid match to the first barcode but no valid match to the second barcode.
     */
    Count get_barcode1_only() const {
        return my_combo_handler.get_barcode1_only();
    }

    /**
     * @return Number of read pairs with a valid match to the second barcode but no valid match to the first barcode.
     */
    Count get_barcode2_only() const {
        return my_combo_handler.get_barcode2_only();
    }
};

/**
 * @cond
 */
// Soft-deprecated back-compatible aliases.
template<SeqLength max_size_>
using DualBarcodesWithDiagnostics = DualBarcodesPairedEndWithDiagnostics<max_size_>;
/**
 * @endcond
 */

}

#endif
