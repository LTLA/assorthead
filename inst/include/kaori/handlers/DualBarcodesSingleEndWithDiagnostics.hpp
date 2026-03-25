#ifndef KAORI_DUAL_BARCODES_SINGLE_END_WITH_DIAGNOSTICS_HPP
#define KAORI_DUAL_BARCODES_SINGLE_END_WITH_DIAGNOSTICS_HPP

#include "DualBarcodesSingleEnd.hpp"
#include "CombinatorialBarcodesSingleEnd.hpp"
#include "../utils.hpp"

/**
 * @file DualBarcodesSingleEndWithDiagnostics.hpp
 *
 * @brief Process dual barcodes with extra diagnostics.
 */

namespace kaori {

/**
 * @brief Handler for dual barcodes with extra diagnostics.
 *
 * This provides the same information as `DualBarcodesSingleEnd` but also captures the frequency of the invalid combinations.
 * These frequences can be helpful for diagnosing problems with library construction.
 *
 * @tparam max_size_ Maximum length of the template sequences on both reads.
 * @tparam num_variable_ Number of the template sequences on both reads.
 */
template<SeqLength max_size_, int num_variable_>
class DualBarcodesSingleEndWithDiagnostics { 
public:
    /**
     * @param[in] template_seq Pointer to an array containing the template sequence.
     * The template may contain any number (usually 2 or more) of variable regions.
     * @param template_length Length of the array pointed to by `template_seq`.
     * This should be less than or equal to `max_size_`.
     * @param barcode_pools Array containing pools of known barcode sequences for each of the variable regions, in the order of their appearance in the template sequence.
     * Each pool should have the same number of barcodes; corresponding entries across pools define a specific combination of barcodes. 
     * @param options Optional parameters.
     */
    DualBarcodesSingleEndWithDiagnostics(
        const char* template_seq, 
        SeqLength template_length, 
        const std::vector<BarcodePool>& barcode_pools, 
        const typename DualBarcodesSingleEnd<max_size_>::Options& options
    ) :
        my_dual_handler(
            template_seq,
            template_length,
            barcode_pools,
            options
        ),
        my_combo_handler(
            template_seq,
            template_length,
            barcode_pools, 
            [&]{
                typename CombinatorialBarcodesSingleEnd<max_size_, num_variable_>::Options combopt;
                combopt.use_first = options.use_first;

                combopt.max_mismatches = options.max_mismatches;
                combopt.strand = options.strand;

                // we allow duplicates in the trie for each individual barcode, as only the combinations are unique in the dual barcode setup.
                combopt.duplicates = DuplicateAction::FIRST; 
                return combopt;
            }()
        )
    {}

private:
    DualBarcodesSingleEnd<max_size_> my_dual_handler;
    CombinatorialBarcodesSingleEnd<max_size_, num_variable_> my_combo_handler;

public:
    /**
     *@cond
     */
    struct State {
        State() {}
        State(typename DualBarcodesSingleEnd<max_size_>::State ds, typename CombinatorialBarcodesSingleEnd<max_size_, num_variable_>::State cs) :
            dual_state(std::move(ds)), combo_state(std::move(cs)) {}

        /**
         * @cond
         */
        typename DualBarcodesSingleEnd<max_size_>::State dual_state;
        typename CombinatorialBarcodesSingleEnd<max_size_, num_variable_>::State combo_state;
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
    void process(State& state, const std::pair<const char*, const char*>& x) const {
        // Only searching for combinations if we couldn't find a proper dual barcode match.
        if (!my_dual_handler.process(state.dual_state, x)) {
            my_combo_handler.process(state.combo_state, x);
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
     * @return All invalid combinations encountered by the handler, along with their frequencies.
     * In each array, the first and second element contains the indices of known barcodes in the first and second pools, respectively.
     */
    const std::unordered_map<std::array<BarcodeIndex, num_variable_>, Count, CombinationHash<num_variable_> >& get_combinations() const {
        return my_combo_handler.get_combinations();
    }

    /**
     * @return Total number of reads processed by the handler.
     */
    Count get_total() const {
        return my_dual_handler.get_total();
    }
};

}

#endif
