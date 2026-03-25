#ifndef KAORI_VARIABLE_LIBRARY_HPP
#define KAORI_VARIABLE_LIBRARY_HPP

#include "BarcodePool.hpp"
#include "MismatchTrie.hpp"
#include "utils.hpp"

#include <cstddef>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>

/**
 * @file BarcodeSearch.hpp
 *
 * @brief Search for barcode sequences.
 */

namespace kaori {

/** 
 * @cond
 */
template<typename Trie_>
inline void fill_library(const std::vector<const char*>& options, std::unordered_map<std::string, BarcodeIndex>& exact, Trie_& trie, bool reverse) {
    std::size_t len = trie.length();
    auto nopt = options.size();

    for (decltype(nopt) i = 0; i < nopt; ++i) {
        auto ptr = options[i];

        std::string current;
        if (!reverse) {
            current = std::string(ptr, ptr + len);
        } else {
            current.reserve(len);
            for (size_t j = 0; j < len; ++j) {
                current += complement_base<true, true>(ptr[len - j - 1]);
            }
        }

        // Note that this must be called, even if the sequence is duplicated;
        // otherwise the trie's internal counter will not be properly incremented.
        auto status = trie.add(current.c_str());

        if (!status.has_ambiguous) {
            if (!status.is_duplicate || status.duplicate_replaced) {
                exact[current] = i;
            } else if (status.duplicate_cleared) {
                exact[current] = STATUS_UNMATCHED;
            }
        }
    }

    trie.optimize();
    return;
}
/** 
 * @endcond
 */

/**
 * @brief Search against known barcodes.
 *
 * Given an input sequence, this class performs exact and mismatch-aware searches against a pool of known barcodes.
 * Mismatches may be distributed anywhere along the length of the sequence, see `AnyMismatches` for details.
 * Caching is used to avoid redundant work when a mismatching sequence has been previously encountered.
 */
class SimpleBarcodeSearch {
public:
    /**
     * @brief Optional parameters for `SimpleBarcodeSearch`.
     */
    struct Options {
        /**
         * Maximum number of mismatches for any search performed by `SimpleBarcodeSearch::search`.
         */
        int max_mismatches = 0;

        /** 
         * Whether to reverse-complement the barcode sequences before indexing them.
         */
        bool reverse = false;

        /** 
         * How duplicated barcode sequences should be handled.
         */
        DuplicateAction duplicates = DuplicateAction::ERROR;
    };

public:
    /**
     * Default constructor.
     * This is only provided for composition purposes; methods of this class should only be called on properly constructed instance.
     */
    SimpleBarcodeSearch() = default;

    /**
     * @param barcode_pool Pool of barcode sequences.
     * @param options Further options. 
     */
    SimpleBarcodeSearch(const BarcodePool& barcode_pool, const Options& options) : 
        my_trie(barcode_pool.length(), options.duplicates), 
        my_max_mm(options.max_mismatches) 
    {
        fill_library(barcode_pool.pool(), my_exact, my_trie, options.reverse);
    }

private:
    AnyMismatches my_trie;
    int my_max_mm;
    std::unordered_map<std::string, BarcodeIndex> my_exact;

    struct CacheEntry {
        CacheEntry() = default;
        CacheEntry(BarcodeIndex index, int mismatches) : index(index), mismatches(mismatches) {}
        BarcodeIndex index;
        int mismatches;
    };
    std::unordered_map<std::string, CacheEntry> my_cache;

public:
    /**
     * @brief State of the search.
     *
     * This contains both the results of `search()` for a given input sequence,
     * as well as cached mismatches to optimize searches for future inputs.
     */
    struct State {
        /**
         * Index of the known barcode that matches best to the input sequence in `search()` (i.e., fewest total mismatches).
         * If all barcodes have more mismatches than `allowed_mismatches`, `STATUS_UNMATCHED` is returned.
         * If multiple barcodes share the same lowest number of mismatches (not greater than `allowed_mismatches`), `STATUS_AMBIGUOUS` is returned.
         */
        BarcodeIndex index = 0;

        /**
         * Number of mismatches with the matching known sequence.
         * This should be ignored if `index == STATUS_UNMATCHED`,
         * as the search will terminate early without computing the exact number of mismatches if `allowed_mismatches` is exceeded.
         */
        int mismatches = 0;

        /**
         * @cond
         */
        std::unordered_map<std::string, CacheEntry> cache;
        /**
         * @endcond
         */
    };

    /**
     * Initialize the search state for thread-safe execution.
     *
     * @return A new state object for use in `search()` and `reduce()`.
     */
    State initialize() const {
        return State();
    }

    /**
     * Incorporate the mismatch cache from `state` into the cache for this `SimpleBarcodeSearch` instance.
     * This enables regular synchronization of the cache (and its related search optimizations) across threads.
     *
     * @param state A state object generated by `initialize()`.
     * Typically this has already been used in `search()` at least once.
     */
    void reduce(State& state) {
        my_cache.merge(state.cache);
        state.cache.clear();
    }

public:
    /**
     * Search the known sequences in the barcode pool against an input sequence.
     * The number of allowed mismatches is equal to the `Options::max_mismatches` specified in the constructor.
     * 
     * @param search_seq The input sequence to use for searching.
     * This is expected to have the same length as the known sequences.
     * @param state A `State` object generated by `initialize()`.
     * On return, `state` is filled with the details of the best-matching barcode sequence, if any exists.
     */
    void search(const std::string& search_seq, State& state) const {
        search(search_seq, state, my_max_mm);
        return;
    }

    /**
     * Search the known sequences in the barcode pool for an input sequence with a user-supplied number of allowed mismatches.
     * It is expected that `allowed_mismatches` is no greater than the `Options::max_mismatches` specified in the constructor. 
     * Lower values can improve efficiency in situations where some mismatches have already been "used up", e.g., by matching the template sequence in `ScanTemplate`.
     *
     * @param search_seq The input sequence to use for searching.
     * This is expected to have the same length as the known sequences.
     * @param state A state object generated by `initialize()`.
     * On return, `state` is filled with the details of the best-matching barcode sequence, if any exists.
     * @param allowed_mismatches Allowed number of mismatches.
     * This should not be greater than the `Options::max_mismatches` specified in the constructor.
     */
    void search(const std::string& search_seq, State& state, int allowed_mismatches) const {
        auto it = my_exact.find(search_seq);
        if (it != my_exact.end()) {
            state.index = it->second;
            state.mismatches = 0;
            return;
        }

        auto set_from_cache = [&](const CacheEntry& cached) -> void {
            if (cached.mismatches > allowed_mismatches) {
                // technically cached.mismatches is only a lower bound if index == UNMATCHED,
                // but if it's already UNMATCHED, then the result will be UNMATCHED either way.
                state.index = STATUS_UNMATCHED;
            } else {
                state.index = cached.index;
            }
            state.mismatches = cached.mismatches;
        };

        auto cIt = my_cache.find(search_seq);
        if (cIt != my_cache.end()) {
            set_from_cache(cIt->second);
            return;
        }

        auto lIt = state.cache.find(search_seq);
        if (lIt != state.cache.end()) {
            set_from_cache(lIt->second);
            return;
        }

        auto missed = my_trie.search(search_seq.c_str(), allowed_mismatches);
        if (is_barcode_index_ok(missed.index)) {
            // No need to check against allowed_mismatches, as we explicitly searched for that in the trie.
            state.index = missed.index;
            state.mismatches = missed.mismatches;
            state.cache[search_seq] = CacheEntry(missed.index, missed.mismatches);
            return;
        }

        // The trie search breaks early when it hits allowed_mismatches, but
        // allowed_mismatches might be different across calls to search(). If
        // we break early and report a miss, the miss will be cached and
        // returned in cases where there is a higher cap (and thus might
        // actually be a hit). As such, we should only store a miss in the
        // cache when the requested number of mismatches is equal to the
        // maximum number of mismatches that was specified in the constructor.
        //
        // Of course, if the search failed because of ambiguity, then it would
        // have failed even if we were searching with maximum mismatches;
        // so we happily cache that.
        if (allowed_mismatches == my_max_mm || missed.index == STATUS_AMBIGUOUS) {
            state.cache[search_seq] = CacheEntry(missed.index, missed.mismatches);
        }

        state.index = missed.index;
        state.mismatches = missed.mismatches;
    }
};

/**
 * @brief Search against known barcode sequences with segmented mismatches.
 *
 * Given an input sequence, this class performs exact and mismatch-aware searches for known sequences with "segmented" mismatches.
 * Specifically, we split the sequence into segments whereby the number of mismatches in each segment is limited - see `SegmentedMismatches` for details. 
 * We use caching to avoid redundant work when a mismatching sequence has been previously encountered.
 *
 * @tparam num_segments_ Number of segments to consider.
 */
template<int num_segments_>
class SegmentedBarcodeSearch {
public:
    /**
     * @brief Optional parameters for a `SegmentedBarcodeSearch`.
     */
    struct Options {
        /**
         * @param max_mismatch_per_segment Maximum number of mismatches per segment.
         * This is used to fill `max_mismatches`.
         */
        Options(int max_mismatch_per_segment = 0) {
            max_mismatches.fill(max_mismatch_per_segment);
        }
        
        /**
         * Maximum number of mismatches in each segment for `SegmentedBarcodeSearch::search()`.
         * All values should be non-negative.
         * Defaults to an all-zero array in the `Options()` constructor.
         */
        std::array<int, num_segments_> max_mismatches;

        /** 
         * Whether to reverse-complement the barcode sequences before indexing them.
         * Note that, even if `reverse = true`, the segment lengths in the `SegmentedBarcodeSearch()` constructor and the numbers of mismatches in `max_mismatches`
         * should still be ordered according to the forward strand.
         */
        bool reverse = false;

        /** 
         * How duplicated barcode sequences should be handled.
         */
        DuplicateAction duplicates = DuplicateAction::ERROR;
    };

public:
    /**
     * Default constructor.
     * This is only provided for composition purposes; methods of this class should only be called on properly constructed instance.
     */
    SegmentedBarcodeSearch() = default;

    /**
     * @param barcode_pool Pool of barcode sequences.
     * @param segments Size of each segment.
     * All values should be positive and their sum should be equal to the barcode length in `barcode_pool`.
     * @param options Optional parameters.
     */
    SegmentedBarcodeSearch(
        const BarcodePool& barcode_pool, 
        std::array<SeqLength, num_segments_> segments, 
        const Options& options
    ) : 
        my_trie(
            [&]{
                if (options.reverse) {
                    std::reverse(segments.begin(), segments.end());
                }
                return segments;
            }(),
            options.duplicates
        ), 
        my_max_mm(
            [&]{
                auto copy = options.max_mismatches;
                if (options.reverse) {
                    std::reverse(copy.begin(), copy.end());
                }
                return copy;
            }()
        )
    {
        if (barcode_pool.length() != my_trie.length()) {
            throw std::runtime_error("variable sequences should have the same length as the sum of segment lengths");
        }
        fill_library(barcode_pool.pool(), my_exact, my_trie, options.reverse);
    }

private:
    SegmentedMismatches<num_segments_> my_trie;
    std::array<int, num_segments_> my_max_mm;
    std::unordered_map<std::string, BarcodeIndex> my_exact;

    struct CacheEntry {
        CacheEntry() = default;
        CacheEntry(BarcodeIndex index, int mismatches, std::array<int, num_segments_> per_segment) :
            index(index), mismatches(mismatches), per_segment(per_segment) {}
        BarcodeIndex index;
        int mismatches;
        std::array<int, num_segments_> per_segment;
    };
    std::unordered_map<std::string, CacheEntry> my_cache;

public:
    /**
     * @brief State of the search.
     *
     * This contains both the results of `search()` for any given input sequence,
     * as well as cached mismatches to optimize searches for future inputs.
     */
    struct State {
        /**
         * Index of the known sequence that matches best to the input sequence in `search()` (i.e., fewest total mismatches).
         * If all barcodes have more mismatches than `allowed_mismatches`, `STATUS_UNMATCHED` is returned.
         * If multiple barcodes share the same lowest number of mismatches (not greater than `allowed_mismatches`), `STATUS_AMBIGUOUS` is returned.
         */
        BarcodeIndex index = 0;

        /**
         * Total number of mismatches with the matching known sequence, summed across all segments.
         * This should be ignored if `index == STATUS_UNMATCHED`,
         * as the search will terminate early without computing the exact number of mismatches if `allowed_mismatches` is exceeded.
         */
        int mismatches = 0;

        /**
         * Number of mismatches in each segment.
         * This should be ignored if `index == STATUS_UNMATCHED`,
         * as the search will terminate early without computing the exact number of mismatches if `allowed_mismatches` is exceeded.
         */
        std::array<int, num_segments_> per_segment;
        
        /**
         * @cond
         */
        State() : per_segment() {}

        std::unordered_map<std::string, CacheEntry> cache;
        /**
         * @endcond
         */
    };

    /**
     * Initialize the search state for thread-safe execution.
     *
     * @return A new state object for use in `search()` and `reduce()`.
     */
    State initialize() const {
        return State();
    }

    /**
     * Incorporate the mismatch cache from `state` into the cache for this `SegmentedBarcodeSEarch` instance.
     * This enables regular synchronization of the cache (and its related search optimizations) across threads.
     *
     * @param state A state object generated by `initialize()`.
     * Typically this has already been used in `search()`.
     */
    void reduce(State& state) {
        my_cache.merge(state.cache);
        state.cache.clear();
    }

public:
    /**
     * Search the known sequences in the barcode pool against an input sequence.
     * The number of allowed mismatches in each segment is equal to the `Options::max_mismatches` specified in the constructor.
     * 
     * @param search_seq The input sequence to use for searching.
     * This is expected to have the same length as the known sequences.
     * @param state A state object generated by `initialize()`.
     * On return, `state` is filled with the details of the best-matching barcode sequence, if any exists.
     */
    void search(const std::string& search_seq, State& state) const {
        search(search_seq, state, my_max_mm);
        return;
    }

    /**
     * Search the known sequences in the barcode pool for an input sequence with a user-supplied number of allowed mismatches in each segment.
     * It is expected that each element of `allowed_mismatches` is no greater than the corresponding element of `Options::max_mismatches` specified in the constructor. 
     * Lower values can improve efficiency in situations where some mismatches have already been "used up", e.g., by matching the template sequence in `ScanTemplate`.
     *
     * @param search_seq The input sequence to use for searching.
     * This is expected to have the same length as the known sequences.
     * @param state A state object generated by `initialize()`.
     * On return, `state` is filled with the details of the best-matching barcode sequence, if any exists.
     * @param allowed_mismatches Allowed number of mismatches in each segment.
     * Each value should not be greater than the corresponding value of `Options::max_mismatches` specified in the constructor.
     */
    void search(const std::string& search_seq, State& state, std::array<int, num_segments_> allowed_mismatches) const {
        auto it = my_exact.find(search_seq);
        if (it != my_exact.end()) {
            state.index = it->second;
            state.mismatches = 0;
            std::fill_n(state.per_segment.begin(), num_segments_, 0);
            return;
        }

        auto set_from_cache = [&](const CacheEntry& cached) -> void {
            state.mismatches = cached.mismatches;
            state.per_segment = cached.per_segment;
            for (int s = 0; s < num_segments_; ++s) {
                if (cached.per_segment[s] > allowed_mismatches[s]) {
                    // technically cached.mismatches is only a lower bound if index == UNMATCHED,
                    // but if it's already UNMATCHED, then the result will be UNMATCHED either way.
                    state.index = STATUS_UNMATCHED;
                    return;
                }
            }
            state.index = cached.index;
        };

        auto cIt = my_cache.find(search_seq);
        if (cIt != my_cache.end()) {
            set_from_cache(cIt->second);
            return;
        }

        auto lIt = state.cache.find(search_seq);
        if (lIt != state.cache.end()) {
            set_from_cache(lIt->second);
            return;
        }

        auto missed = my_trie.search(search_seq.c_str(), allowed_mismatches);
        if (is_barcode_index_ok(missed.index)) {
            // No need to check against allowed_mismatches, as we explicitly searched for that in the trie.
            state.index = missed.index;
            state.mismatches = missed.mismatches;
            state.per_segment = missed.per_segment;
            state.cache[search_seq] = CacheEntry(missed.index, missed.mismatches, missed.per_segment);
            return;
        }

        // The trie search breaks early when it hits allowed_mismatches, but
        // the allowed_mismatches might be different across calls to search().
        // If we break early and report a miss, the miss will be cached and
        // returned in cases where there is a higher cap (and thus might
        // actually be a hit). As such, we should only store a miss in the
        // cache when the requested number of mismatches is equal to the
        // maximum number of mismatches that was specified in the constructor.
        //
        // Of course, if the search failed because of ambiguity, then it would
        // have failed even if we were searching with maximum mismatches;
        // so we happily cache that.
        if (allowed_mismatches == my_max_mm || missed.index == STATUS_AMBIGUOUS) {
            state.cache[search_seq] = CacheEntry(missed.index, missed.mismatches, missed.per_segment);
        }

        state.index = missed.index;
        state.mismatches = missed.mismatches;
        state.per_segment = missed.per_segment;
    }
};

}

#endif
