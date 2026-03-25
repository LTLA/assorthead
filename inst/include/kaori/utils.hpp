#ifndef KAORI_UTILS_HPP
#define KAORI_UTILS_HPP

#include <bitset>
#include <vector>
#include <array>
#include <cstddef>
#include <limits>
#include <functional>

/**
 * @file utils.hpp
 * @brief Utilites for sequence matching.
 */

namespace kaori {

/**
 * How to deal with duplicated sequences in the pool of known barcodes.
 *
 * - `FIRST`: keep the first encountered instance of a set of duplicates.
 * - `LAST`: keep the last encountered instance of a set of duplicates.
 * - `NONE`: do not keep any duplicate sequences.
 * - `ERROR`: throw an error upon observing a duplicate sequence.
 */
enum class DuplicateAction : char { FIRST, LAST, NONE, ERROR };

/**
 * Strands of the read sequence to search.
 */
enum class SearchStrand : char { FORWARD, REVERSE, BOTH };

/**
 * Integer type for the sequence lengths.
 * This is used for barcodes, templates and reads.
 */
typedef std::size_t SeqLength;

/**
 * Integer type for barcode indices in a pool of known barcodes (specifically, inside a `BarcodePool`).
 * This can have the special values `STATUS_UNMATCHED` and `STATUS_AMBIGUOUS`, see `is_barcode_index_ok()` to check for these error codes.
 */
typedef typename std::vector<const char*>::size_type BarcodeIndex; // we use the size_type from the BarcodePool's internal vector of barcodes.

/**
 * No match to a known barcode.
 */
inline constexpr BarcodeIndex STATUS_UNMATCHED = static_cast<BarcodeIndex>(-1);

/**
 * Ambiguous match to two or more known barcodes. 
 */
inline constexpr BarcodeIndex STATUS_AMBIGUOUS = static_cast<BarcodeIndex>(-2);

/**
 * @param index A barcode index.
 * @return Whether `index` corresponds to an actual barcode.
 * If false, `index` represents one of the error codes, i.e., `STATUS_MISSING` or `STATUS_AMBIGUOUS`.
 */
inline bool is_barcode_index_ok(BarcodeIndex index) {
    return index < STATUS_AMBIGUOUS;
}

/**
 * Integer type to count the frequency of each barcode.
 */
typedef unsigned long long Count;

/**
 * @cond
 */
inline bool search_forward(SearchStrand x) {
    return x == SearchStrand::FORWARD || x == SearchStrand::BOTH;
}

inline bool search_reverse(SearchStrand x) {
    return x == SearchStrand::REVERSE || x == SearchStrand::BOTH;
}

template<bool allow_n_ = false, bool allow_iupac_ = false>
char complement_base(char b) {
    char output;
    switch (b) {
        case 'A': case 'a':
            output = 'T';
            break;
        case 'C': case 'c':
            output = 'G';
            break;
        case 'G': case 'g':
            output = 'C';
            break;
        case 'T': case 't':
            output = 'A';
            break;

        case 'N': case 'n':
            if constexpr(allow_n_ || allow_iupac_) {
                output = 'N';
                break;
            }

        case 'R': case 'r':
            if constexpr(allow_iupac_) {
                output = 'Y';
                break;
            }
        case 'Y': case 'y':
            if constexpr(allow_iupac_) {
                output = 'R';
                break;
            }
        case 'S': case 's':
            if constexpr(allow_iupac_) {
                output = 'S'; // S = A/T, so complement is just itself.
                break;
            }
        case 'W': case 'w':
            if constexpr(allow_iupac_) {
                output = 'W'; // W = C/G, so complement is just itself.
                break;
            }
        case 'K': case 'k':
            if constexpr(allow_iupac_) {
                output = 'M';
                break;
            }
        case 'M': case 'm':
            if constexpr(allow_iupac_) {
                output = 'K';
                break;
            }

        case 'B': case 'b':
            if constexpr(allow_iupac_) {
                output = 'V'; // B can't be A, so complement can't be T ==> V.
                break;
            }
        case 'D': case 'd':
            if constexpr(allow_iupac_) {
                output = 'H'; // D can't be C, so complement can't be G ==> H.
                break;
            }
        case 'H': case 'h':
            if constexpr(allow_iupac_) {
                output = 'D'; // H can't be G, so complement can't be C ==> D.
                break;
            }
        case 'V': case 'v':
            if constexpr(allow_iupac_) {
                output = 'B'; // V can't be T, so complement can't be A ==> B.
                break;
            }

        default:
            throw std::runtime_error("cannot complement unknown base '" + std::string(1, b) + "'");
    }
    return output;
}

inline bool is_standard_base(char b) {
    bool okay = false;
    switch (b) {
        case 'A': case 'a': 
        case 'C': case 'c': 
        case 'G': case 'g': 
        case 'T': case 't':
            okay = true;
            break;
    }
    return okay;
}

template<size_t N>
void shift_hash(std::bitset<N>& x) {
    x <<= 4;
}

template<size_t N>
void add_base_to_hash(std::bitset<N>& x, char b) {
    shift_hash(x);
    switch (b) {
        case 'A': case 'a':
            x.set(0);
            break;
        case 'C': case 'c':
            x.set(1);
            break;
        case 'G': case 'g':
            x.set(2);
            break;
        case 'T': case 't':
            x.set(3);
            break;
        default:
            throw std::runtime_error("unknown base '" + std::string(1, b) + "'");
            break;
    }
    return;
}

template<size_t N>
void add_other_to_hash(std::bitset<N>& x) {
    shift_hash(x);
    x.set(0);
    x.set(1);
    x.set(2);
    x.set(3);
    return;
}

inline constexpr int NUM_BASES = 4;

/**
 * @brief Hash a combination of barcode indices.
 *
 * Create a hash from an array representing a combination of barcode indices, usually for use in a `std::unordered_map`.
 * This involves hashing each individual index and then iteratively applying the Boost `hash_combine` function. 
 * 
 * @tparam num_variable_ Number of variable regions, each with their own barcode indices.
 */
template<int num_variable_>
class CombinationHash {
public:
    /**
     * @param key Array of barcode indices.
     * @return The hash value of `key`.
     */
    std::size_t operator()(const std::array<BarcodeIndex, num_variable_>& key) const {
        unsigned long long seed = 0; 

        for (int v = 0; v < num_variable_; ++v) {
            seed = [&](){
                // Adapted from https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x/78509978#78509978, which was in turn derived from Boost.ContainerHash:
                // https://github.com/boostorg/container_hash/blob/ee5285bfa64843a11e29700298c83a37e3132fcd/include/boost/container_hash/detail/hash_mix.hpp#L67
                // We hard-code it to use the 64-bit implementation for simplicity, given that a ULL is guaranteed to have at least 64 bits.
                unsigned long long x = seed + 0x9e3779b9 + key[v];
                constexpr unsigned long long m = 0xe9846af9b1a615d;
                x ^= x >> 32;
                x *= m;
                x ^= x >> 32;
                x *= m;
                x ^= x >> 28;
                return x;
            }();
        }

        return seed;
    }
};

}

#endif
