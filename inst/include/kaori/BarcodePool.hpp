#ifndef KAORI_BARCODE_POOL_HPP
#define KAORI_BARCODE_POOL_HPP

#include <vector>
#include <string>
#include <cstddef>

#include "utils.hpp"

/**
 * @file BarcodePool.hpp
 *
 * @brief Defines the `BarcodePool` class.
 */

namespace kaori {

/**
 * @brief Pool of barcode sequences. 
 *
 * The `BarcodePool` class defines the pool of possible barcode sequences for one of the variable regions in the template sequence.
 * All sequences in this set are assumed to have the same length.
 */
class BarcodePool {
public:
    /**
     * Default constructor.
     */
    BarcodePool() = default;
    
    /**
     * @param barcode_pool Vector of pointers to arrays of length `barcode_length`.
     * Each array corresponds to one possible barcode sequence.
     * @param barcode_length Length of each barcode sequence.
     */
    BarcodePool(std::vector<const char*> barcode_pool, std::size_t barcode_length) : my_pool(std::move(barcode_pool)), my_length(barcode_length) {}

    /**
     * @param barcode_pool Vector of sequences of the same length, containing the pool of possible barcode sequences.
     * It is assumed that the lifetime of `barcode_pool` (and its strings) exceeds that of the constructed `BarcodePool`. 
     */
    BarcodePool(const std::vector<std::string>& barcode_pool) {
        if (barcode_pool.size()) {
            my_length = barcode_pool.front().size();
            my_pool.reserve(barcode_pool.size());
            for (const auto& x : barcode_pool) {
                if (x.size() != my_length) {
                    throw std::runtime_error("sequences for a given variable region should be of a constant length");
                }
                my_pool.push_back(x.c_str());
            }
        }
    }

private:
    std::vector<const char*> my_pool;
    std::size_t my_length = 0;

public:
    /**
     * @return Vector containing pointers to sequences of length equal to `length()`.
     */
    const std::vector<const char*>& pool() const {
        return my_pool;
    }

    /**
     * @return Length of each sequence in `pool()`.
     */
    SeqLength length() const {
        return my_length;
    }

    /**
     * @return Number of sequences in `pool()`.
     */
    BarcodeIndex size() const {
        return my_pool.size();
    }

    /**
     * @param i Index of a barcode sequence. 
     * @return Pointer to the `i`-th sequence in the pool.
     */
    const char* operator[](BarcodeIndex i) const {
        return my_pool[i];
    }
};

}

#endif
