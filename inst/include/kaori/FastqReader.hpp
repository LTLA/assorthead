#ifndef KAORI_FASTQ_READER_HPP
#define KAORI_FASTQ_READER_HPP

#include <cctype>
#include <vector>
#include <stdexcept>

#include "byteme/byteme.hpp"

#include "utils.hpp"

/**
 * @file FastqReader.hpp
 *
 * @brief Defines the `FastqReader` class.
 */

namespace kaori {

/**
 * @brief Stream reads from a FASTQ file.
 *
 * Pretty much what it says on the tin.
 * Multi-line sequence and quality strings are supported.
 * The name of each read is only considered up to the first whitespace.
 *
 * @tparam Pointer_ Pointer to a class that serves as a source of input bytes.
 * The pointed-to class should satisfy the `byteme::Reader` interface; it may also be a concrete `byteme::Reader` subclass to enable devirtualization. 
 * Either a smart or raw pointer may be supplied depending on how the caller wants to manage the lifetime of the pointed-to object. 
 */
template<typename Pointer_>
class FastqReader {
public:
    /**
     * @param p Pointer to a text stream containing a FASTQ file.
     * @param buffer_size Size of the buffer size for parsing the FASTQ file.
     * Larger values improve speed at the cost of increased memory usage.
     */
    FastqReader(
        Pointer_ p,
        std::size_t buffer_size = 65536 /* default for back-compatibility */
    ) :
        my_pb(p, buffer_size)
    {
        my_sequence.reserve(200);
        my_name.reserve(200);
        my_okay = my_pb.valid();
    }

    /**
     * Extract details for the next read in the file.
     *
     * @return Whether or not a record was successfully extracted.
     * If `true`, `get_sequence()` and `get_name()` may be used.
     * If `false`, the end of the file was reached.
     */
    bool operator()() {
        // Quitting early if the buffer is already empty. 
        if (!my_okay) {
            return false;
        }

        auto init_line = my_line_count;

        // Processing the name. This should be on a single line, hopefully.
        my_name.clear();
        char val = my_pb.get();
        if (val != '@') {
            throw std::runtime_error("read name should start with '@' (starting line " + std::to_string(init_line + 1) + ")");
        }

        val = advance_and_check();
        while (!std::isspace(val)) {
            my_name.push_back(val);
            val = advance_and_check();
        }

        while (val != '\n') {
            val = advance_and_check();
        }
        ++my_line_count;

        // Processing the sequence itself until we get to a '+'.
        my_sequence.clear();
        val = advance_and_check();
        while (1) {
            if (val == '\n') {
                val = advance_and_check();
                if (val == '+') {
                    break;
                }
            }
            my_sequence.push_back(val);
            val = advance_and_check();
        }
        ++my_line_count;

        // Line 3 should be a single line; starting with '+' is implicit from above.
        val = advance_and_check();
        while (val != '\n') {
            val = advance_and_check();
        } 
        ++my_line_count;

        // Processing the qualities. Extraction is allowed to fail if we're at
        // the end of the file. Note that we can't check for '@' as a
        // delimitor, as this can be a valid score, so instead we check at each
        // newline whether we've reached the specified length, and quit if so.
        SeqLength seq_length = my_sequence.size(), qual_length = 0;
        my_okay = false;

        while (my_pb.advance()) {
            val = my_pb.get();
            if (val != '\n') {
                ++qual_length;
            } else if (qual_length >= seq_length) {
                my_okay = my_pb.advance(); // sneak past the newline.
                break;
            }
        }

        if (qual_length != seq_length) {
            // Technically qual_length could overflow as the length of the quality string is unbounded.
            // This would cause this check to not be triggered (unlike the other overflow cases where we should get a bad_alloc). 
            // In practice, who cares, and besides, the quality strings aren't even used for anything here. 
            throw std::runtime_error("non-equal lengths for quality and sequence strings (starting line " + std::to_string(init_line + 1) + ")");
        }

        ++my_line_count;
        return true;
    }

private:
    byteme::SerialBufferedReader<char, Pointer_> my_pb;

    char advance_and_check() {
        if (!my_pb.advance()) {
            throw std::runtime_error("premature end of the file at line " + std::to_string(my_line_count + 1));
        }
        return my_pb.get();
    }

private:
    std::vector<char> my_sequence;
    std::vector<char> my_name;
    bool my_okay;
    unsigned long long my_line_count = 0; // guarantee at least 64 bits for the line counter.

public:
    /**
     * @return Vector containing the sequence for the current read.
     * This should only be called if `load()` returns true.
     */
    const std::vector<char>& get_sequence() const {
        return my_sequence;
    }

    /**
     * @return Vector containing the name for the current read.
     * Note that the name is considered to end at the first whitespace on the line.
     * This should only be called if `load()` returns true.
     */
    const std::vector<char>& get_name() const {
        return my_name;
    }
};

}

#endif
