#ifndef EMINEM_FROM_TEXT_HPP
#define EMINEM_FROM_TEXT_HPP

#include <memory>
#include <cstddef>

#include "Parser.hpp"
#include "byteme/PerByte.hpp"
#include "byteme/RawFileReader.hpp"
#include "byteme/RawBufferReader.hpp"

/**
 * @file from_text.hpp
 * @brief Read a Matrix Market file from text.
 */

namespace eminem {

/**
 * @brief Options for `parse_text_file()`.
 */
struct ParseTextFileOptions {
    /**
     * Buffer size (in bytes) to use for reading and decompression.
     */
    std::size_t buffer_size = 65536;

    /**
     * Number of threads to use to parallelize the parsing.
     */
    int num_threads = 1;

    /**
     * Block size (in bytes) to define the work for each thread.
     * Only relevant when `num_threads > 1`.
     */
    std::size_t block_size = 65536;
};

/**
 * Parse a Matrix Market text file.
 * @param path Pointer to a string containing a path to an uncompressed Matrix Market file.
 * @param options Further options.
 */
inline Parser<byteme::PerByteSerial<char> > parse_text_file(const char* path, const ParseTextFileOptions& options) {
    ParserOptions popt;
    popt.num_threads = options.num_threads;
    popt.block_size = options.block_size;

    byteme::RawFileReaderOptions topt;
    topt.buffer_size = options.buffer_size;
    auto reader = std::make_unique<byteme::RawFileReader>(path, topt);
    auto pb = std::make_unique<byteme::PerByteSerial<char> >(std::move(reader));
    return Parser<byteme::PerByteSerial<char> >(std::move(pb), popt);
}

/**
 * @brief Options for `parse_text_buffer()`.
 */
struct ParseTextBufferOptions {
    /**
     * Number of threads to use to parallelize the parsing.
     */
    int num_threads = 1;

    /**
     * Block size (in bytes) to define the work for each thread.
     * Only relevant when `num_threads > 1`.
     */
    std::size_t block_size = 65536;
};

/**
 * Parse a Matrix Market text buffer.
 * @param buffer Pointer to an array containing the contents of an uncompressed Matrix Market file.
 * @param len Length of the array referenced by `buffer`.
 * @param options Further options.
 */
inline Parser<byteme::PerByteSerial<char> > parse_text_buffer(const unsigned char* buffer, std::size_t len, const ParseTextBufferOptions& options) {
    ParserOptions popt;
    popt.num_threads = options.num_threads;
    popt.block_size = options.block_size;

    auto reader = std::make_unique<byteme::RawBufferReader>(buffer, len);
    auto pb = std::make_unique<byteme::PerByteSerial<char> >(std::move(reader));
    return Parser<byteme::PerByteSerial<char> >(std::move(pb), popt);
}

}

#endif
