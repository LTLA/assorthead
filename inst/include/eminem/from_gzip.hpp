#ifndef EMINEM_FROM_GZIP_HPP
#define EMINEM_FROM_GZIP_HPP

#include <memory>
#include <cstddef>

#include "Parser.hpp"
#include "byteme/PerByte.hpp"
#include "byteme/GzipFileReader.hpp"
#include "byteme/ZlibBufferReader.hpp"
#include "byteme/SomeFileReader.hpp"
#include "byteme/SomeBufferReader.hpp"

/**
 * @file from_gzip.hpp
 * @brief Read a Gzipped Matrix Market file.
 */

namespace eminem {

/**
 * @brief Options for `parse_gzip_file()`.
 */
struct ParseGzipFileOptions {
    /**
     * Buffer size to use for reading and decompression.
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
 * Parse a Gzip-compressed Matrix Market file.
 * @param path Pointer to a string containing a path to a Gzip-compressed Matrix Market file.
 * @param options Further options.
 */
inline Parser<byteme::PerByteSerial<char> > parse_gzip_file(const char* path, const ParseGzipFileOptions& options) {
    ParserOptions popt;
    popt.num_threads = options.num_threads;
    popt.block_size = options.block_size;

    byteme::GzipFileReaderOptions gopt;
    gopt.buffer_size = options.buffer_size;
    auto reader = std::make_unique<byteme::GzipFileReader>(path, gopt);
    auto pb = std::make_unique<byteme::PerByteSerial<char> >(std::move(reader));
    return Parser<byteme::PerByteSerial<char> >(std::move(pb), popt);
}

/**
 * @brief Options for `parse_zlib_buffer()`.
 */
struct ParseZlibBufferOptions {
    /**
     * Buffer size to use for reading.
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

    /**
     * Compression of the stream, i.e., DEFLATE, zlib or gzip.
     * This is set to 3 to perform auto-detection, see the `byteme::ZlibBufferReader` constructor.
     */
    int mode = 3;
};

/**
 * Parse a Zlib-compressed Matrix Market buffer.
 * @param buffer Pointer to an array containing the contents of a Zlib-compressed Matrix Market file.
 * @param len Length of the array referenced by `buffer`.
 * @param options Further options.
 */
inline Parser<byteme::PerByteSerial<char> > parse_zlib_buffer(const unsigned char* buffer, std::size_t len, const ParseZlibBufferOptions& options) {
    ParserOptions popt;
    popt.num_threads = options.num_threads;
    popt.block_size = options.block_size;

    byteme::ZlibBufferReaderOptions zopt;
    zopt.buffer_size = options.buffer_size;
    zopt.mode = options.mode;
    auto reader = std::make_unique<byteme::ZlibBufferReader>(buffer, len, zopt);
    auto pb = std::make_unique<byteme::PerByteSerial<char> >(std::move(reader));
    return Parser<byteme::PerByteSerial<char> >(std::move(pb), popt);
}

/**
 * @brief Options for `parse_some_file()`.
 */
struct ParseSomeFileOptions {
    /**
     * Buffer size to use for reading and decompression.
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
 * Parse a possibly Gzip-compressed or uncompressed Matrix Market file.
 * @param path Pointer to a string containing a path to a possibly-compressed Matrix Market file.
 * @param options Further options.
 */
inline Parser<byteme::PerByteSerial<char> > parse_some_file(const char* path, const ParseSomeFileOptions& options) {
    ParserOptions popt;
    popt.num_threads = options.num_threads;
    popt.block_size = options.block_size;

    byteme::SomeFileReaderOptions sopt;
    sopt.buffer_size = options.buffer_size;
    auto reader = std::make_unique<byteme::SomeFileReader>(path, sopt);
    auto pb = std::make_unique<byteme::PerByteSerial<char> >(std::move(reader));

    return Parser<byteme::PerByteSerial<char> >(std::move(pb), popt);
}

/**
 * @brief Options for `parse_some_buffer()`.
 */
struct ParseSomeBufferOptions {
    /**
     * Buffer size to use for reading and decompression.
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
 * Parse a possibly Zlib-compressed or uncompressed Matrix Market buffer.
 * @param buffer Pointer to an array containing the contents of a possibly-compressed Matrix Market file.
 * @param len Length of the array referenced by `buffer`.
 * @param options Further options.
 */
inline Parser<byteme::PerByteSerial<char> > parse_some_buffer(const unsigned char* buffer, std::size_t len, const ParseSomeBufferOptions& options) {
    ParserOptions popt;
    popt.num_threads = options.num_threads;
    popt.block_size = options.block_size;

    byteme::SomeBufferReaderOptions sopt;
    sopt.buffer_size = options.buffer_size;
    auto reader = std::make_unique<byteme::SomeBufferReader>(buffer, len, sopt);
    auto pb = std::make_unique<byteme::PerByteSerial<char> >(std::move(reader));
    return Parser<byteme::PerByteSerial<char> >(std::move(pb), popt);
}

}

#endif
