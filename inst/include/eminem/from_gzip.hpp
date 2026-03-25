#ifndef EMINEM_FROM_GZIP_HPP
#define EMINEM_FROM_GZIP_HPP

#include <memory>
#include <cstddef>

#include "Parser.hpp"
#include "byteme/byteme.hpp"

/**
 * @file from_gzip.hpp
 * @brief Read a Gzipped Matrix Market file.
 */

namespace eminem {

/**
 * @cond
 */
// For back-compatibility.
// If people want to parametrize the Reader construction, they should just directly create the Parser themselves.
typedef ParserOptions ParseGzipFileOptions;
typedef ParserOptions ParseZlibBufferOptions;
typedef ParserOptions ParseSomeFileOptions;
typedef ParserOptions ParseSomeBufferOptions;
/**
 * @endcond
 */

/**
 * Parse a Gzip-compressed Matrix Market file.
 *
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param path Pointer to a string containing a path to a Gzip-compressed Matrix Market file.
 * @param options Further options.
 *
 * @return A `Parser` instance that reads from `path`.
 * This uses `Index_` as the integer type of its row/column indices.
 */
template<typename Index_ = unsigned long long>
auto parse_gzip_file(const char* path, const ParserOptions& options) {
    auto reader = std::make_unique<byteme::GzipFileReader>(path, byteme::GzipFileReaderOptions());
    return Parser<decltype(reader), Index_>(std::move(reader), options);
}

/**
 * Parse a Zlib-compressed Matrix Market buffer.
 *
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param buffer Pointer to an array containing the contents of a Zlib-compressed Matrix Market file.
 * @param len Length of the array referenced by `buffer`.
 * @param options Further options.
 *
 * @return A `Parser` instance that reads from `buffer`.
 * This uses `Index_` as the integer type of its row/column indices.
 */
template<typename Index_ = unsigned long long>
auto parse_zlib_buffer(const unsigned char* buffer, std::size_t len, const ParserOptions& options) {
    auto reader = std::make_unique<byteme::ZlibBufferReader>(buffer, len, byteme::ZlibBufferReaderOptions());
    return Parser<decltype(reader), Index_>(std::move(reader), options);
}

/**
 * Parse a possibly Gzip-compressed or uncompressed Matrix Market file.
 *
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param path Pointer to a string containing a path to a possibly-compressed Matrix Market file.
 * @param options Further options.
 *
 * @return A `Parser` instance that reads from `buffer`.
 * This uses `Index_` as the integer type of its row/column indices.
 */
template<typename Index_ = unsigned long long>
auto parse_some_file(const char* path, const ParserOptions& options) {
    std::unique_ptr<byteme::Reader> ptr;
    if (byteme::is_gzip(path)) {
        ptr.reset(new byteme::GzipFileReader(path, {}));
    } else {
        ptr.reset(new byteme::RawFileReader(path, {}));
    }
    return Parser<std::unique_ptr<byteme::Reader>, Index_>(std::move(ptr), options);
}

/**
 * Parse a possibly Zlib-compressed or uncompressed Matrix Market buffer.
 *
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param buffer Pointer to an array containing the contents of a possibly-compressed Matrix Market file.
 * @param len Length of the array referenced by `buffer`.
 * @param options Further options.
 *
 * @return A `Parser` instance that reads from `buffer`.
 * This uses `Index_` as the integer type of its row/column indices.
 */
template<typename Index_ = unsigned long long>
auto parse_some_buffer(const unsigned char* buffer, std::size_t len, const ParserOptions& options) {
    std::unique_ptr<byteme::Reader> ptr;
    if (byteme::is_zlib_or_gzip(buffer, len)) {
        ptr.reset(new byteme::ZlibBufferReader(buffer, len, {}));
    } else {
        ptr.reset(new byteme::RawBufferReader(buffer, len));
    }
    return Parser<std::unique_ptr<byteme::Reader>, Index_>(std::move(ptr), options);
}

}

#endif
