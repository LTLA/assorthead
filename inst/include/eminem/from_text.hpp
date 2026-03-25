#ifndef EMINEM_FROM_TEXT_HPP
#define EMINEM_FROM_TEXT_HPP

#include <memory>
#include <cstddef>

#include "Parser.hpp"
#include "byteme/byteme.hpp"

/**
 * @file from_text.hpp
 * @brief Read a Matrix Market file from text.
 */

namespace eminem {

/**
 * @cond
 */
// For back-compatibility.
// If people want to parametrize the Reader construction, they should just directly create the Parser themselves.
typedef ParserOptions ParseTextFileOptions;
typedef ParserOptions ParseTextBufferOptions;
/**
 * @endcond
 */

/**
 * Parse a Matrix Market text file.
 *
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param path Pointer to a string containing a path to an uncompressed Matrix Market file.
 * @param options Further options.
 *
 * @return A `Parser` instance that reads from `path`.
 * This uses `Index_` as the integer type of its row/column indices.
 */
template<typename Index_ = unsigned long long>
auto parse_text_file(const char* path, const ParserOptions& options) {
    auto reader = std::make_unique<byteme::RawFileReader>(path, byteme::RawFileReaderOptions());
    return Parser<decltype(reader), Index_>(std::move(reader), options);
}

/**
 * Parse a Matrix Market text buffer.
 *
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param buffer Pointer to an array containing the contents of an uncompressed Matrix Market file.
 * @param len Length of the array referenced by `buffer`.
 * @param options Further options.
 *
 * @return A `Parser` instance that reads from `path`.
 * This uses `Index_` as the integer type of its row/column indices.
 */
template<typename Index_ = unsigned long long>
auto parse_text_buffer(const unsigned char* buffer, std::size_t len, const ParserOptions& options) {
    auto reader = std::make_unique<byteme::RawBufferReader>(buffer, len);
    return Parser<decltype(reader), Index_>(std::move(reader), options);
}

}

#endif
