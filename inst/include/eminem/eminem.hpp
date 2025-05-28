#ifndef EMINEM_EMINEM_HPP
#define EMINEM_EMINEM_HPP

#include "Parser.hpp"
#include "from_text.hpp"

#if __has_include("zlib.h")
#include "from_gzip.hpp"
#endif

/**
 * @file eminem.hpp
 * @brief Umbrella header for the **eminem** library.
 *
 * If ZLib is not available, all of the Zlib-related headers are omitted.
 * This will skip classes such as the `GzipFileParser` and `SomeBufferParser`.
 */

/**
 * @namespace eminem
 * @brief Classes and methods for parsing Matrix Market files.
 */
namespace eminem {}

#endif
