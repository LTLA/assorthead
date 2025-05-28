#ifndef EMINEM_UTILS_HPP
#define EMINEM_UTILS_HPP

/**
 * @file utils.hpp
 * @brief Utilities for matrix parsing.
 */

namespace eminem {

// See https://networkrepository.com/mtx-matrix-market-format.html for legal values.

/**
 * Type of object in the Matrix Market file.
 * For the most part, vectors are treated as matrices with 1 column.
 */
enum class Object : char { MATRIX, VECTOR };

/**
 * Format of the matrix data.
 * For `COORDINATE`, each line contains the coordinates for a non-zero value.
 * For `ARRAY`, each line contains the value in column-major format.
 */
enum class Format : char { COORDINATE, ARRAY };

/**
 * Type of the matrix value field:
 * 
 * - `REAL`: integer or floating-point values, in a single field.
 * - `DOUBLE`: floating-point values only, in a single field.
 * - `INTEGER`: integer values only, in a single field.
 * - `COMPLEX`: complex values, including both real and imaginary parts in two separate fields.
 * - `PATTERN`: the presence of a matrix entry, no field is required. 
 */
enum class Field : char { REAL, DOUBLE, COMPLEX, INTEGER, PATTERN };
    
/**
 * Symmetry in the matrix.
 *
 * - `GENERAL`: no symmetric, valid for all choices of `Field`.
 * - `SYMMETRIC`: symmetric, valid for all choices of `Field`. 
 * - `SKEW_SYMMETRIC`: symmetric, valid for all choices of `Field` except `Field::PATTERN`.
 * - `HERMITIAN`: symmetric, valid for `Field::COMPLEX` only.
 */
enum class Symmetry : char { GENERAL, SYMMETRIC, SKEW_SYMMETRIC, HERMITIAN };

/**
 * @brief Details extracted from the Matrix Market banner.
 *
 * Typically generated after calling `Parser::scan_preamble()` on a Matrix Market file.
 */
struct MatrixDetails {
    /**
     * Type of the matrix object in the file.
     */
    Object object;

    /**
     * Format of the matrix. 
     */
    Format format;

    /**
     * Type of data field.
     */
    Field field;

    /**
     * Symmetry of the matrix.
     */
    Symmetry symmetry;
};

}

#endif
