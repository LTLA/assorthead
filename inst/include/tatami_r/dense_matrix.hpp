#ifndef TATAMI_R_DENSE_MATRIX_HPP
#define TATAMI_R_DENSE_MATRIX_HPP

#include "tatami/tatami.hpp"
#include <algorithm>

namespace tatami_r { 

template<typename InputValue_, class InputObject_,  typename CachedValue_>
void parse_dense_matrix_internal(const InputObject_& data, size_t data_start_row, size_t data_start_col, bool row, CachedValue_* cache, size_t cache_num_rows, size_t cache_num_cols) {
    size_t data_num_rows = data.rows();
    auto input = static_cast<const InputValue_*>(data.begin()) + data_start_row + data_start_col * data_num_rows;

    if (row) {
        // 'data' is a column-major matrix, but transpose() expects a row-major
        // input, so we just conceptually transpose it.
        tatami::transpose(input, cache_num_cols, cache_num_rows, data_num_rows, cache, cache_num_cols);
    } else {
        // Use an offset so that we don't accidentally create a pointer past
        // the end of the array at the final loop iteration (which is UB).
        size_t in_offset = 0;
        for (size_t c = 0; c < cache_num_cols; ++c) {
            std::copy_n(input + in_offset, cache_num_rows, cache);
            in_offset += data_num_rows;
            cache += cache_num_rows;
        }
    }
}

template<typename CachedValue_>
void parse_dense_matrix(const Rcpp::RObject& seed, size_t data_start_row, size_t data_start_col, bool row, CachedValue_* cache, size_t cache_num_rows, size_t cache_num_cols) {
    auto stype = seed.sexp_type();
    if (stype == REALSXP) {
        Rcpp::NumericMatrix y(seed);
        parse_dense_matrix_internal<double>(y, data_start_row, data_start_col, row, cache, cache_num_rows, cache_num_cols);
    } else if (stype == INTSXP) {
        Rcpp::IntegerMatrix y(seed);
        parse_dense_matrix_internal<int>(y, data_start_row, data_start_col, row, cache, cache_num_rows, cache_num_cols);
    } else if (stype == LGLSXP) {
        Rcpp::LogicalMatrix y(seed);
        parse_dense_matrix_internal<int>(y, data_start_row, data_start_col, row, cache, cache_num_rows, cache_num_cols);
    } else {
        throw std::runtime_error("unsupported SEXP type (" + std::to_string(stype) + ") from the matrix returned by 'extract_array'");
    }
}

}

#endif
