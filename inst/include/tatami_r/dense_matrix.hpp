#ifndef TATAMI_R_DENSE_MATRIX_HPP
#define TATAMI_R_DENSE_MATRIX_HPP

#include "Rcpp.h"
#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include <algorithm>
#include <cstddef>

namespace tatami_r { 

template<typename InputValue_, class InputObject_, typename Index_, typename CachedValue_>
void parse_dense_matrix_internal(const InputObject_& data, Index_ data_start_row, Index_ data_start_col, bool row, CachedValue_* cache, Index_ cache_num_rows, Index_ cache_num_cols) {
    Index_ data_num_rows = data.rows();
    auto input = static_cast<const InputValue_*>(data.begin()) + sanisizer::nd_offset<std::size_t>(data_start_row, data_num_rows, data_start_col);

    if (row) {
        // 'data' is a column-major matrix, but transpose() expects a row-major
        // input, so we just conceptually transpose it.
        tatami::transpose(input, cache_num_cols, cache_num_rows, data_num_rows, cache, cache_num_cols);
    } else {
        for (Index_ c = 0; c < cache_num_cols; ++c) {
            std::copy_n(
                input + sanisizer::product_unsafe<std::size_t>(c, data_num_rows),
                cache_num_rows,
                cache + sanisizer::product_unsafe<std::size_t>(c, cache_num_rows)
            );
        }
    }
}

template<typename Index_, typename CachedValue_>
void parse_dense_matrix(const Rcpp::RObject& seed, Index_ data_start_row, Index_ data_start_col, bool row, CachedValue_* cache, Index_ cache_num_rows, Index_ cache_num_cols) {
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
