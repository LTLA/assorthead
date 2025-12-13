#ifndef TATAMI_R_DENSE_MATRIX_HPP
#define TATAMI_R_DENSE_MATRIX_HPP

#include "Rcpp.h"
#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include <algorithm>
#include <cstddef>

namespace tatami_r { 

/* It's worth stressing here that 'data' is just a big matrix of data that we pulled out of R,
 * saving time by avoiding repeated invocations of the R interpreter (at the expense of memory).
 * Here, we need to split up that big blob into each cache buffer to make it easier to manage. 
 * Each call to parse_dense_matrix() just extracts part of that big blob into one cache pointer;
 * specifically, we want to extract rows from '[data_start_row, data_start_row + cache_num_rows)'
 * and columns from '[data_start_column, data_start_column + cache_num_columns)'.
 */
template<typename InputValue_, class InputObject_, typename Index_, typename CachedValue_>
void parse_dense_matrix_internal(
    const InputObject_& data,
    const Index_ data_start_row,
    const Index_ data_start_col,
    const bool row,
    CachedValue_* const cache,
    const Index_ cache_num_rows,
    const Index_ cache_num_cols
) {
    const Index_ data_num_rows = data.rows();
    const auto input = static_cast<const InputValue_*>(data.begin()) + sanisizer::nd_offset<std::size_t>(data_start_row, data_num_rows, data_start_col);

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
void parse_dense_matrix(    
    const Rcpp::RObject& seed,
    const Index_ data_start_row,
    const Index_ data_start_col,
    const bool row,
    CachedValue_* const cache,
    const Index_ cache_num_rows,
    const Index_ cache_num_cols
) {
    const auto stype = seed.sexp_type();
    if (stype == REALSXP) {
        const Rcpp::NumericMatrix y(seed);
        parse_dense_matrix_internal<double>(y, data_start_row, data_start_col, row, cache, cache_num_rows, cache_num_cols);
    } else if (stype == INTSXP) {
        const Rcpp::IntegerMatrix y(seed);
        parse_dense_matrix_internal<int>(y, data_start_row, data_start_col, row, cache, cache_num_rows, cache_num_cols);
    } else if (stype == LGLSXP) {
        const Rcpp::LogicalMatrix y(seed);
        parse_dense_matrix_internal<int>(y, data_start_row, data_start_col, row, cache, cache_num_rows, cache_num_cols);
    } else {
        throw std::runtime_error("unsupported SEXP type (" + std::to_string(stype) + ") from the matrix returned by 'extract_array'");
    }
}

}

#endif
