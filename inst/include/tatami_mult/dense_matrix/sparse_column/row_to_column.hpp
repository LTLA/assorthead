#ifndef TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_ROW_TO_COLUMN_HPP
#define TATAMI_MULT_DENSE_MATRIX_SPARSE_COLUMN_ROW_TO_COLUMN_HPP

#include <cstddef>
#include <vector>
#include <optional>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include "../../utils.hpp"

/**
 * @file row_to_column.hpp
 * @brief Sparse column-major LHS, dense column-major matrix RHS, column-major output.
 */

namespace tatami_mult {

/* See https://github.com/tatami-inc/test-multiplication/tree/master/sparse_column/dense_matrix
 * for an explanation of the choice of algorithm.
 */

/**
 * @brief Options for `multiply_sparse_column_with_dense_row_matrix_to_column_output()`.
 */
struct MultiplySparseColumnWithDenseRowMatrixToColumnOutputOptions {
    /**
     * Number of threads to use.
     * Different numbers of threads may slightly change the results due to differences in floating-point round-off error.
     */
    int num_threads = 1;

    /**
     * Block size, i.e., number of LHS columns to be loaded at once.
     * See the \f$B\f$ parameter in the @ref sparse-blocking "Blocking for sparse matrices" section for more details.
     */
    int block_size = 16;
};

/**
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightColumns_ Integer type of the number of RHS columns.
 * @tparam GetRightRow_ Functor that accepts a `LeftIndex_` and returns a pointer to an RHS row.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param right_columns Number of columns of the RHS matrix to be multiplied.
 * @param get_right_row Function that accepts a `LeftIndex_` in `[0, left.ncol())` and returns a pointer to an array of length `right_columns`.
 * The array referenced by `get_right_row(i)` represents the `i`-th row of the RHS matrix.
 * This function should be thread-safe.
 * @param[out] output Pointer to an array of length equal to `left.nrow() * right_columns`.
 * On output, this contains the matrix product in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightColumns_, typename GetRightRow_, typename Output_>
void multiply_sparse_column_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const RightColumns_ right_columns,
    GetRightRow_ get_right_row,
    Output_* const output,
    const MultiplySparseColumnWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();

    // Product must fit in a size_t in order for output to have been allocated correctly in the first place.
    // Technically, right_columns could be larger than a size_t if left_NR == 0, but the product after wraparound would still be zero, so it's fine.
    std::fill_n(output, sanisizer::product<std::size_t>(left_NR, right_columns), 0);

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_columns));
            outptr = tmp_output->data();
        }

        if (options.block_size == 1) {
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto lrange = left_ext->fetch(vbuffer.data(), ibuffer.data());
                if (lrange.number == 0) {
                    continue;
                }
                const auto rptr = get_right_row(start + cd);
                for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                    const Output_ mult = rptr[rc];
                    for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                        outptr[sanisizer::nd_offset<std::size_t>(lrange.index[x], left_NR, rc)] += mult * static_cast<Output_>(lrange.value[x]); 
                    }
                }
            }

        } else {
            // Our blocking strategy is to collect multiple LHS columns so that, for each RHS vector,
            // we can keep the corresponding output vector in cache for re-use with each LHS column.
            std::vector<std::vector<LeftValue_> > left_vbuffers;
            std::vector<std::vector<LeftIndex_> > left_ibuffers;
            std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
            std::vector<LeftIndex_> left_non_empty;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_cols);
                left_ibuffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(left_NR));
                }
                tatami::resize_container_to_Index_size(left_ranges, max_block_cols);
                left_non_empty.reserve(max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                // Only processing LHS columns (and the corresponding RHS rows) if the LHS column has some structural non-zeros.
                // If not, we just skip it altogether; no need to zero or do anything else, as we're skipping the corresponding RHS row too.
                LeftIndex_ cd_num = 0, cd_copy = cd;
                bool left_all_non_empty = true;
                left_non_empty.clear();
                do {
                    auto lrange = left_ext->fetch(left_vbuffers[cd_num].data(), left_ibuffers[cd_num].data());
                    if (lrange.number == 0) {
                        ++cd_copy;
                        left_all_non_empty = false;
                        continue;
                    }

                    left_ranges[cd_num] = std::move(lrange);
                    left_non_empty.push_back(cd_copy);
                    ++cd_num;
                    ++cd_copy;

                    if (sanisizer::is_equal(cd_num, options.block_size)) {
                        break;
                    }
                } while (cd_copy < length);

                if (left_all_non_empty) {
                    for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                        for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto& currange = left_ranges[cd_counter];
                            const Output_ mult = get_right_row(start + cd + cd_counter)[rc];
                            for (LeftIndex_ x = 0; x < currange.number; ++x) {
                                outptr[sanisizer::nd_offset<std::size_t>(currange.index[x], left_NR, rc)] += mult * static_cast<Output_>(currange.value[x]);
                            }
                        }
                    }
                } else {
                    for (auto& cdne : left_non_empty) {
                        cdne += start;
                    }
                    for (RightColumns_ rc = 0; rc < right_columns; ++rc) {
                        for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                            const auto& currange = left_ranges[cd_counter];
                            const Output_ mult = get_right_row(left_non_empty[cd_counter])[rc];
                            for (LeftIndex_ x = 0; x < currange.number; ++x) {
                                outptr[sanisizer::nd_offset<std::size_t>(currange.index[x], left_NR, rc)] += mult * static_cast<Output_>(currange.value[x]);
                            }
                        }
                    }
                }

                cd = cd_copy;
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            const auto N = tmp.size();
            for (I<decltype(N)> x = 0; x < N; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

/**
 * Overload of `multiply_sparse_column_with_dense_row_matrix_to_column_output()` for a RHS `tatami::Matrix`.
 * This function will iterate over both `left` and `right` simultaneously, realizing columns and rows respectively into memory as needed.
 *
 * @tparam LeftValue_ Numeric type of the LHS matrix value.
 * @tparam LeftIndex_ Integer type of the LHS matrix index.
 * @tparam RightValue_ Numeric type of the RHS matrix value.
 * @tparam RightIndex_ Integer type of the RHS matrix index.
 * @tparam Output_ Numeric type of the output array.
 * 
 * @param left LHS matrix to be multiplied.
 * This function is optimized for sparse matrices that prefer column access, but will work with all matrices.
 * @param right RHS matrix to be multiplied.
 * This function is optimized for dense matrices that prefer row access, but will work with all matrices.
 * The number of rows in `right` should be equal to the number of columns in `left`.
 * @param[out] output Vector of pointers, each of which points to an array of length `left.nrow()`.
 * On output, this contains the product `left * right` in column-major format.
 * @param options Further options.
 */
template<typename LeftValue_, typename LeftIndex_, typename RightValue_, typename RightIndex_, typename Output_>
void multiply_sparse_column_with_dense_row_matrix_to_column_output(
    const tatami::Matrix<LeftValue_, LeftIndex_>& left,
    const tatami::Matrix<RightValue_, RightIndex_>& right,
    Output_* const output,
    const MultiplySparseColumnWithDenseRowMatrixToColumnOutputOptions& options
) {
    const auto left_NR = left.nrow();
    const auto common_dim = left.ncol();
    const auto right_NC = right.ncol();
    std::fill_n(output, sanisizer::product<std::size_t>(left_NR, right_NC), 0);

    const bool do_parallel = options.num_threads > 1;
    std::optional<std::vector<std::optional<std::vector<Output_> > > > tmp_results;
    if (do_parallel) {
        tmp_results.emplace(sanisizer::cast<I<decltype(tmp_results->size())> >(options.num_threads - 1));
    }

    const auto num_used = tatami::parallelize([&](int t, LeftIndex_ start, LeftIndex_ length) -> void {
        auto left_ext = tatami::consecutive_extractor<true>(left, false, start, length);
        auto right_ext = tatami::consecutive_extractor<false>(right, true, start, length);

        std::optional<std::vector<Output_> > tmp_output;
        Output_* outptr; 
        if (!do_parallel || t == 0) {
            outptr = output;
        } else {
            tmp_output.emplace(sanisizer::product<I<decltype(tmp_output->size())> >(left_NR, right_NC));
            outptr = tmp_output->data();
        }

        if (options.block_size == 1) {
            auto vbuffer = tatami::create_container_of_Index_size<std::vector<LeftValue_> >(left_NR);
            auto ibuffer = tatami::create_container_of_Index_size<std::vector<LeftIndex_> >(left_NR);
            auto dbuffer = tatami::create_container_of_Index_size<std::vector<RightValue_> >(right_NC);

            for (LeftIndex_ cd = 0; cd < length; ++cd) {
                const auto lrange = left_ext->fetch(vbuffer.data(), ibuffer.data());
                const auto rptr = right_ext->fetch(dbuffer.data());

                // This skip must be done after right_ext->fetch(), otherwise the two extractors won't be in sync along the common dimension.
                // No need to zero anything as the output buffer should already be zeroed at this point.
                if (lrange.number == 0) {
                    continue;
                }

                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    const Output_ mult = rptr[rc];
                    for (LeftIndex_ x = 0; x < lrange.number; ++x) {
                        outptr[sanisizer::nd_offset<std::size_t>(lrange.index[x], left_NR, rc)] += mult * static_cast<Output_>(lrange.value[x]); 
                    }
                }
            }

        } else {
            // Our blocking strategy is to collect multiple LHS columns so that, for each RHS vector,
            // we can keep the corresponding output vector in cache for re-use with each LHS column.
            std::vector<std::vector<LeftValue_> > left_vbuffers;
            std::vector<std::vector<LeftIndex_> > left_ibuffers;
            std::vector<tatami::SparseRange<LeftValue_, LeftIndex_> > left_ranges;
            std::vector<std::vector<RightValue_> > right_dbuffers;
            std::vector<const RightValue_*> right_ptrs;
            {
                const LeftIndex_ max_block_cols = sanisizer::min(length, options.block_size);
                left_vbuffers.reserve(max_block_cols);
                left_ibuffers.reserve(max_block_cols);
                right_dbuffers.reserve(max_block_cols);
                for (LeftIndex_ cd = 0; cd < max_block_cols; ++cd) {
                    left_vbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftValue_> >(left_NR));
                    left_ibuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<LeftIndex_> >(left_NR));
                    right_dbuffers.emplace_back(tatami::cast_Index_to_container_size<std::vector<RightValue_> >(right_NC));
                }
                tatami::resize_container_to_Index_size(left_ranges, max_block_cols);
                tatami::resize_container_to_Index_size(right_ptrs, max_block_cols);
            }

            LeftIndex_ cd = 0;
            while (cd < length) {
                // Only processing LHS columns (and the corresponding RHS rows) if the LHS column has some structural non-zeros.
                // If not, we just skip it altogether; no need to zero or do anything else, as we're skipping the corresponding RHS row too.
                LeftIndex_ cd_num = 0;
                do {
                    auto lrange = left_ext->fetch(left_vbuffers[cd_num].data(), left_ibuffers[cd_num].data());
                    auto rptr = right_ext->fetch(right_dbuffers[cd_num].data());

                    // Again, this skip must be done after the RHS row is fetched, otherwise the extractors will be out of sync.
                    if (lrange.number == 0) {
                        ++cd;
                        continue;
                    }

                    left_ranges[cd_num] = std::move(lrange);
                    right_ptrs[cd_num] = rptr;
                    ++cd_num;
                    ++cd;

                    if (sanisizer::is_equal(cd_num, options.block_size)) {
                        break;
                    }
                } while (cd < length);

                for (RightIndex_ rc = 0; rc < right_NC; ++rc) {
                    for (LeftIndex_ cd_counter = 0; cd_counter < cd_num; ++cd_counter) {
                        const auto& currange = left_ranges[cd_counter];
                        const Output_ mult = right_ptrs[cd_counter][rc];
                        for (LeftIndex_ x = 0; x < currange.number; ++x) {
                            outptr[sanisizer::nd_offset<std::size_t>(currange.index[x], left_NR, rc)] += mult * static_cast<Output_>(currange.value[x]);
                        }
                    }
                }
            }
        }

        if (do_parallel && t > 0) {
            (*tmp_results)[t - 1] = std::move(tmp_output);
        }
    }, common_dim, options.num_threads);

    if (do_parallel) {
        for (int u = 1; u < num_used; ++u) {
            const auto& tmp = *((*tmp_results)[u - 1]);
            const auto N = tmp.size();
            for (I<decltype(N)> x = 0; x < N; ++x) {
                output[x] += tmp[x];
            }
        }
    }
}

}

#endif
