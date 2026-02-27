#ifndef TATAMI_STATS_COUNTS_HPP
#define TATAMI_STATS_COUNTS_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file counts.hpp
 *
 * @brief Compute row and column counts from a `tatami::Matrix`.
 */

namespace tatami_stats {

/**
 * @brief Functions for computing dimension-wise counts.
 * @namespace tatami_stats::counts
 */
namespace counts {

/**
 * Count the number of values that satisfy the `condition` in each element of a chosen dimension.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`.
 * @tparam Condition_ Function that accepts a single `Value_` and returns a `bool`.
 *
 * @param row Whether to perform the count within each row.
 * If false, the count is performed within each column instead.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows (if `row = true`) or columns (otherwise).
 * On output, this will contain the row/column variances.
 * @param num_threads Number of threads to use, for parallelization via `tatami::parallelize()`.
 * @param condition Function to indicate whether a value should be counted. 
 * This function is also responsible for handling any NaNs that might be present in `p`.
 */
template<typename Value_, typename Index_, typename Output_, class Condition_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, int num_threads, Condition_ condition) {
    const Index_ dim = (row ? mat.nrow() : mat.ncol());
    const Index_ otherdim = (row ? mat.ncol() : mat.nrow());
    std::fill(output, output + dim, 0);

    if (mat.prefer_rows() == row) {
        if (mat.sparse()) {
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            bool count_zero = condition(0);

            tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(otherdim);
                auto ext = tatami::consecutive_extractor<true>(mat, row, start, len, opt);

                for (Index_ x = 0; x < len; ++x) {
                    auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                    Output_ target = 0;
                    for (Index_ j = 0; j < range.number; ++j) {
                        target += condition(range.value[j]);
                    }
                    if (count_zero) {
                        target += otherdim - range.number;
                    }
                    output[x + start] = target;
                }
            }, dim, num_threads);

        } else {
            tatami::parallelize([&](int, Index_ start, Index_ len) -> void {
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(otherdim);
                auto ext = tatami::consecutive_extractor<false>(mat, row, start, len);

                for (Index_ x = 0; x < len; ++x) {
                    auto ptr = ext->fetch(xbuffer.data());
                    Output_ target = 0;
                    for (Index_ j = 0; j < otherdim; ++j) {
                        target += condition(ptr[j]);
                    }
                    output[x + start] = target;
                }
            }, dim, num_threads);
        }

    } else {
        auto threaded_output = sanisizer::create<std::vector<std::vector<Output_> > >(num_threads - 1);
        const auto get_output_ptr = [&](const int thread) -> Output_* {
            if (thread == 0) {
                return output;
            }
            auto& outvec = threaded_output[thread - 1];
            outvec.resize(dim);
            return outvec.data();
        };

        int num_used;
        if (mat.sparse()) {
            tatami::Options opt;
            opt.sparse_ordered_index = false;
            bool count_zero = condition(0);

            num_used = tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
                auto ibuffer = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);
                auto ext = tatami::consecutive_extractor<true>(mat, !row, start, len, opt);
                auto curoutput = get_output_ptr(thread);
                auto nonzeros = tatami::create_container_of_Index_size<std::vector<Index_> >(dim);

                for (Index_ x = 0; x < len; ++x) {
                    auto range = ext->fetch(xbuffer.data(), ibuffer.data());
                    for (Index_ j = 0; j < range.number; ++j) {
                        auto idx = range.index[j];
                        curoutput[idx] += condition(range.value[j]);
                        ++(nonzeros[idx]);
                    }
                }

                if (count_zero) {
                    for (Index_ d = 0; d < dim; ++d) {
                        curoutput[d] += len - nonzeros[d];
                    }
                }
            }, otherdim, num_threads);

        } else {
            num_used = tatami::parallelize([&](int thread, Index_ start, Index_ len) -> void {
                auto xbuffer = tatami::create_container_of_Index_size<std::vector<Value_> >(dim);
                auto ext = tatami::consecutive_extractor<false>(mat, !row, start, len);
                auto curoutput = get_output_ptr(thread);

                for (Index_ x = 0; x < len; ++x) {
                    auto ptr = ext->fetch(xbuffer.data());
                    for (Index_ j = 0; j < dim; ++j) {
                        curoutput[j] += condition(ptr[j]);
                    }
                }
            }, otherdim, num_threads);
        }

        for (int thread = 1; thread < num_used; ++thread) {
            const auto& curout = threaded_output[thread - 1];
            for (Index_ d = 0; d < dim; ++d) {
                output[d] += curout[d];
            }
        }
    }
}

/**
 * @cond
 */
// Back-compatibility only.
template<typename Value_, typename Index_, typename Output_, class Condition_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, Output_* output, int num_threads, Condition_ condition) {
    apply(row, *p, output, num_threads, std::move(condition));
}
/**
 * @endcond
 */

/**
 * @brief Functions for counting NaNs on each dimension.
 * @namespace tatami_stats::counts::nan
 */
namespace nan {

/**
 * @brief NaN-counting options.
 */
struct Options {
    /**
     * Number of threads to use when obtaining counts across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;

};

/**
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 *
 * @param row Whether to obtain a count for each row.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows.
 * On output, this will store the number of NaNs in each row.
 * @param nopt Counting options.
 */
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const Options& nopt) {
    counts::apply(row, mat, output, nopt.num_threads, [](Value_ x) -> bool { return std::isnan(x); });
}

/**
 * @cond
 */
// Back-compatibility only.
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, Output_* output, const Options& nopt) {
    apply(row, *p, output, nopt);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for row NaN counts.
 *
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param nopt Counting options.
 *
 * @return A vector of length equal to the number of rows, containing the number of NaNs in each row.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat, const Options& nopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.nrow());
    apply(true, mat, output.data(), nopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility only.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p, const Options& nopt) {
    return by_row<Output_>(*p, nopt);
}
/**
 * @endcond
 */

/**
 * Overload with default options.
 *
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @return A vector of length equal to the number of rows, containing the number of NaNs in each row.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat) {
    return by_row<Output_>(mat, Options());
}

/**
 * @cond
 */
// Back-compatibility only.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p) {
    return by_row<Output_>(*p);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column NaN counts.
 *
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param nopt Counting options.
 *
 * @return A vector of length equal to the number of columns, containing the number of NaNs in each column.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat, const Options& nopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.ncol());
    apply(false, mat, output.data(), nopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility only.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p, const Options& nopt) {
    return by_column<Output_>(*p, nopt);
}
/**
 * @endcond
 */

/**
 * Overload with default options.
 *
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 *
 * @return A vector of length equal to the number of columns, containing the number of NaNs in each column.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat) {
    return by_column<Output_>(mat, Options());
}

/**
 * @cond
 */
// Back-compatibility only.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p) {
    return by_column<Output_>(*p);
}
/**
 * @endcond
 */

}

/**
 * @brief Functions for counting zeros on each dimension.
 * @namespace tatami_stats::counts::zero
 */
namespace zero {

/**
 * @brief Zero-counting options.
 */
struct Options {
    /**
     * Number of threads to use when obtaining counts across a `tatami::Matrix`.
     * See `tatami::parallelize()` for more details on the parallelization mechanism.
     */
    int num_threads = 1;
};

/**
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 *
 * @param row Whether to obtain a count for each row.
 * @param mat Instance of a `tatami::Matrix`.
 * @param[out] output Pointer to an array of length equal to the number of rows.
 * On output, this will store the number of zeros in each row.
 * @param zopt Counting options.
 */
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>& mat, Output_* output, const Options& zopt) {
    counts::apply(row, mat, output, zopt.num_threads, [](Value_ x) -> bool { return x == 0; });
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Value_, typename Index_, typename Output_>
void apply(bool row, const tatami::Matrix<Value_, Index_>* p, Output_* output, const Options& zopt) {
    apply(row, *p, output, zopt);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for row-wise zero counts.
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param zopt Counting options.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat, const Options& zopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.nrow());
    apply(true, mat, output.data(), zopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p, const Options& zopt) {
    return by_row<Output_>(*p, zopt);
}
/**
 * @endcond
 */

/**
 * Overload with default options. 
 *
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 *
 * @param mat Instance of a `tatami::Matrix`.
 *
 * @return A vector of length equal to the number of rows, containing the number of zeros in each row.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>& mat) {
    return by_row<Output_>(mat, Options());
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_row(const tatami::Matrix<Value_, Index_>* p) {
    return by_row<Output_>(*p);
}
/**
 * @endcond
 */

/**
 * Wrapper around `apply()` for column-wise zero counts.
 *
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param mat Instance of a `tatami::Matrix`.
 * @param zopt Counting options.
 *
 * @return A vector of length equal to the number of columns, containing the number of zeros in each column.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat, const Options& zopt) {
    auto output = tatami::create_container_of_Index_size<std::vector<Output_> >(mat.ncol());
    apply(false, mat, output.data(), zopt);
    return output;
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p, const Options& zopt) {
    return by_column<Output_>(*p, zopt);
}
/**
 * @endcond
 */

/**
 * @tparam Output_ Numeric type of the output count.
 * To avoid overflow, we recommend using a type that is large enough to hold the dimension extents of `mat`. 
 * @tparam Value_ Numeric type of the matrix value.
 * @tparam Index_ Integer type of the row/column indices.
 * This should be at least large enough to hold the dimensions of `p`.
 *
 * @param mat Instance of a `tatami::Matrix`.
 *
 * @return A vector of length equal to the number of columns, containing the number of zeros in each column.
 */
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>& mat) {
    return by_column<Output_>(mat, Options());
}

/**
 * @cond
 */
// Back-compatibility.
template<typename Output_ = int, typename Value_, typename Index_>
std::vector<Output_> by_column(const tatami::Matrix<Value_, Index_>* p) {
    return by_column<Output_>(*p);
}
/**
 * @endcond
 */

}

}

}

#endif
