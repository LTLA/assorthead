#ifndef TATAMI_MTX_WRITE_MATRIX_HPP
#define TATAMI_MTX_WRITE_MATRIX_HPP

#include "byteme/byteme.hpp"
#include "tatami/tatami.hpp"
#include "sanisizer/sanisizer.hpp"

#include <vector>
#include <cstddef>
#include <charconv>
#include <stdexcept>
#include <type_traits>
#include <cassert>
#include <memory>

#include "utils.hpp"

/**
 * @file write_matrix.hpp
 * @brief Write a **tatami** matrix to a Matrix Market file.
 */

namespace tatami_mtx {

/**
 * @cond
 */
template<typename Value_>
std::size_t convert(Value_ val, std::vector<char>& buffer, const std::optional<std::chars_format>& format, const std::optional<int>& precision) {
    std::size_t store;

    while (1) {
        const auto bufptr = buffer.data();
        const auto bufsize = buffer.size();

        const auto out = [&]() {
            if constexpr(std::is_integral<Value_>::value) {
                return std::to_chars(bufptr, bufptr + bufsize, val); 
            } else if (!format.has_value()) {
                return std::to_chars(bufptr, bufptr + bufsize, val); 
            } else if (!precision.has_value()) {
                return std::to_chars(bufptr, bufptr + bufsize, val, *format); 
            } else {
                return std::to_chars(bufptr, bufptr + bufsize, val, *format, *precision); 
            }
        }();

        if (out.ec != std::errc::value_too_large) {
            store = out.ptr - bufptr;
            break;
        }

        // Highly unlikely we get to this point, but if we do, we need to safely resize the buffer.
        constexpr auto maxlimit = std::numeric_limits<I<decltype(bufsize)> >::max();
        if (sanisizer::is_equal(maxlimit, bufsize)) {
            throw std::runtime_error("cannot allocate more buffer space for numeric conversion");
        }

        // Buffer should have non-zero size before we get to this point!
        assert(bufsize > 0);
        const auto newlimit = (sanisizer::is_less_than(maxlimit / 2, bufsize) ? maxlimit : bufsize * 2);
        buffer.resize(newlimit);
    }

    return store;
}
/**
 * @endcond
 */

/**
 * @brief Options for `write_matrix()` and friends.
 */
struct WriteMatrixOptions {
    /**
     * Whether to save the matrix in the Matrix Market coordinate (sparse) format.
     * If `false`, the matrix is saved in the array (dense) format instead.
     * If unset, defaults to `true` if the matrix is sparse.
     */
    std::optional<bool> coordinate;

    /**
     * Whether to save coordinate matrices in row-major order, i.e., coordinates are sorted by row first and then by column.
     * If `false`, the matrix is saved in column-major order, i.e., column first and then row.
     * If unset, defaults to `true` if the matrix prefers row access.
     */
    std::optional<bool> by_row;

    /**
     * Whether to write the Matrix Market banner.
     * This can be disabled to, e.g., write a custom banner.
     */
    bool banner = true;

    /**
     * String format for floating-point values.
     * If unset, the default behavior of `std::to_chars()` is used.
     */
    std::optional<std::chars_format> format;

    /**
     * Precision of floating-point values when converted to a string.
     * If unset, the default behavior of `std::to_chars()` is used. 
     */
    std::optional<int> precision;

    /**
     * Size of the buffer in which to cache written values before flushing them to the `byteme::Writer` instance.
     * Larger values improve efficiency at the cost of increasing memory usage.
     */
    std::size_t buffer_size = sanisizer::cap<std::size_t>(65536);

    /**
     * Number of threads for counting the number of structural non-zeros in a coordinate matrix.
     * This should be positive.
     * Note that this does not affect the writing itself, which is still done in serial.
     */ 
    int num_threads = 1;
};

/**
 * Write a `tatami::Matrix` to a Matrix Market file.
 * This can either be stored in an array or coordinate format depending on the options.
 *
 * @tparam Value_ Numeric type of the matrix data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param matrix Input matrix.
 * @param writer A `byteme::Writer` instance representing the file to which the matrix contents will be written.
 * It is the caller's responsibility to call `byteme::Writer::finish()` once this function returns.
 * @param options Options for writing the matrix.
 */
template<typename Value_, typename Index_>
void write_matrix(const tatami::Matrix<Value_, Index_>& matrix, byteme::Writer& writer, const WriteMatrixOptions& options) {
    std::unique_ptr<byteme::BufferedWriter<char> > bufwriter;
    if (options.num_threads > 1) {
        bufwriter.reset(new byteme::ParallelBufferedWriter<char, byteme::Writer*>(&writer, options.buffer_size));
    } else {
        bufwriter.reset(new byteme::SerialBufferedWriter<char, byteme::Writer*>(&writer, options.buffer_size));
    }

    const bool coordinate = (options.coordinate.has_value() ? *(options.coordinate) : matrix.is_sparse());
    if (options.banner) {
        bufwriter->write("%%MatrixMarket matrix");
        if (coordinate) {
            bufwriter->write(" coordinate");
        } else {
            bufwriter->write(" array");
        }
        if constexpr(std::is_integral<Value_>::value) {
            bufwriter->write(" integer");
        } else {
            bufwriter->write(" real");
        }
        bufwriter->write(" general\n");
    }

    const auto NR = matrix.nrow();
    const auto NC = matrix.ncol();
    std::vector<char> conversion_buffer(100);

    if (!coordinate){ 
        const auto NR_size = convert(NR, conversion_buffer, options.format, options.precision);
        bufwriter->write(conversion_buffer.data(), NR_size);
        bufwriter->write('\t');
        const auto NC_size = convert(NC, conversion_buffer, options.format, options.precision);
        bufwriter->write(conversion_buffer.data(), NC_size);
        bufwriter->write('\n');

        auto ext = tatami::consecutive_extractor<false>(matrix, false, static_cast<Index_>(0), NC);
        auto vbuffer = sanisizer::create<std::vector<Value_> >(NR);

        for (I<decltype(NC)> c = 0; c < NC; ++c) {
            auto ptr = ext->fetch(vbuffer.data());
            for (I<decltype(NR)> r = 0; r < NR; ++r) {
                const auto used = convert(ptr[r], conversion_buffer, options.format, options.precision); 
                bufwriter->write(conversion_buffer.data(), used);
                bufwriter->write('\n');
            }
        }
        return;
    }

    // Building a look-up table so we don't have to do repeated string conversions.
    const auto nums = std::max(NR, NC);
    std::vector<std::size_t> lookup(sanisizer::sum<typename std::vector<std::size_t>::size_type>(nums, 1));
    std::vector<char> dictionary;
    for (Index_ i = 0; i < nums; ++i) {
        const auto used = convert(i + 1, conversion_buffer, options.format, options.precision);
        dictionary.insert(dictionary.end(), conversion_buffer.data(), conversion_buffer.data() + used);
        lookup[i + 1] = dictionary.size();
    }

    if (NR >= 1) {
        bufwriter->write(dictionary.data() + lookup[NR - 1], lookup[NR] - lookup[NR - 1]);
    } else {
        bufwriter->write('0');
    }
    bufwriter->write('\t');

    if (NC >= 1) {
        bufwriter->write(dictionary.data() + lookup[NC - 1], lookup[NC] - lookup[NC - 1]);
    } else {
        bufwriter->write('0');
    }
    bufwriter->write('\t');

    if (NR == 0 || NC == 0) {
        bufwriter->write("0\n");
        return;
    }

    const bool by_row = (options.by_row.has_value() ? *(options.by_row) : matrix.prefer_rows());

    if (matrix.is_sparse()) {
        assert(options.num_threads >= 0);
        auto totals = sanisizer::create<std::vector<unsigned long long> >(options.num_threads);

        // Trying to figure out how many non-zero elements we have before starting.
        if (matrix.prefer_rows()) {
            tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
                tatami::Options opt;
                opt.sparse_extract_index = false;
                opt.sparse_extract_value = false;
                opt.sparse_ordered_index = false;
                auto ext = tatami::consecutive_extractor<true>(matrix, true, start, length, opt);
                unsigned long long count = 0;
                for (Index_ r = start, end = start + length; r < end; ++r) {
                    auto range = ext->fetch(NULL, NULL);
                    count = sanisizer::sum<I<decltype(count)> >(count, range.number);
                }
                totals[t] = count;
            }, NR, options.num_threads);

        } else {
            tatami::parallelize([&](int t, Index_ start, Index_ length) -> void {
                tatami::Options opt;
                opt.sparse_extract_index = false;
                opt.sparse_extract_value = false;
                opt.sparse_ordered_index = false;
                auto ext = tatami::consecutive_extractor<true>(matrix, false, start, length, opt);
                unsigned long long count = 0;
                for (Index_ c = start, end = start + length; c < end; ++c) {
                    auto range = ext->fetch(NULL, NULL);
                    count = sanisizer::sum<I<decltype(count)> >(count, range.number);
                }
                totals[t] = count;
            }, NC, options.num_threads);
        }

        unsigned long long total_size = 0;
        for (auto t : totals) {
            total_size = sanisizer::sum<I<decltype(total_size)> >(total_size, t);
        }
        const auto used = convert(total_size, conversion_buffer, options.format, options.precision);
        bufwriter->write(conversion_buffer.data(), used);
        bufwriter->write('\n');

        if (by_row) {
            auto ext = tatami::consecutive_extractor<true>(matrix, true, static_cast<Index_>(0), NR);
            auto vbuffer = sanisizer::create<std::vector<Value_> >(NC);
            auto ibuffer = sanisizer::create<std::vector<Index_> >(NC);
            for (I<decltype(NR)> r = 0; r < NR; ++r) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(range.number)> i = 0; i < range.number; ++i) {
                    bufwriter->write(dictionary.data() + lookup[r], lookup[r + 1] - lookup[r]);
                    bufwriter->write('\t');
                    const auto index = range.index[i];
                    bufwriter->write(dictionary.data() + lookup[index], lookup[index + 1] - lookup[index]);
                    bufwriter->write('\t');
                    auto used = convert(range.value[i], conversion_buffer, options.format, options.precision); 
                    bufwriter->write(conversion_buffer.data(), used);
                    bufwriter->write('\n');
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<true>(matrix, false, static_cast<Index_>(0), NC);
            auto vbuffer = sanisizer::create<std::vector<Value_> >(NR);
            auto ibuffer = sanisizer::create<std::vector<Index_> >(NR);
            for (I<decltype(NC)> c = 0; c < NC; ++c) {
                auto range = ext->fetch(vbuffer.data(), ibuffer.data());
                for (I<decltype(range.number)> i = 0; i < range.number; ++i) {
                    const auto index = range.index[i];
                    bufwriter->write(dictionary.data() + lookup[index], lookup[index + 1] - lookup[index]);
                    bufwriter->write('\t');
                    bufwriter->write(dictionary.data() + lookup[c], lookup[c + 1] - lookup[c]);
                    bufwriter->write('\t');
                    auto used = convert(range.value[i], conversion_buffer, options.format, options.precision); 
                    bufwriter->write(conversion_buffer.data(), used);
                    bufwriter->write('\n');
                }
            }
        }

    } else {
        const auto total_size = sanisizer::product<unsigned long long>(NR, NC);
        const auto total_used = convert(total_size, conversion_buffer, options.format, options.precision);
        bufwriter->write(conversion_buffer.data(), total_used);
        bufwriter->write('\n');

        if (by_row) {
            auto ext = tatami::consecutive_extractor<false>(matrix, true, static_cast<Index_>(0), NR);
            auto vbuffer = sanisizer::create<std::vector<Value_> >(NC);
            for (I<decltype(NR)> r = 0; r < NR; ++r) {
                auto ptr = ext->fetch(vbuffer.data());
                for (I<decltype(NC)> c = 0; c < NC; ++c) {
                    bufwriter->write(dictionary.data() + lookup[r], lookup[r + 1] - lookup[r]);
                    bufwriter->write('\t');
                    bufwriter->write(dictionary.data() + lookup[c], lookup[c + 1] - lookup[c]);
                    bufwriter->write('\t');
                    const auto used = convert(ptr[c], conversion_buffer, options.format, options.precision); 
                    bufwriter->write(conversion_buffer.data(), used);
                    bufwriter->write('\n');
                }
            }

        } else {
            auto ext = tatami::consecutive_extractor<false>(matrix, false, static_cast<Index_>(0), NC);
            auto vbuffer = sanisizer::create<std::vector<Value_> >(NR);
            for (I<decltype(NC)> c = 0; c < NC; ++c) {
                auto ptr = ext->fetch(vbuffer.data());
                for (I<decltype(NR)> r = 0; r < NR; ++r) {
                    bufwriter->write(dictionary.data() + lookup[r], lookup[r + 1] - lookup[r]);
                    bufwriter->write('\t');
                    bufwriter->write(dictionary.data() + lookup[c], lookup[c + 1] - lookup[c]);
                    bufwriter->write('\t');
                    const auto used = convert(ptr[r], conversion_buffer, options.format, options.precision); 
                    bufwriter->write(conversion_buffer.data(), used);
                    bufwriter->write('\n');
                }
            }
        }
    }
}

/**
 * Write a `tatami::Matrix` from a Matrix Market text file, see `write_matrix()` for details.
 *
 * @tparam Value_ Numeric type of the matrix data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param matrix Input matrix.
 * @param filepath Path to the Matrix Market file to be written.
 * @param options Options for writing the matrix.
 */
template<typename Value_, typename Index_>
void write_matrix_to_text_file(const tatami::Matrix<Value_, Index_>& matrix, const char* filepath, const WriteMatrixOptions& options) {
    byteme::RawFileWriter writer(filepath, {});
    write_matrix<Value_, Index_>(matrix, writer, options);
    writer.finish();
}

#if __has_include("zlib.h")

/**
 * Write a `tatami::Matrix` to a Gzip-compressed Matrix Market text file, see `write_matrix()` for details.
 *
 * @tparam Value_ Numeric type of the matrix data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param matrix Input matrix.
 * @param filepath Path to the (Gzip-compressed) Matrix Market file to be written.
 * @param options Options for writing the matrix.
 */
template<typename Value_, typename Index_, typename StoredValue_ = Automatic, typename StoredIndex_ = Automatic>
void write_matrix_to_gzip_file(const tatami::Matrix<Value_, Index_>& matrix, const char* filepath, const WriteMatrixOptions& options) {
    byteme::GzipFileWriter writer(filepath, {});
    write_matrix<Value_, Index_>(matrix, writer, options);
    writer.finish();
}

#endif

/**
 * Write a `tatami::Matrix` to a buffer containing a Matrix Market text file, see `write_matrix()` for details.
 *
 * @tparam Value_ Numeric type of the matrix data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param matrix Input matrix.
 * @param options Options for writing the matrix.
 *
 * @return Vector with the buffer contents.
 */
template<typename Value_, typename Index_>
std::vector<unsigned char> write_matrix_to_text_buffer(const tatami::Matrix<Value_, Index_>& matrix, const WriteMatrixOptions& options) {
    byteme::RawBufferWriter writer({});
    write_matrix<Value_, Index_>(matrix, writer, options);
    writer.finish();
    return writer.get_output();
}

#if __has_include("zlib.h")

/**
 * Write a `tatami::Matrix` to a buffer containing a Gzip/Zlib-compressed Matrix Market file, see `write_matrix()` for details.
 *
 * @tparam Value_ Numeric type of the matrix data.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * @param matrix Input matrix.
 * @param mode Compression mode for the Zlib library.
 * This should be one of `DEFLATE`, `ZLIB` or `GZIP`.
 * @param options Options for writing the matrix.
 *
 * @return Vector with the buffer contents.
 */
template<typename Value_, typename Index_>
std::vector<unsigned char> write_matrix_to_zlib_buffer(const tatami::Matrix<Value_, Index_>& matrix, const byteme::ZlibCompressionMode mode, const WriteMatrixOptions& options) {
    byteme::ZlibBufferWriter writer([&]{
        byteme::ZlibBufferWriterOptions opt;
        opt.mode = mode;
        return opt;
    }());
    write_matrix<Value_, Index_>(matrix, writer, options);
    writer.finish();
    return writer.get_output();
}

#endif

}

#endif
