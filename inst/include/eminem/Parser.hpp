#ifndef EMINEM_PARSER_HPP
#define EMINEM_PARSER_HPP

#include <vector>
#include <string>
#include <complex>
#include <type_traits>
#include <stdexcept>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <limits>

#include "byteme/RawBufferReader.hpp"
#include "byteme/PerByte.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

/**
 * @file Parser.hpp
 *
 * @brief Parse a matrix from a Matrix Market file.
 */

namespace eminem {

/**
 * @brief Options for the `Parser` constructor.
 */
struct ParserOptions {
    /**
     * Number of threads to use for parsing.
     */
    int num_threads = 1;

    /**
     * Approximate size of the block (in bytes) to be processed by each thread.
     * This is rounded up to the nearest newline before parallel processing.
     * Only relevant when `num_threads > 1`.
     */
    std::size_t block_size = sanisizer::cap<std::size_t>(65536);
};

/**
 * @cond
 */
template<typename Workspace_>
class ThreadPool {
public:
    template<typename RunJob_>
    ThreadPool(RunJob_ run_job, int num_threads) :
        my_helpers(sanisizer::cast<decltype(my_helpers.size())>(num_threads)) 
    {
        std::mutex init_mut;
        std::condition_variable init_cv;
        int num_initialized = 0;

        my_threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            // Copy lambda as it will be gone once this constructor finishes.
            my_threads.emplace_back([run_job,this,&init_mut,&init_cv,&num_initialized](int thread) -> void { 
                Helper env; // allocating this locally within each thread to reduce the risk of false sharing.
                my_helpers[thread] = &env;
                {
                    std::lock_guard lck(init_mut);
                    ++num_initialized;
                    init_cv.notify_one();
                }

                while (1) {
                    std::unique_lock lck(env.mut);
                    env.cv.wait(lck, [&]() -> bool { return env.input_ready; });
                    if (env.terminated) {
                        return;
                    }
                    env.input_ready = false;

                    try {
                        run_job(env.work);
                    } catch (...) {
                        std::lock_guard elck(my_error_mut);
                        if (!my_error) {
                            my_error = std::current_exception();
                        }
                    }

                    env.has_output = true;
                    env.available = true;
                    env.cv.notify_one();
                }
            }, t);
        }

        // Only returning once all threads (and their specific mutexes) are initialized.
        {
            std::unique_lock ilck(init_mut);
            init_cv.wait(ilck, [&]() -> bool { return num_initialized == num_threads; });
        }
    }

    ~ThreadPool() {
        for (auto envptr : my_helpers) {
            auto& env = *envptr;
            {
                std::lock_guard lck(env.mut);
                env.terminated = true;
                env.input_ready = true;
            }
            env.cv.notify_one();
        }
        for (auto& thread : my_threads) {
            thread.join();
        }
    }

private:
    std::vector<std::thread> my_threads;

    struct Helper {
        std::mutex mut;
        std::condition_variable cv;
        bool input_ready = false;
        bool available = true;
        bool has_output = false;
        bool terminated = false;
        Workspace_ work;
    };
    std::vector<Helper*> my_helpers;

    std::mutex my_error_mut;
    std::exception_ptr my_error;

public:
    template<typename CreateJob_, typename MergeJob_>
    bool run(CreateJob_ create_job, MergeJob_ merge_job) {
        auto num_threads = my_threads.size();
        bool finished = false;
        decltype(num_threads) thread = 0, finished_count = 0;

        // We submit jobs by cycling through all threads, then we merge their results in order of submission.
        // This is a less efficient worksharing scheme but it guarantees the same order of merges.
        while (1) {
            auto& env = *(my_helpers[thread]);
            std::unique_lock lck(env.mut);
            env.cv.wait(lck, [&]() -> bool { return env.available; });

            {
                std::lock_guard elck(my_error_mut);
                if (my_error) {
                    std::rethrow_exception(my_error);
                }
            }
            env.available = false;

            if (env.has_output) {
                // If the user requests an early quit from the merge job,
                // there's no point processing the later merge jobs from 
                // other threads, so we just break out at this point.
                if (!merge_job(env.work)) {
                    return false;
                }
                env.has_output = false;
            }

            if (finished) {
                // Go through all threads one last time, making sure all results are merged.
                ++finished_count;
                if (finished_count == num_threads) {
                    break;
                }
            } else {
                finished = !create_job(env.work);
                env.input_ready = true;
                lck.unlock();
                env.cv.notify_one();
            }

            ++thread;
            if (thread == num_threads) {
                thread = 0;
            }
        }

        return true;
    }
};

template<class Input_>
bool fill_to_next_newline(Input_& input, std::vector<char>& buffer, std::size_t block_size) {
    buffer.resize(block_size);
    auto done = input.extract(block_size, buffer.data());
    buffer.resize(done.first);
    if (!done.second || buffer.empty()) {
        return false;
    }
    char last = buffer.back();
    while (last != '\n') {
        last = input.get();
        buffer.push_back(last);
        if (!input.advance()) {
            return false;
        }
    }
    return true;
}

inline std::size_t count_newlines(const std::vector<char>& buffer) {
    std::size_t n = 0;
    for (auto x : buffer) {
        n += (x == '\n');
    }
    return n;
}

typedef unsigned long long Index; // for back-compatibility.
/**
 * @endcond
 */

/**
 * Integer type of the line count and line numbers in `Parser`.
 * We use an `unsigned long long` by default to guarantee at least 64 bits of storage.
 */
typedef unsigned long long LineIndex;

/**
 * @brief Parse a matrix from a Matrix Market file.
 *
 * @tparam Input_ Class of the source of input bytes.
 * This should be a pointer to an object satisfying the `byteme::PerByteInterface` instance.
 * @tparam Index_ Integer type of the row/column indices.
 *
 * This parses a Matrix Market file according to the specification described at https://math.nist.gov/MatrixMarket/reports/MMformat.ps.gz.
 * It is expected that users call `scan_preamble()` to determine the field type (see `eminem::Field` for supported values),
 * after which they may call one of `scan_integer()`, `scan_real()`, etc. to parse the rest of the file.
 * An error will be thrown upon detecting a non-compliant file.
 *
 * As the Matrix Market specification is somewhat vague in parts, we apply the following refinements:
 *
 * - We define a "blank" character as a horizontal tab, a space (ASCII 32) or a carriage return.
 *   A new line is only defined by the line feed character (ASCII 10).
 *   The carriage return is only considered as a blank for compatibility with systems that use CRLF to define new lines,
 *   and we do not check for whether the carriage return is followed by a line feed.
 * - The final line of the file may or may not be newline terminated.
 * - Integer values for the row/column indices, number of rows/columns and number of lines should be a sequence of one or more digits.
 *   Leading zeros are ignored and will not be interpreted as octal.
 *   An error is thrown if overflow of the `Index_` type occurs in `scan_preamble()`.
 *   An error is also thrown if the row/column indices in any data line exceed the number of rows/columns in the preamble,
 *   or if the observed number of data lines exceeds the expected number for coordinate matrices/vectors.
 * - Integer data values should be a sequence of one or more digits, optionally preceded by a `+` or `-` sign.
 *   Leading zeros are ignored and will not be interpreted as octal.
 *   An error is thrown if overflow of the `Type_` type occurs in `scan_integer()`.
 *
 * Real data values should follow one of the following formats:
 *
 * - `(sign)[digit sequence](exponent)`
 * - `(sign)[digit sequence].(exponent)`
 * - `(sign).[digit sequence](exponent)`
 * - `(sign)[digit sequence].[digit sequence](exponent)`
 *
 * The digit sequence should contain one or mor digits, possibly containing leading zeros that will be ignored.
 * The sign is optional and should be either `+` or `-`.
 * The exponent is optional should have the form `e(sign)[digit sequence]` or `E(sign)[digit sequence]`, where the sign is again optional.
 * We also support case-insensitive matches to `inf`, `infinity` or `nan`.
 * This will be converted to their corresponding IEEE special values if they are available for the provided `Type_` in `scan_real()`, otherwise an error is thrown.
 * For non-special values, no additional checks for overflow are applied. 
 * If IEEE arithmetic is available, overflow will manifest as infinities, otherwise they will be undefined behavior.
 * 
 * No validation is performed to determine whether coordinates are consistent with non-general symmetries.
 * Similarly, we do not check for the existence of multiple lines with the same row/column indices in coordinate matrices/vectors.
 */
template<class Input_, typename Index_ = unsigned long long>
class Parser {
public:
    /**
     * @param input Source of input bytes, typically a pointer to a `byteme::PerByteInterface` instance.
     * @param options Further options.
     */
    Parser(Input_ input, const ParserOptions& options) : 
        my_input(std::move(input)),
        my_nthreads(options.num_threads),
        my_block_size(options.block_size)
    {
        sanisizer::cast<typename std::vector<char>::size_type>(my_block_size); // checking that there won't be any overflow in fill_to_next_newline().
    }

private:
    Input_ my_input;
    int my_nthreads;
    std::size_t my_block_size;

    LineIndex my_current_line = 0;
    MatrixDetails my_details;

    template<typename Input2_>
    static bool chomp(Input2_& input) {
        while (1) {
            char x = input.get();
            if (x != ' ' && x != '\t' && x != '\r') {
                return true;
            }
            if (!(input.advance())) {
                break;
            }
        }
        return false;
    }

    template<typename Input2_>
    static bool advance_and_chomp(Input2_& input) {
        // When the input is currently on a whitespace, we advance first so we
        // avoid a redundant iteration where the comparison is always true.
        if (!(input.advance())) {
            return false;
        }
        return chomp(input);
    }

    template<typename Input2_>
    static bool skip_lines(Input2_& input, LineIndex& current_line) {
        // Skip comments and empty lines.
        while (1) {
            char x = input.get();
            if (x == '%') {
                do {
                    if (!(input.advance())) {
                        return false;
                    }
                } while (input.get() != '\n');
            } else if (x != '\n') {
                break;
            }

            if (!input.advance()) { // move past the newline.
                return false;
            }
            ++current_line;
        }
        return true;
    }

private:
    bool my_passed_banner = false;

    struct ExpectedMatch {
        ExpectedMatch(bool found, bool newline, bool remaining) : found(found), newline(newline), remaining(remaining) {}
        ExpectedMatch() : ExpectedMatch(false, false, false) {}
        bool found;
        bool newline;
        bool remaining;
    };

    ExpectedMatch advance_past_expected_string() {
        if (!(my_input->advance())) { // move off the last character.
            return ExpectedMatch(true, false, false);
        }

        char next = my_input->get();
        if (next == ' ' || next == '\t' || next == '\r') {
            if (!advance_and_chomp(*my_input)) { // gobble up all of the remaining horizontal space.
                return ExpectedMatch(true, false, false);
            }
            if (my_input->get() == '\n') {
                bool remaining = my_input->advance(); // move past the newline for consistency with other functions.
                return ExpectedMatch(true, true, remaining); // move past the newline for consistency with other functions.
            }
            return ExpectedMatch(true, false, true);

        } else if (next == '\n') {
            bool remaining = my_input->advance(); // move past the newline for consistency with other functions.
            return ExpectedMatch(true, true, remaining);
        }

        // If the next character is not a space or whitespace, it's not a match.
        return ExpectedMatch(false, true, true);
    }

    ExpectedMatch is_expected_string(const char* ptr, std::size_t len, std::size_t start) {
        // It is assumed that the first 'start' characters of 'ptr' where
        // already checked and matched before entering this function, and that
        // 'my_input' is currently positioned at the start-th character, i.e.,
        // 'ptr[start-1]' (and thus requires an advance() call before we can
        // compare against 'ptr[start]').
        for (std::size_t i = start; i < len; ++i) {
            if (!my_input->advance()) {
                return ExpectedMatch(false, false, false);
            }
            if (my_input->get() != ptr[i]) {
                return ExpectedMatch(false, false, true);
            }
        }
        return advance_past_expected_string();
    }

    ExpectedMatch is_expected_string(const char* ptr, std::size_t len) {
        // Using a default start of 1, assuming that we've already compared
        // the first character before entering this function.
        return is_expected_string(ptr, len, 1);
    }

    bool parse_banner_object() {
        ExpectedMatch res;

        char x = my_input->get();
        if (x == 'm') {
            res = is_expected_string("matrix", 6);
            my_details.object = Object::MATRIX;
        } else if (x == 'v') {
            res = is_expected_string("vector", 6);
            my_details.object = Object::VECTOR;
        }

        if (!res.found) {
            throw std::runtime_error("first banner field should be one of 'matrix' or 'vector'");
        }
        if (!res.remaining) {
            throw std::runtime_error("end of file reached after the first banner field");
        }

        return res.newline;
    }

    bool parse_banner_format() {
        ExpectedMatch res;

        char x = my_input->get();
        if (x == 'c') {
            res = is_expected_string("coordinate", 10);
            my_details.format = Format::COORDINATE;
        } else if (x == 'a') {
            res = is_expected_string("array", 5);
            my_details.format = Format::ARRAY;
        }

        if (!res.found) {
            throw std::runtime_error("second banner field should be one of 'coordinate' or 'array'");
        }
        if (!res.remaining) {
            throw std::runtime_error("end of file reached after the second banner field");
        }

        return res.newline;
    }

    bool parse_banner_field() {
        ExpectedMatch res;

        char x = my_input->get();
        if (x == 'i') { 
            res = is_expected_string("integer", 7);
            my_details.field = Field::INTEGER;
        } else if (x == 'd') { 
            res = is_expected_string("double", 6);
            my_details.field = Field::DOUBLE;
        } else if (x == 'c') {
            res = is_expected_string("complex", 7);
            my_details.field = Field::COMPLEX;
        } else if (x == 'p') {
            res = is_expected_string("pattern", 7);
            my_details.field = Field::PATTERN;
        } else if (x == 'r') {
            res = is_expected_string("real", 4);
            my_details.field = Field::REAL;
        }

        if (!res.found) {
            throw std::runtime_error("third banner field should be one of 'real', 'integer', 'double', 'complex' or 'pattern'");
        }
        if (!res.remaining) {
            throw std::runtime_error("end of file reached after the third banner field");
        }

        return res.newline;
    }

    bool parse_banner_symmetry() {
        ExpectedMatch res;

        char x = my_input->get();
        if (x == 'g') {
            res = is_expected_string("general", 7);
            my_details.symmetry = Symmetry::GENERAL;
        } else if (x == 'h') {
            res = is_expected_string("hermitian", 9);
            my_details.symmetry = Symmetry::HERMITIAN;
        } else if (x == 's') {
            if (my_input->advance()) {
                char x = my_input->get();
                if (x == 'k') {
                    res = is_expected_string("skew-symmetric", 14, 2);
                    my_details.symmetry = Symmetry::SKEW_SYMMETRIC;
                } else {
                    res = is_expected_string("symmetric", 9, 2);
                    my_details.symmetry = Symmetry::SYMMETRIC;
                }
            }
        }

        if (!res.found) {
            throw std::runtime_error("fourth banner field should be one of 'general', 'hermitian', 'skew-symmetric' or 'symmetric'");
        }
        if (!res.remaining) {
            throw std::runtime_error("end of file reached after the fourth banner field");
        }

        return res.newline;
    }

    void scan_banner() {
        if (my_passed_banner) {
            throw std::runtime_error("banner has already been scanned");
        }
        if (!(my_input->valid())) {
            throw std::runtime_error("failed to find banner line before end of file");
        }
        if (my_input->get() != '%') {
            throw std::runtime_error("first line of the file should be the banner");
        }
 
        auto found_banner = is_expected_string("%%MatrixMarket", 14);
        if (!found_banner.remaining) {
            throw std::runtime_error("end of file reached before matching the banner");
        }
        if (!found_banner.found) {
            throw std::runtime_error("first line of the file should be the banner");
        }
        if (found_banner.newline) {
            throw std::runtime_error("end of line reached before matching the banner");
        }

        if (parse_banner_object()) {
            throw std::runtime_error("end of line reached after the first banner field");
        }
        if (parse_banner_format()) {
            throw std::runtime_error("end of line reached after the second banner field");
        }

        bool eol = false;
        if (my_details.object == Object::MATRIX) {
            if (parse_banner_field()) {
                throw std::runtime_error("end of line reached after the third banner field");
            }
            eol = parse_banner_symmetry();
        } else {
            // The NIST spec doesn't say anything about symmetry for vector,
            // and it doesn't really make sense anyway. We'll just set it to
            // general and hope for the best.
            my_details.symmetry = Symmetry::GENERAL;

            // No need to throw on newline because this might be the last field AFAICT.
            eol = parse_banner_field();
        }

        my_passed_banner = true;

        // Ignoring all other fields until the newline. We can use a do/while
        // to skip the initial comparison because we know that the current
        // character cannot be a newline if eol = false.
        if (!eol) {
            do {
                if (!(my_input->advance())) {
                    throw std::runtime_error("end of file reached before the end of the banner line");
                }
            } while (my_input->get() != '\n'); 
            my_input->advance(); // move past the newline.
        }

        ++my_current_line;
        return;
    }

public:
    /**
     * Retrieve the Matrix Market banner, containing information about the data format and type.
     * This should only be called after `scan_preamble()`.
     *
     * @return Details about the matrix in this file.
     */
    const MatrixDetails& get_banner() const {
        if (!my_passed_banner) {
            throw std::runtime_error("banner has not yet been scanned");
        }
        return my_details;
    }

private:
    // Only calls with 'last_ = true' need to know if there are any remaining bytes after the newline.
    // This is because all non-last calls with no remaining bytes must have thrown.
    template<typename Integer_>
    struct NotLastSizeInfo {
        Integer_ index = 0;
    };

    template<typename Integer_>
    struct LastSizeInfo {
        Integer_ index = 0;
        bool remaining = false;
    };

    template<bool last_, typename Integer_>
    using SizeInfo = typename std::conditional<last_, LastSizeInfo<Integer_>, NotLastSizeInfo<Integer_> >::type;

    template<bool last_, typename Integer_, class Input2_>
    static SizeInfo<last_, Integer_> scan_integer_field(bool size, Input2_& input, LineIndex overall_line_count) {
        SizeInfo<last_, Integer_> output;
        bool found = false;

        auto what = [&]() -> std::string {
            if (size) {
                return "size";
            } else {
                return "index";
            }
        };

        constexpr Integer_ max_limit = std::numeric_limits<Integer_>::max();
        constexpr Integer_ max_limit_before_mult = max_limit / 10; 
        constexpr Integer_ max_limit_mod = max_limit % 10; 

        while (1) {
            char x = input.get();
            switch(x) {
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    { 
                        Integer_ delta = x - '0';
                        // Structuring the conditionals so that it's most likely to short-circuit after only testing the first one.
                        if (output.index >= max_limit_before_mult && !(output.index == max_limit_before_mult && delta <= max_limit_mod)) {
                            throw std::runtime_error("integer overflow in " + what() + " field on line " + std::to_string(overall_line_count + 1));
                        }
                        output.index *= 10;
                        output.index += delta;
                    }
                    found = true;
                    break;
                case '\n':
                    // This check only needs to be put here, as all blanks should be chomped before calling
                    // this function; so we must start on a non-blank character. This starting character is either:
                    // - a digit, in which case found = true and this check is unnecessary.
                    // - a non-newline non-digit, in case we throw.
                    // - a newline, in which case we arrive here.
                    if (!found) {
                        throw std::runtime_error("empty " + what() + " field on line " + std::to_string(overall_line_count + 1));
                    }
                    if constexpr(last_) {
                        output.remaining = input.advance(); // advance past the newline.
                        return output;
                    }
                    throw std::runtime_error("unexpected newline when parsing " + what() + " field on line " + std::to_string(overall_line_count + 1));
                case ' ': case '\t': case '\r':
                    if (!advance_and_chomp(input)) { // skipping the current and subsequent blanks.
                        if constexpr(last_) {
                            return output;
                        } else {
                            throw std::runtime_error("unexpected end of file when parsing " + what() + " field on line " + std::to_string(overall_line_count + 1));
                        }
                    }
                    if constexpr(last_) {
                        if (input.get() != '\n') {
                            throw std::runtime_error("expected newline after the last " + what() + " field on line " + std::to_string(overall_line_count + 1));
                        }
                        output.remaining = input.advance(); // advance past the newline.
                    }
                    return output;
                default:
                    throw std::runtime_error("unexpected character when parsing " + what() + " field on line " + std::to_string(overall_line_count + 1));
            }

            if (!(input.advance())) { // moving past the current digit.
                if constexpr(last_) {
                    break;
                } else {
                    throw std::runtime_error("unexpected end of file when parsing " + what() + " field on line " + std::to_string(overall_line_count + 1));
                }
            }
        }

        return output;
    }

    template<bool last_, class Input2_>
    static SizeInfo<last_, Index_> scan_size_field(Input2_& input, LineIndex overall_line_count) {
        return scan_integer_field<last_, Index_>(true, input, overall_line_count);
    }

    template<class Input2_>
    static SizeInfo<true, LineIndex> scan_line_count_field(Input2_& input, LineIndex overall_line_count) {
        return scan_integer_field<true, LineIndex>(true, input, overall_line_count);
    }

    template<bool last_, class Input2_>
    static SizeInfo<last_, Index_> scan_index_field(Input2_& input, LineIndex overall_line_count) {
        return scan_integer_field<last_, Index_>(false, input, overall_line_count);
    }

private:
    bool my_passed_size = false;
    Index_ my_nrows = 0, my_ncols = 0;
    LineIndex my_nlines = 0;

    void scan_size() {
        if (!(my_input->valid())) {
            throw std::runtime_error("failed to find size line before end of file");
        }

        // Handling stray comments, empty lines, and leading whitespace.
        if (!skip_lines(*my_input, my_current_line)) {
            throw std::runtime_error("failed to find size line before end of file");
        }
        if (!chomp(*my_input)) {
            throw std::runtime_error("expected at least one size field on line " + std::to_string(my_current_line + 1));
        }

        if (my_details.object == Object::MATRIX) {
            if (my_details.format == Format::COORDINATE) {
                auto first_field = scan_size_field<false>(*my_input, my_current_line);
                my_nrows = first_field.index;

                auto second_field = scan_size_field<false>(*my_input, my_current_line);
                my_ncols = second_field.index;

                auto third_field = scan_line_count_field(*my_input, my_current_line);
                my_nlines = third_field.index;

            } else { // i.e., my_details.format == Format::ARRAY
                auto first_field = scan_size_field<false>(*my_input, my_current_line);
                my_nrows = first_field.index;

                auto second_field = scan_size_field<true>(*my_input, my_current_line);
                my_ncols = second_field.index;
                my_nlines = sanisizer::product<LineIndex>(my_nrows, my_ncols);
            }

        } else {
            if (my_details.format == Format::COORDINATE) {
                auto first_field = scan_size_field<false>(*my_input, my_current_line);
                my_nrows = first_field.index;

                auto second_field = scan_line_count_field(*my_input, my_current_line);
                my_nlines = second_field.index;

            } else { // i.e., my_details.format == Format::ARRAY
                auto first_field = scan_size_field<true>(*my_input, my_current_line);
                my_nlines = first_field.index;
                my_nrows = my_nlines;
            }
            my_ncols = 1;
        }

        ++my_current_line;
        my_passed_size = true;
    }

public:
    /**
     * Get the number of rows in the matrix.
     * This should only be called after `scan_preamble()`.
     * If the object type is `Object::VECTOR`, the number of rows is equal to the length of the vector.
     *
     * @return Number of rows.
     */
    Index_ get_nrows() const {
        if (!my_passed_size) {
            throw std::runtime_error("size line has not yet been scanned");
        }
        return my_nrows;
    }

    /**
     * Get the number of columns in the matrix.
     * This should only be called after `scan_preamble()`.
     * If the object type is `Object::VECTOR`, the number of columns is set to 1.
     *
     * @return Number of columns.
     */
    Index_ get_ncols() const {
        if (!my_passed_size) {
            throw std::runtime_error("size line has not yet been scanned");
        }
        return my_ncols;
    }

    /**
     * Get the number of non-zero lines in the coordinate format.
     * This should only be called after `scan_preamble()`.
     * If the object type is `Object::ARRAY`, the number of lines is defined as the product of the number of rows and columns.
     *
     * @return Number of non-zero lines. 
     */
    LineIndex get_nlines() const {
        if (!my_passed_size) {
            throw std::runtime_error("size line has not yet been scanned");
        }
        return my_nlines;
    }

public:
    /**
     * Scan the preamble from the Matrix Market file, including the banner and the size line.
     * This should only be called once.
     */
    void scan_preamble() {
        scan_banner();
        scan_size();
        return;
    }

private:
    template<typename Type_>
    struct ParseInfo {
        ParseInfo() = default;
        ParseInfo(Type_ value, bool remaining) : value(value), remaining(remaining) {}
        Type_ value;
        bool remaining;
    };

    template<typename Workspace_>
    bool configure_parallel_workspace(Workspace_& work) {
        bool available = fill_to_next_newline(*my_input, work.buffer, my_block_size);
        work.contents.clear();
        work.overall_line = my_current_line;
        my_current_line += count_newlines(work.buffer);
        return available;
    }

    void check_num_lines_loop(LineIndex data_line_count) const {
        if (data_line_count >= my_nlines) {
            throw std::runtime_error("more lines present than specified in the header (" + std::to_string(data_line_count) + " versus " + std::to_string(my_nlines) + ")");
        }
    }

    void check_num_lines_final(bool finished, LineIndex data_line_count) const {
        if (finished) {
            if (data_line_count != my_nlines) {
                // Must be fewer, otherwise we would have triggered the error in check_num_lines_loop() during iteration.
                throw std::runtime_error("fewer lines present than specified in the header (" + std::to_string(data_line_count) + " versus " + std::to_string(my_nlines) + ")");
            }
        }
    }

private:
    void check_matrix_coordinate_line(Index_ currow, Index_ curcol, LineIndex overall_line_count) const {
        if (!currow) {
            throw std::runtime_error("row index must be positive on line " + std::to_string(overall_line_count + 1));
        }
        if (currow > my_nrows) {
            throw std::runtime_error("row index out of range on line " + std::to_string(overall_line_count + 1));
        }
        if (!curcol) {
            throw std::runtime_error("column index must be positive on line " + std::to_string(overall_line_count + 1));
        }
        if (curcol > my_ncols) {
            throw std::runtime_error("column index out of range on line " + std::to_string(overall_line_count + 1));
        }
    }

    template<typename Type_, class Input2_, typename FieldParser_, class WrappedStore_>
    bool scan_matrix_coordinate_non_pattern_base(Input2_& input, LineIndex& overall_line_count, FieldParser_& fparser, WrappedStore_ wstore) const {
        bool valid = input.valid();
        while (valid) {
            // Handling stray comments, empty lines, and leading spaces.
            if (!skip_lines(input, overall_line_count)) {
                break;
            }
            if (!chomp(input)) {
                throw std::runtime_error("expected at least three fields for a coordinate matrix on line " + std::to_string(overall_line_count + 1));
            }

            auto first_field = scan_index_field<false>(input, overall_line_count);
            auto second_field = scan_index_field<false>(input, overall_line_count);
            check_matrix_coordinate_line(first_field.index, second_field.index, overall_line_count);

            // 'fparser' should leave 'input' at the start of the next line, if any exists.
            ParseInfo<Type_> res = fparser(input, overall_line_count);
            if (!wstore(first_field.index, second_field.index, res.value)) {
                return false;
            }
            ++overall_line_count;
            valid = res.remaining;
        }

        return true;
    }

    template<typename Type_, class FieldParser_, class Store_>
    bool scan_matrix_coordinate_non_pattern(Store_ store) {
        bool finished = false;
        LineIndex current_data_line = 0;

        if (my_nthreads == 1) {
            FieldParser_ fparser;
            finished = scan_matrix_coordinate_non_pattern_base<Type_>(
                *my_input,
                my_current_line,
                fparser,
                [&](Index_ r, Index_ c, Type_ value) -> bool {
                    check_num_lines_loop(current_data_line);
                    ++current_data_line;
                    return store(r, c, value);
                }
            );

        } else {
            struct Workspace {
                std::vector<char> buffer;
                FieldParser_ fparser;
                std::vector<std::tuple<Index_, Index_, Type_> > contents;
                LineIndex overall_line;
            };

            ThreadPool<Workspace> tp(
                [&](Workspace& work) -> bool {
                    byteme::RawBufferReader reader(reinterpret_cast<const unsigned char*>(work.buffer.data()), work.buffer.size());
                    byteme::PerByteSerial<char, byteme::RawBufferReader*> pb(&reader);
                    return scan_matrix_coordinate_non_pattern_base<Type_>(
                        pb,
                        work.overall_line,
                        work.fparser,
                        [&](Index_ r, Index_ c, Type_ value) -> bool {
                            work.contents.emplace_back(r, c, value);
                            return true; // threads cannot quit early in their parallel sections; this (and thus scan_*_base) must always return true.
                        }
                    );
                },
                my_nthreads
            );

            finished = tp.run(
                [&](Workspace& work) -> bool {
                    return configure_parallel_workspace(work);
                },
                [&](Workspace& work) -> bool {
                    for (const auto& con : work.contents) {
                        check_num_lines_loop(current_data_line); // defer check here for the correctly sync'd value of current_data_line.
                        if (!store(std::get<0>(con), std::get<1>(con), std::get<2>(con))) {
                            return false;
                        }
                        ++current_data_line;
                    }
                    return true;
                }
            );
        }

        check_num_lines_final(finished, current_data_line);
        return finished;
    }

private:
    template<class Input2_, class WrappedStore_>
    bool scan_matrix_coordinate_pattern_base(Input2_& input, LineIndex& overall_line_count, WrappedStore_ wstore) const {
        bool valid = input.valid();
        while (valid) {
            // Handling stray comments, empty lines, and leading spaces.
            if (!skip_lines(input, overall_line_count)) {
                break;
            }
            if (!chomp(input)) {
                throw std::runtime_error("expected two fields for a pattern matrix on line " + std::to_string(overall_line_count + 1));
            }

            auto first_field = scan_index_field<false>(input, overall_line_count);
            auto second_field = scan_index_field<true>(input, overall_line_count);
            check_matrix_coordinate_line(first_field.index, second_field.index, overall_line_count);

            if (!wstore(first_field.index, second_field.index)) {
                return false;
            }
            ++overall_line_count;
            valid = second_field.remaining;
        }

        return true;
    }

    template<class Store_>
    bool scan_matrix_coordinate_pattern(Store_ store) {
        bool finished = false;
        LineIndex current_data_line = 0;

        if (my_nthreads == 1) {
            finished = scan_matrix_coordinate_pattern_base(
                *my_input,
                my_current_line,
                [&](Index_ r, Index_ c) -> bool {
                    check_num_lines_loop(current_data_line);
                    ++current_data_line;
                    return store(r, c);
                }
            );

        } else {
            struct Workspace {
                std::vector<char> buffer;
                std::vector<std::tuple<Index_, Index_> > contents;
                LineIndex overall_line;
            };

            ThreadPool<Workspace> tp(
                [&](Workspace& work) -> bool {
                    byteme::RawBufferReader reader(reinterpret_cast<const unsigned char*>(work.buffer.data()), work.buffer.size());
                    byteme::PerByteSerial<char, byteme::RawBufferReader*> pb(&reader);
                    return scan_matrix_coordinate_pattern_base(
                        pb,
                        work.overall_line,
                        [&](Index_ r, Index_ c) -> bool {
                            work.contents.emplace_back(r, c);
                            return true; // threads cannot quit early in their parallel sections; this (and thus scan_*_base) must always return true.
                        }
                    );
                },
                my_nthreads
            );

            finished = tp.run(
                [&](Workspace& work) -> bool {
                    return configure_parallel_workspace(work);
                },
                [&](Workspace& work) -> bool {
                    for (const auto& con : work.contents) {
                        check_num_lines_loop(current_data_line);
                        if (!store(std::get<0>(con), std::get<1>(con))) {
                            return false;
                        }
                        ++current_data_line;
                    }
                    return true;
                }
            );
        }

        check_num_lines_final(finished, current_data_line);
        return finished;
    }

private:
    void check_vector_coordinate_line(Index_ currow, LineIndex overall_line_count) const {
        if (!currow) {
            throw std::runtime_error("row index must be positive on line " + std::to_string(overall_line_count + 1));
        }
        if (currow > my_nrows) {
            throw std::runtime_error("row index out of range on line " + std::to_string(overall_line_count + 1));
        }
    }

    template<typename Type_, class Input2_, class FieldParser_, class WrappedStore_>
    bool scan_vector_coordinate_non_pattern_base(Input2_& input, LineIndex& overall_line_count, FieldParser_& fparser, WrappedStore_ wstore) const {
        bool valid = input.valid();
        while (valid) {
            // handling stray comments, empty lines, and leading spaces.
            if (!skip_lines(input, overall_line_count)) {
                break;
            }
            if (!chomp(input)) {
                throw std::runtime_error("expected at least two fields for a coordinate vector on line " + std::to_string(overall_line_count + 1));
            }

            auto first_field = scan_index_field<false>(input, overall_line_count);
            check_vector_coordinate_line(first_field.index, overall_line_count);

            // 'fparser' should leave 'input' at the start of the next line, if any exists.
            ParseInfo<Type_> res = fparser(input, overall_line_count);
            if (!wstore(first_field.index, res.value)) {
                return false;
            }
            ++overall_line_count;
            valid = res.remaining;
        }

        return true;
    }

    template<typename Type_, class FieldParser_, class Store_>
    bool scan_vector_coordinate_non_pattern(Store_ store) {
        bool finished = false;
        LineIndex current_data_line = 0;

        if (my_nthreads == 1) {
            FieldParser_ fparser;
            finished = scan_vector_coordinate_non_pattern_base<Type_>(
                *my_input,
                my_current_line,
                fparser,
                [&](Index_ r, Type_ value) -> bool {
                    check_num_lines_loop(current_data_line);
                    ++current_data_line;
                    return store(r, 1, value);
                }
            );

        } else {
            struct Workspace {
                std::vector<char> buffer;
                FieldParser_ fparser;
                std::vector<std::tuple<Index_, Type_> > contents;
                LineIndex overall_line;
            };

            ThreadPool<Workspace> tp(
                [&](Workspace& work) -> bool {
                    byteme::RawBufferReader reader(reinterpret_cast<const unsigned char*>(work.buffer.data()), work.buffer.size());
                    byteme::PerByteSerial<char, byteme::RawBufferReader*> pb(&reader);
                    return scan_vector_coordinate_non_pattern_base<Type_>(
                        pb,
                        work.overall_line,
                        work.fparser,
                        [&](Index_ r, Type_ value) -> bool {
                            work.contents.emplace_back(r, value);
                            return true; // threads cannot quit early in their parallel sections; this (and thus scan_*_base) must always return true.
                        }
                    );
                },
                my_nthreads
            );

            finished = tp.run(
                [&](Workspace& work) -> bool {
                    return configure_parallel_workspace(work);
                },
                [&](Workspace& work) -> bool {
                    for (const auto& con : work.contents) {
                        check_num_lines_loop(current_data_line);
                        if (!store(std::get<0>(con), 1, std::get<1>(con))) {
                            return false;
                        }
                        ++current_data_line;
                    }
                    return true;
                }
            );
        }

        check_num_lines_final(finished, current_data_line);
        return finished;
    }

private:
    template<class Input2_, class WrappedStore_>
    bool scan_vector_coordinate_pattern_base(Input2_& input, LineIndex& overall_line_count, WrappedStore_ wstore) const {
        bool valid = input.valid();
        while (valid) {
            // Handling stray comments, empty lines, and leading spaces.
            if (!skip_lines(input, overall_line_count)) {
                break;
            }
            if (!chomp(input)) {
                throw std::runtime_error("expected one field for a coordinate vector on line " + std::to_string(overall_line_count + 1));
            }

            auto first_field = scan_index_field<true>(input, overall_line_count);
            check_vector_coordinate_line(first_field.index, overall_line_count);

            if (!wstore(first_field.index)) {
                return false;
            }
            ++overall_line_count;
            valid = first_field.remaining;
        }

        return true;
    }

    template<class Store_>
    bool scan_vector_coordinate_pattern(Store_ store) {
        bool finished = false;
        LineIndex current_data_line = 0;

        if (my_nthreads == 1) {
            finished = scan_vector_coordinate_pattern_base(
                *my_input,
                my_current_line,
                [&](Index_ r) -> bool {
                    check_num_lines_loop(current_data_line);
                    ++current_data_line;
                    return store(r, 1);
                }
            );

        } else {
            struct Workspace {
                std::vector<char> buffer;
                std::vector<Index_> contents;
                LineIndex overall_line;
            };

            ThreadPool<Workspace> tp(
                [&](Workspace& work) -> bool {
                    byteme::RawBufferReader reader(reinterpret_cast<const unsigned char*>(work.buffer.data()), work.buffer.size());
                    byteme::PerByteSerial<char, byteme::RawBufferReader*> pb(&reader);
                    return scan_vector_coordinate_pattern_base(
                        pb,
                        work.overall_line,
                        [&](Index_ r) -> bool {
                            work.contents.emplace_back(r);
                            return true; // threads cannot quit early in their parallel sections; this (and thus scan_*_base) must always return true.
                        }
                    );
                },
                my_nthreads
            );

            finished = tp.run(
                [&](Workspace& work) -> bool {
                    return configure_parallel_workspace(work);
                },
                [&](Workspace& work) -> bool {
                    for (const auto& r : work.contents) {
                        check_num_lines_loop(current_data_line);
                        if (!store(r, 1)) {
                            return false;
                        }
                        ++current_data_line;
                    }
                    return true;
                }
            );
        }

        check_num_lines_final(finished, current_data_line);
        return finished;
    }

private:
    template<typename Type_, class Input2_, class FieldParser_, class WrappedStore_>
    bool scan_matrix_array_base(Input2_& input, LineIndex& overall_line_count, FieldParser_& fparser, WrappedStore_ wstore) const {
        bool valid = input.valid();
        while (valid) {
            // Handling stray comments, empty lines, and leading spaces.
            if (!skip_lines(input, overall_line_count)) {
                break;
            }
            if (!chomp(input)) {
                throw std::runtime_error("expected at least one field for an array matrix on line " + std::to_string(overall_line_count + 1));
            }

            // 'fparser' should leave 'input' at the start of the next line, if any exists.
            ParseInfo<Type_> res = fparser(input, overall_line_count);
            if (!wstore(res.value)) {
                return false;
            }
            ++overall_line_count;
            valid = res.remaining;
        }

        return true;
    }

    template<typename Type_, class FieldParser_, class Store_>
    bool scan_matrix_array(Store_ store) {
        bool finished = false;
        LineIndex current_data_line = 0;

        Index_ currow = 1, curcol = 1;
        auto increment = [&]() {
            ++currow;
            if (currow > my_nrows) {
                ++curcol;
                currow = 1;
            }
        };

        if (my_nthreads == 1) {
            FieldParser_ fparser;
            finished = scan_matrix_array_base<Type_>(
                *my_input,
                my_current_line,
                fparser,
                [&](Type_ value) -> bool {
                    check_num_lines_loop(current_data_line);
                    if (!store(currow, curcol, value)) {
                        return false;
                    }
                    ++current_data_line;
                    increment();
                    return true;
                }
            );

        } else {
            struct Workspace {
                std::vector<char> buffer;
                FieldParser_ fparser;
                std::vector<Type_> contents;
                LineIndex overall_line;
            };

            ThreadPool<Workspace> tp(
                [&](Workspace& work) -> bool {
                    byteme::RawBufferReader reader(reinterpret_cast<const unsigned char*>(work.buffer.data()), work.buffer.size());
                    byteme::PerByteSerial<char, byteme::RawBufferReader*> pb(&reader);
                    return scan_matrix_array_base<Type_>(
                        pb,
                        work.overall_line,
                        work.fparser,
                        [&](Type_ value) -> bool {
                            work.contents.emplace_back(value);
                            return true; // threads cannot quit early in their parallel sections; this (and thus scan_*_base) must always return true.
                        }
                    );
                },
                my_nthreads
            );

            finished = tp.run(
                [&](Workspace& work) -> bool {
                    return configure_parallel_workspace(work);
                },
                [&](Workspace& work) -> bool {
                    for (const auto& val : work.contents) {
                        check_num_lines_loop(current_data_line);
                        if (!store(currow, curcol, val)) {
                            return false;
                        }
                        ++current_data_line;
                        increment();
                    }
                    return true;
                }
            );
        }

        check_num_lines_final(finished, current_data_line);
        return finished;
    }

private:
    template<typename Type_, class Input2_, class FieldParser_, class WrappedStore_>
    bool scan_vector_array_base(Input2_& input, LineIndex& overall_line_count, FieldParser_& fparser, WrappedStore_ wstore) const {
        bool valid = input.valid();
        while (valid) {
            // Handling stray comments, empty lines, and leading spaces.
            if (!skip_lines(input, overall_line_count)) {
                break;
            }
            if (!chomp(input)) {
                throw std::runtime_error("expected at least one field for an array vector on line " + std::to_string(overall_line_count + 1));
            }

            // 'fparser' should leave 'input' at the start of the next line, if any exists.
            ParseInfo<Type_> res = fparser(input, overall_line_count);
            if (!wstore(res.value)) {
                return false;
            }
            ++overall_line_count;
            valid = res.remaining;
        }

        return true;
    }

    template<typename Type_, class FieldParser_, class Store_>
    bool scan_vector_array(Store_ store) {
        bool finished = false;
        LineIndex current_data_line = 0;
        if (my_nthreads == 1) {
            FieldParser_ fparser;
            finished = scan_vector_array_base<Type_>(
                *my_input,
                my_current_line,
                fparser,
                [&](Type_ value) -> bool {
                    check_num_lines_loop(current_data_line);
                    ++current_data_line;
                    return store(current_data_line, 1, value);
                }
            );

        } else {
            struct Workspace {
                std::vector<char> buffer;
                FieldParser_ fparser;
                std::vector<Type_> contents;
                LineIndex overall_line;
            };

            ThreadPool<Workspace> tp(
                [&](Workspace& work) -> bool {
                    byteme::RawBufferReader reader(reinterpret_cast<const unsigned char*>(work.buffer.data()), work.buffer.size());
                    byteme::PerByteSerial<char, byteme::RawBufferReader*> pb(&reader);
                    return scan_vector_array_base<Type_>(
                        pb,
                        work.overall_line,
                        work.fparser,
                        [&](Type_ value) -> bool {
                            work.contents.emplace_back(value);
                            return true; // threads cannot quit early in their parallel sections; this (and thus scan_*_base) must always return true.
                        }
                    );
                },
                my_nthreads
            );

            finished = tp.run(
                [&](Workspace& work) -> bool {
                    return configure_parallel_workspace(work);
                },
                [&](Workspace& work) -> bool {
                    for (const auto& val : work.contents) {
                        check_num_lines_loop(current_data_line);
                        ++current_data_line;
                        if (!store(current_data_line, 1, val)) {
                            return false;
                        }
                    }
                    return true;
                }
            );
        }

        check_num_lines_final(finished, current_data_line);
        return finished;
    }

private:
    void check_preamble() const {
        if (!my_passed_banner || !my_passed_size) {
            throw std::runtime_error("banner or size lines have not yet been parsed");
        }
    }

    template<typename Type_>
    class IntegerFieldParser {
    public:
        template<class Input2_>
        ParseInfo<Type_> operator()(Input2_& input, LineIndex overall_line_count) {
            char firstchar = input.get();
            bool negative = (firstchar == '-');
            if (negative || firstchar == '+') {
                if (!(input.advance())) {
                    throw std::runtime_error("premature termination of an integer on line " + std::to_string(overall_line_count + 1));
                }
            }

            constexpr Type_ upper_limit = std::numeric_limits<Type_>::max();
            constexpr Type_ upper_limit_before_mult = upper_limit / 10;
            constexpr Type_ upper_limit_mod = upper_limit % 10;
            constexpr Type_ lower_limit = std::numeric_limits<Type_>::lowest();
            constexpr Type_ lower_limit_before_mult = lower_limit / 10;
            constexpr Type_ lower_limit_mod = -(lower_limit % 10);

            Type_ val = 0;
            bool found = false;
            while (1) {
                char x = input.get();
                switch (x) {
                    case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                        {
                            Type_ delta = x - '0';
                            // We have to handle negative and positive cases separately as they overflow at different thresholds.
                            if (negative) {
                                // Structuring the conditionals so that it's most likely to short-circuit after only testing the first one.
                                if (val <= lower_limit_before_mult && !(val == lower_limit_before_mult && delta <= lower_limit_mod)) {
                                    throw std::runtime_error("integer underflow on line " + std::to_string(overall_line_count + 1));
                                }
                                val *= 10;
                                val -= delta;
                            } else {
                                if (val >= upper_limit_before_mult && !(val == upper_limit_before_mult && delta <= upper_limit_mod)) {
                                    throw std::runtime_error("integer overflow on line " + std::to_string(overall_line_count + 1));
                                }
                                val *= 10;
                                val += delta;
                            }
                        }
                        found = true;
                        break;
                    case ' ': case '\t': case '\r':
                        if (!advance_and_chomp(input)) { // skipping past the current position before chomping.
                            return ParseInfo<Type_>(val, false);
                        }
                        if (input.get() != '\n') {
                            throw std::runtime_error("more fields than expected on line " + std::to_string(overall_line_count + 1));
                        }
                        return ParseInfo<Type_>(val, input.advance()); // move past the newline.
                    case '\n':
                        // This check only needs to be put here, as all blanks should be chomped before calling
                        // this function; so we must start on a non-blank character. This starting character is either:
                        // - a digit, in which case found = true and this check is unnecessary.
                        // - a non-newline non-digit, in case we throw.
                        // - a newline, in which case we arrive here.
                        if (!found) {
                            throw std::runtime_error("empty integer field on line " + std::to_string(overall_line_count + 1));
                        }
                        return ParseInfo<Type_>(val, input.advance()); // move past the newline.
                    default:
                        throw std::runtime_error("expected an integer value on line " + std::to_string(overall_line_count + 1));
                }

                if (!(input.advance())) {
                    break;
                }
            }

            return ParseInfo<Type_>(val, false);
        }
    };

public:
    /**
     * Scan the file for integer lines, assuming that the field in the banner is `Field::INTEGER`.
     * 
     * @tparam Type_ Type to represent the integer.
     * @tparam Store_ Function to process each line.
     *
     * @param store Function with the signature `void(Index_ row, Index_ column, Type_ value)`,
     * which is passed the corresponding values at each line.
     * Both `row` and `column` will be 1-based indices; for `Object::VECTOR`, `column` will be set to 1.
     * Alternatively, this may return `bool`, where a `false` indicates that the scanning should terminate early and a `true` indicates that the scanning should continue.
     *
     * @return Whether the scanning terminated early, based on `store` returning `false`. 
     */
    template<typename Type_ = int, class Store_>
    bool scan_integer(Store_ store) {
        check_preamble();
        static_assert(std::is_integral<Type_>::value);

        auto wrapped_store = [&](Index_ r, Index_ c, Type_ val) -> bool {
            if constexpr(std::is_same<typename std::invoke_result<Store_, Index_, Index_, Type_>::type, bool>::value) {
                return store(r, c, val);
            } else {
                store(r, c, val);
                return true;
            }
        };

        if (my_details.format == Format::COORDINATE) {
            if (my_details.object == Object::MATRIX) {
                return scan_matrix_coordinate_non_pattern<Type_, IntegerFieldParser<Type_> >(std::move(wrapped_store));
            } else {
                return scan_vector_coordinate_non_pattern<Type_, IntegerFieldParser<Type_> >(std::move(wrapped_store));
            }
        } else {
            if (my_details.object == Object::MATRIX) {
                return scan_matrix_array<Type_, IntegerFieldParser<Type_> >(std::move(wrapped_store));
            } else {
                return scan_vector_array<Type_, IntegerFieldParser<Type_> >(std::move(wrapped_store));
            }
        }
    }

private:
    template<bool last_, typename Type_, typename Input2_>
    static ParseInfo<Type_> parse_real(Input2_& input, std::string& temporary, Index overall_line_count) {
        temporary.clear();
        ParseInfo<Type_> output(0, true);

        while (1) {
            char x = input.get();
            switch(x) {
                case '\n':
                    if constexpr(last_) {
                        // This is actually the only place we need to check for an empty temporary;
                        // it can be assumed that this function is only called after chomping to the next non-blank,
                        // so if it's not a newline, it'll be handled by the default case, and subsequently temporary will be non-empty.
                        if (temporary.empty()) {
                            throw std::runtime_error("empty number field on line " + std::to_string(overall_line_count + 1));
                        }
                        output.remaining = input.advance(); // move past the newline.
                    } else {
                        throw std::runtime_error("unexpected newline on line " + std::to_string(overall_line_count + 1));
                    }
                    goto final_processing;

                case ' ': case '\t': case '\r':
                    if constexpr(last_) {
                        if (!advance_and_chomp(input)) { // skipping past the current position before chomping.
                            output.remaining = false;
                        } else {
                            if (input.get() != '\n') {
                                throw std::runtime_error("more fields than expected on line " + std::to_string(overall_line_count + 1));
                            }
                            output.remaining = input.advance(); // move past the newline.
                        }
                    } else {
                        if (!advance_and_chomp(input)) { // skipping past the current position before chomping.
                            throw std::runtime_error("unexpected end of file on line " + std::to_string(overall_line_count + 1));
                        }
                        if (input.get() == '\n') {
                            throw std::runtime_error("unexpected newline on line " + std::to_string(overall_line_count + 1));
                        }
                    }
                    goto final_processing;

                default: 
                    temporary += x;
                    break;
            }

            if (!(input.advance())) {
                if constexpr(last_) {
                    output.remaining = false;
                    goto final_processing;
                }
                throw std::runtime_error("unexpected end of file on line " + std::to_string(overall_line_count + 1));
            }
        }

final_processing:
        if (temporary.size() >= 2 && temporary[0] == '0' && (temporary[1] == 'x' || temporary[1] == 'X')) {
            throw std::runtime_error("hexadecimal numbers are not allowed on line " + std::to_string(overall_line_count + 1));
        }

        std::size_t n = 0;
        try {
            if constexpr(std::is_same<Type_, float>::value) {
                output.value = std::stof(temporary, &n);
            } else if constexpr(std::is_same<Type_, long double>::value) {
                output.value = std::stold(temporary, &n);
            } else {
                output.value = std::stod(temporary, &n);
            }
        } catch (std::invalid_argument& e) {
            throw std::runtime_error("failed to convert value to a real number on line " + std::to_string(overall_line_count + 1));
        }

        if (n != temporary.size()) {
            throw std::runtime_error("failed to convert value to a real number on line " + std::to_string(overall_line_count + 1));
        }
        return output;
    }

    template<typename Type_>
    class RealFieldParser {
    public:
        template<class Input2_>
        ParseInfo<Type_> operator()(Input2_& input, LineIndex overall_line_count) {
            return parse_real<true, Type_>(input, my_temporary, overall_line_count);
        }
    private:
        std::string my_temporary;
    };

public:
    /**
     * Scan the file for real lines, assuming that the field in the banner is `Field::REAL`.
     * 
     * @tparam Type_ Type to represent the real value.
     * @tparam Store_ Function to process each line.
     *
     * @param store Function with the signature `void(Index_ row, Index_ column, Type_ value)`,
     * which is passed the corresponding values at each line.
     * Both `row` and `column` will be 1-based indices; for `Object::VECTOR`, `column` will be set to 1.
     * Alternatively, this function may return `bool`, where a `false` indicates that the scanning should terminate early and a `true` indicates that the scanning should continue.
     *
     * @return Whether the scanning terminated early, based on `store` returning `false`. 
     */
    template<typename Type_ = double, class Store_>
    bool scan_real(Store_&& store) {
        check_preamble();
        static_assert(std::is_floating_point<Type_>::value);

        auto store_real = [&](Index_ r, Index_ c, Type_ val) -> bool {
            if constexpr(std::is_same<typename std::invoke_result<Store_, Index_, Index_, Type_>::type, bool>::value) {
                return store(r, c, val);
            } else {
                store(r, c, val);
                return true;
            }
        };

        if (my_details.format == Format::COORDINATE) {
            if (my_details.object == Object::MATRIX) {
                return scan_matrix_coordinate_non_pattern<Type_, RealFieldParser<Type_> >(std::move(store_real));
            } else {
                return scan_vector_coordinate_non_pattern<Type_, RealFieldParser<Type_> >(std::move(store_real));
            }
        } else {
            if (my_details.object == Object::MATRIX) {
                return scan_matrix_array<Type_, RealFieldParser<Type_> >(std::move(store_real));
            } else {
                return scan_vector_array<Type_, RealFieldParser<Type_> >(std::move(store_real));
            }
        }
    }

    /**
     * Scan the file for double-precision lines, assuming that the field in the banner is `Field::DOUBLE`.
     * This is just an alias for `scan_real()`.
     * 
     * @tparam Type_ Type to represent the double-precision value.
     * @tparam Store_ Function to process each line.
     *
     * @param store Function with the signature `void(Index_ row, Index_ column, Type_ value)`,
     * which is passed the corresponding values at each line.
     * Both `row` and `column` will be 1-based indices; for `Object::VECTOR`, `column` will be set to 1.
     * Alternatively, this function may return `bool`, where a `false` indicates that the scanning should terminate early and a `true` indicates that the scanning should continue.
     *
     * @return Whether the scanning terminated early, based on `store` returning `false`. 
     */
    template<typename Type_ = double, class Store_>
    bool scan_double(Store_ store) {
        return scan_real<Type_, Store_>(std::move(store));
    }

private:
    template<typename InnerType_>
    class ComplexFieldParser {
    public:
        template<typename Input2_>
        ParseInfo<std::complex<InnerType_> > operator()(Input2_& input, LineIndex overall_line_count) {
            auto first = parse_real<false, InnerType_>(input, my_temporary, overall_line_count);
            auto second = parse_real<true, InnerType_>(input, my_temporary, overall_line_count);
            ParseInfo<std::complex<InnerType_> > output;
            output.value.real(first.value);
            output.value.imag(second.value);
            output.remaining = second.remaining;
            return output;
        }
    private:
        std::string my_temporary;
    };

public:
    /**
     * Scan the file for complex lines, assuming that the field in the banner is `Field::COMPLEX`.
     * 
     * @tparam Type_ Type to represent the real and imaginary parts of the complex value.
     * @tparam Store_ Function to process each line.
     *
     * @param store Function with the signature `void(Index_ row, Index_ column, std::complex<Type_> value)`,
     * which is passed the corresponding values at each line.
     * Both `row` and `column` will be 1-based indices; for `Object::VECTOR`, `column` will be set to 1.
     * Alternatively, this function may return `bool`, where a `false` indicates that the scanning should terminate early and a `true` indicates that the scanning should continue.
     *
     * @return Whether the scanning terminated early, based on `store` returning `false`. 
     */
    template<typename Type_ = double, class Store_>
    bool scan_complex(Store_ store) {
        check_preamble();
        static_assert(std::is_floating_point<Type_>::value);

        typedef std::complex<Type_> FullType;
        auto store_comp = [&](Index_ r, Index_ c, FullType val) -> bool {
            if constexpr(std::is_same<typename std::invoke_result<Store_, Index_, Index_, FullType>::type, bool>::value) {
                return store(r, c, val);
            } else {
                store(r, c, val);
                return true;
            }
        };

        if (my_details.format == Format::COORDINATE) {
            if (my_details.object == Object::MATRIX) {
                return scan_matrix_coordinate_non_pattern<FullType, ComplexFieldParser<Type_> >(std::move(store_comp));
            } else {
                return scan_vector_coordinate_non_pattern<FullType, ComplexFieldParser<Type_> >(std::move(store_comp));
            }
        } else {
            if (my_details.object == Object::MATRIX) {
                return scan_matrix_array<FullType, ComplexFieldParser<Type_> >(std::move(store_comp));
            } else {
                return scan_vector_array<FullType, ComplexFieldParser<Type_> >(std::move(store_comp));
            }
        }
    }

    /**
     * Scan the file for pattern lines, assuming that the field in the banner is `Field::PATTERN`.
     * This function only works when the format field is set to `Format::COORDINATE`.
     * 
     * @tparam Type_ Type to represent the presence of a non-zero entry.
     * @tparam Store_ Function to process each line.
     *
     * @param store Function with the signature `void(Index_ row, Index_ column, Type_ value)`,
     * which is passed the corresponding values at each line.
     * Both `row` and `column` will be 1-based indices; for `Object::VECTOR`, `column` will be set to 1.
     * `value` will always be `true` and can be ignored; it is only required here for consistency with the other methods.
     * Alternatively, this function may return `bool`, where a `false` indicates that the scanning should terminate early and a `true` indicates that the scanning should continue.
     *
     * @return Whether the scanning terminated early, based on `store` returning `false`. 
     */
    template<typename Type_ = bool, class Store_>
    bool scan_pattern(Store_ store) {
        check_preamble();
        if (my_details.format != Format::COORDINATE) {
            throw std::runtime_error("'array' format for 'pattern' field is not supported");
        }

        auto store_pat = [&](Index_ r, Index_ c) -> bool {
            if constexpr(std::is_same<typename std::invoke_result<Store_, Index_, Index_, bool>::type, bool>::value) {
                return store(r, c, true);
            } else {
                store(r, c, true);
                return true;
            }
        };

        if (my_details.object == Object::MATRIX) {
            return scan_matrix_coordinate_pattern(std::move(store_pat));
        } else {
            return scan_vector_coordinate_pattern(std::move(store_pat));
        }
    }
};

}

#endif
