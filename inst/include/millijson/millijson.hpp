#ifndef MILLIJSON_MILLIJSON_HPP
#define MILLIJSON_MILLIJSON_HPP

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>

#include "byteme/byteme.hpp"
#include "sanisizer/sanisizer.hpp"

/**
 * @file millijson.hpp
 * @brief Header-only library for JSON parsing.
 */

/**
 * @namespace millijson
 * @brief A lightweight header-only JSON parser.
 */
namespace millijson {

/**
 * All known JSON types.
 * `NUMBER_AS_STRING` indicates a JSON number that is represented as its input string.
 */
enum Type {
    NUMBER,
    NUMBER_AS_STRING,
    STRING,
    BOOLEAN,
    NOTHING,
    ARRAY,
    OBJECT
};

/**
 * @brief Virtual base class for all JSON types.
 */
class Base {
public:
    /**
     * @return Type of the JSON value.
     */
    virtual Type type() const = 0;

    /**
     * @cond
     */
    Base() = default;
    Base(Base&&) = default;
    Base(const Base&) = default;
    Base& operator=(Base&&) = default;
    Base& operator=(const Base&) = default;
    virtual ~Base() {}
    /**
     * @endcond
     */
};

/**
 * @brief JSON number.
 */
class Number final : public Base {
public:
    /**
     * @param x Value of the number.
     */
    Number(double x) : my_value(x) {}

    Type type() const { return NUMBER; }

public:
    /**
     * @return Value of the number.
     */
    const double& value() const { return my_value; }

    /**
     * @return Value of the number.
     */
    double& value() { return my_value; }

private:
    double my_value;
};

/**
 * @brief JSON number as a string.
 */
class NumberAsString final : public Base {
public:
    /**
     * @param x Value of the number as a string.
     */
    NumberAsString(std::string x) : my_value(x) {}

    Type type() const { return NUMBER_AS_STRING; }

public:
    /**
     * @return Value of the number.
     */
    const std::string& value() const { return my_value; }

    /**
     * @return Value of the number.
     */
    std::string& value() { return my_value; }

private:
    std::string my_value;
};

/**
 * @brief JSON string.
 */
class String final : public Base {
public:
    /**
     * @param x Value of the string.
     */
    String(std::string x) : my_value(std::move(x)) {}

    Type type() const { return STRING; }

public:
    /**
     * @return Value of the string.
     */
    const std::string& value() const { return my_value; }

    /**
     * @return Value of the string.
     */
    std::string& value() { return my_value; }

private:
    std::string my_value;
};

/**
 * @brief JSON boolean.
 */
class Boolean final : public Base {
public:
    /**
     * @param x Value of the boolean.
     */
    Boolean(bool x) : my_value(x) {}

    Type type() const { return BOOLEAN; }

public:
    /**
     * @return Value of the boolean.
     */
    const bool& value() const { return my_value; }

    /**
     * @return Value of the string.
     */
    bool& value() { return my_value; }

private:
    bool my_value;
};

/**
 * @brief JSON null.
 */
class Nothing final : public Base {
public:
    Type type() const { return NOTHING; }
};

/**
 * @brief JSON array.
 */
class Array final : public Base {
public:
    /**
     * @param x Contents of the array.
     */
    Array(std::vector<std::shared_ptr<Base> > x) : my_value(std::move(x)) {}

    Type type() const { return ARRAY; }

public:
    /**
     * @return Contents of the array.
     */
    const std::vector<std::shared_ptr<Base> >& value() const {
        return my_value;
    }

    /**
     * @return Contents of the array.
     */
    std::vector<std::shared_ptr<Base> >& value() {
        return my_value;
    }

private:
    std::vector<std::shared_ptr<Base> > my_value;
};

/**
 * @brief JSON object.
 */
class Object final : public Base {
public:
     /**
     * @param x Key-value pairs of the object.
     */
    Object(std::unordered_map<std::string, std::shared_ptr<Base> > x) : my_value(std::move(x)) {}

    Type type() const { return OBJECT; }

public:
    /**
     * @return Key-value pairs of the object.
     */
    const std::unordered_map<std::string, std::shared_ptr<Base> >& value() const {
        return my_value;
    }

    /**
     * @return Key-value pairs of the object.
     */
    std::unordered_map<std::string, std::shared_ptr<Base> >& value() {
        return my_value;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Base> > my_value;
};

/**
 * @brief Options for `parse()`.
 */
struct ParseOptions {
    /**
     * Whether to preserve the string representation for numbers.
     * If true, all JSON numbers will be loaded as `NumberAsString`.
     * Otherwise, they will be loaded as `Number`.
     */
    bool number_as_string = false;

    /**
     * Size of the buffer for storing bytes before parsing.
     * Larger values typically improve speed at the cost of memory efficiency.
     */
    std::size_t buffer_size = sanisizer::cap<std::size_t>(65536);

    /**
     * Whether to parse and read bytes in parallel.
     * This can reduce runtime on multi-core machines.
     */
    bool parallel = false;
};

/**
 * @cond
 */
// Return value of the various chomp functions indicates whether there are any
// characters left in 'input', allowing us to avoid an extra call to valid(). 
template<class Input_>
bool raw_chomp(Input_& input, bool ok) {
    while (ok) {
        switch(input.get()) {
            // Allowable whitespaces as of https://www.rfc-editor.org/rfc/rfc7159#section-2.
            case ' ': case '\n': case '\r': case '\t':
                break;
            default:
                return true;
        }
        ok = input.advance();
    }
    return false;
}

template<class Input_>
bool check_and_chomp(Input_& input) {
    bool ok = input.valid();
    return raw_chomp(input, ok);
}

template<class Input_>
bool advance_and_chomp(Input_& input) {
    bool ok = input.advance();
    return raw_chomp(input, ok);
}

inline bool is_digit(char val) {
    return val >= '0' && val <= '9';
}

template<class Input_>
bool is_expected_string(Input_& input, const char* ptr, std::size_t len) {
    // We use a hard-coded 'len' instead of scanning for '\0' to enable loop unrolling.
    for (std::size_t i = 1; i < len; ++i) {
        // The current character was already used to determine what string to
        // expect, so we can skip past it in order to match the rest of the
        // string. This is also why we start from i = 1 instead of i = 0.
        if (!input.advance()) {
            return false;
        }
        if (input.get() != ptr[i]) {
            return false;
        }
    }
    input.advance(); // move off the last character.
    return true;
}

template<class Input_>
std::string extract_string(Input_& input) {
    unsigned long long start = input.position() + 1;
    input.advance(); // get past the opening quote.
    std::string output;

    while (1) {
        char next = input.get();
        switch (next) {
            case '"':
                input.advance(); // get past the closing quote.
                return output;

            case '\\':
                if (!input.advance()) {
                    throw std::runtime_error("unterminated string at position " + std::to_string(start));
                } else {
                    char next2 = input.get();
                    switch (next2) {
                        case '"':
                            output += '"';          
                            break;
                        case 'n':
                            output += '\n';
                            break;
                        case 'r':
                            output += '\r';
                            break;
                        case '\\':
                            output += '\\';
                            break;
                        case '/':
                            output += '/';
                            break;
                        case 'b':
                            output += '\b';
                            break;
                        case 'f':
                            output += '\f';
                            break;
                        case 't':
                            output += '\t';
                            break;
                        case 'u':
                            {
                                unsigned short mb = 0;
                                for (int i = 0; i < 4; ++i) {
                                    if (!input.advance()){
                                        throw std::runtime_error("unterminated string at position " + std::to_string(start));
                                    }
                                    mb *= 16;
                                    char val = input.get();
                                    switch (val) {
                                        case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                                            mb += val - '0';
                                            break;
                                        case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': 
                                            mb += (val - 'a') + 10;
                                            break;
                                        case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': 
                                            mb += (val - 'A') + 10;
                                            break;
                                        default:
                                            throw std::runtime_error("invalid unicode escape detected at position " + std::to_string(input.position() + 1));
                                    }
                                }

                                // Manually convert Unicode code points to UTF-8. We only allow
                                // 3 bytes at most because there's only 4 hex digits in JSON. 
                                if (mb <= 127) {
                                    output += static_cast<char>(mb);
                                } else if (mb <= 2047) {
                                    unsigned char left = (mb >> 6) | 0b11000000;
                                    output += *(reinterpret_cast<char*>(&left));
                                    unsigned char right = (mb & 0b00111111) | 0b10000000;
                                    output += *(reinterpret_cast<char*>(&right));
                                } else {
                                    unsigned char left = (mb >> 12) | 0b11100000;
                                    output += *(reinterpret_cast<char*>(&left));
                                    unsigned char middle = ((mb >> 6) & 0b00111111) | 0b10000000;
                                    output += *(reinterpret_cast<char*>(&middle));
                                    unsigned char right = (mb & 0b00111111) | 0b10000000;
                                    output += *(reinterpret_cast<char*>(&right));
                                }
                            }
                            break;
                        default:
                            throw std::runtime_error("unrecognized escape '\\" + std::string(1, next2) + "'");
                    }
                }
                break;

            case (char) 0: case (char) 1: case (char) 2: case (char) 3: case (char) 4: case (char) 5: case (char) 6: case (char) 7: case (char) 8: case (char) 9:
            case (char)10: case (char)11: case (char)12: case (char)13: case (char)14: case (char)15: case (char)16: case (char)17: case (char)18: case (char)19:
            case (char)20: case (char)21: case (char)22: case (char)23: case (char)24: case (char)25: case (char)26: case (char)27: case (char)28: case (char)29:
            case (char)30: case (char)31:
            case (char)127:
                throw std::runtime_error("string contains ASCII control character at position " + std::to_string(input.position() + 1));

            default:
                output += next;
                break;
        }

        if (!input.advance()) {
            throw std::runtime_error("unterminated string at position " + std::to_string(start));
        }
    }

    return output; // Technically unreachable, but whatever.
}

template<bool as_string_, class Input_>
typename std::conditional<as_string_, std::string, double>::type extract_number(Input_& input) {
    unsigned long long start = input.position() + 1;
    auto value = []{
        if constexpr(as_string_) {
            return std::string("");
        } else {
            return static_cast<double>(0);
        }
    }();
    bool in_fraction = false;
    bool in_exponent = false;

    auto add_string_value = [&](char x) -> void {
        if constexpr(as_string_) {
            value += x;
        }
    };

    // We assume we're starting from the absolute value, after removing any preceding negative sign.
    char lead = input.get();
    add_string_value(lead);
    if (lead == '0') {
        if (!input.advance()) {
            return value;
        }

        auto after_zero = input.get();
        switch (after_zero) {
            case '.':
                add_string_value(after_zero);
                in_fraction = true;
                break;
            case 'e': case 'E':
                add_string_value(after_zero);
                in_exponent = true;
                break;
            case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                return value;
            default:
                throw std::runtime_error("invalid number starting with 0 at position " + std::to_string(start));
        }

    } else { // 'lead' must be a digit, as extract_number is only called when the current character is a digit.
        if constexpr(!as_string_) {
            value += lead - '0';
        }

        while (input.advance()) {
            char val = input.get();
            switch (val) {
                case '.':
                    add_string_value(val);
                    in_fraction = true;
                    goto integral_end;
                case 'e': case 'E':
                    add_string_value(val);
                    in_exponent = true;
                    goto integral_end;
                case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                    goto total_end;
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    if constexpr(as_string_) {
                        value += val;
                    } else {
                        value *= 10;
                        value += val - '0';
                    }
                    break;
                default:
                    throw std::runtime_error("invalid number containing '" + std::string(1, val) + "' at position " + std::to_string(start));
            }
        }

integral_end:;
    }

    if (in_fraction) {
        if (!input.advance()) {
            throw std::runtime_error("invalid number with trailing '.' at position " + std::to_string(start));
        }

        char val = input.get();
        if (!is_digit(val)) {
            throw std::runtime_error("'.' must be followed by at least one digit at position " + std::to_string(start));
        }

        double fractional = 10;
        if constexpr(as_string_) {
            value += val;
        } else {
            value += (val - '0') / fractional;
        }

        while (input.advance()) {
            char val = input.get();
            switch (val) {
                case 'e': case 'E':
                    in_exponent = true;
                    add_string_value(val);
                    goto fraction_end;
                case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                    goto total_end;
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    if constexpr(as_string_) {
                        value += val;
                    } else {
                        fractional *= 10;
                        value += (val - '0') / fractional;
                    }
                    break;
                default:
                    throw std::runtime_error("invalid number containing '" + std::string(1, val) + "' at position " + std::to_string(start));
            }
        } 

fraction_end:;
    }

    if (in_exponent) {
        double exponent = 0; 
        bool negative_exponent = false;

        if (!input.advance()) {
            throw std::runtime_error("invalid number with trailing 'e/E' at position " + std::to_string(start));
        }

        char val = input.get();
        if (!is_digit(val)) {
            if (val == '-') {
                negative_exponent = true;
                add_string_value(val);
            } else if (val != '+') {
                throw std::runtime_error("'e/E' should be followed by a sign or digit in number at position " + std::to_string(start));
            }

            if (!input.advance()) {
                throw std::runtime_error("invalid number with trailing exponent sign at position " + std::to_string(start));
            }
            val = input.get();
            if (!is_digit(val)) {
                throw std::runtime_error("exponent sign must be followed by at least one digit in number at position " + std::to_string(start));
            }
        }

        if constexpr(as_string_) {
            value += val;
        } else {
            exponent += (val - '0');
        }

        while (input.advance()) {
            char val = input.get();
            switch (val) {
                case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                    goto exponent_end;
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    if constexpr(as_string_) {
                        value += val;
                    } else {
                        exponent *= 10;
                        exponent += (val - '0');
                    }
                    break;
                default:
                    throw std::runtime_error("invalid number containing '" + std::string(1, val) + "' at position " + std::to_string(start));
            }
        }

exponent_end:
        if constexpr(!as_string_) {
            if (exponent) {
                if (negative_exponent) {
                    exponent *= -1;
                }
                value *= std::pow(10.0, exponent);
            }
        }
    }

total_end:
    return value;
}

struct FakeProvisioner {
    class FakeBase {
    public:
        virtual Type type() const = 0;
        virtual ~FakeBase() {}
    };
    typedef FakeBase Base;

    class FakeBoolean final : public FakeBase {
    public:
        Type type() const { return BOOLEAN; }
    };
    static FakeBoolean* new_boolean(bool) {
        return new FakeBoolean; 
    }

    class FakeNumber final : public FakeBase {
    public:    
        Type type() const { return NUMBER; }
    };
    static FakeNumber* new_number(double) {
        return new FakeNumber;
    }

    class FakeNumberAsString final : public FakeBase {
    public:
        Type type() const { return NUMBER_AS_STRING; }
    };
    static FakeNumberAsString* new_number_as_string(std::string) {
        return new FakeNumberAsString;
    }

    class FakeString final : public FakeBase {
    public:
        Type type() const { return STRING; }
    };
    static FakeString* new_string(std::string) {
        return new FakeString;
    }

    class FakeNothing final : public FakeBase {
    public:
        Type type() const { return NOTHING; }
    };
    static FakeNothing* new_nothing() {
        return new FakeNothing;
    }

    class FakeArray final : public FakeBase {
    public:
        Type type() const { return ARRAY; }
    };
    static FakeArray* new_array(std::vector<std::shared_ptr<FakeBase> >) {
        return new FakeArray;
    }

    class FakeObject final : public FakeBase {
    public:
        Type type() const { return OBJECT; }
    };
    static FakeObject* new_object(std::unordered_map<std::string, std::shared_ptr<FakeBase> >) {
        return new FakeObject;
    }
};

template<class Provisioner_, class Input_>
std::shared_ptr<typename Provisioner_::Base> parse_internal(Input_& input, const ParseOptions& options) {
    if (!check_and_chomp(input)) {
        throw std::runtime_error("invalid JSON with no contents");
    }

    // The most natural algorithm for parsing nested JSON arrays/objects would involve recursion,
    // but we avoid this to eliminate the associated risk of stack overflows (and maybe improve perf?).
    // Instead, we use an iterative algorithm with a manual stack for the two nestable JSON types.
    // We only have to worry about OBJECTs and ARRAYs so there's only two sets of states to manage.
    std::vector<Type> stack;
    typedef std::vector<std::shared_ptr<typename Provisioner_::Base> > ArrayContents;
    std::vector<ArrayContents> array_stack;
    struct ObjectContents {
        ObjectContents() = default;
        ObjectContents(std::string key) : key(std::move(key)) {}
        std::unordered_map<std::string, std::shared_ptr<typename Provisioner_::Base> > mapping;
        std::string key;
    };
    std::vector<ObjectContents> object_stack;

    unsigned long long start = input.position() + 1;
    auto extract_object_key = [&]() -> std::string {
        char next = input.get();
        if (next != '"') {
            throw std::runtime_error("expected a string as the object key at position " + std::to_string(input.position() + 1));
        }
        auto key = extract_string(input);
        if (!check_and_chomp(input)) {
            throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
        }
        if (input.get() != ':') {
            throw std::runtime_error("expected ':' to separate keys and values at position " + std::to_string(input.position() + 1));
        }
        if (!advance_and_chomp(input)) {
            throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
        }
        return key;
    };

    std::shared_ptr<typename Provisioner_::Base> output;
    while (1) {
        const char current = input.get();
        switch(current) {
            case 't':
                if (!is_expected_string(input, "true", 4)) {
                    throw std::runtime_error("expected a 'true' string at position " + std::to_string(start));
                }
                output.reset(Provisioner_::new_boolean(true));
                break;

            case 'f':
                if (!is_expected_string(input, "false", 5)) {
                    throw std::runtime_error("expected a 'false' string at position " + std::to_string(start));
                }
                output.reset(Provisioner_::new_boolean(false));
                break;

            case 'n':
                if (!is_expected_string(input, "null", 4)) {
                    throw std::runtime_error("expected a 'null' string at position " + std::to_string(start));
                }
                output.reset(Provisioner_::new_nothing());
                break;

            case '"': 
                output.reset(Provisioner_::new_string(extract_string(input)));
                break;

            case '[':
                if (!advance_and_chomp(input)) {
                    throw std::runtime_error("unterminated array starting at position " + std::to_string(start));
                }
                if (input.get() != ']') {
                    stack.push_back(ARRAY);
                    array_stack.emplace_back();
                    continue; // prepare to parse the first element of the array.
                } 
                input.advance(); // move past the closing bracket.
                output.reset(Provisioner_::new_array(std::vector<std::shared_ptr<typename Provisioner_::Base> >{}));
                break;

            case '{':
                if (!advance_and_chomp(input)) {
                    throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                }
                if (input.get() != '}') {
                    stack.push_back(OBJECT);
                    object_stack.emplace_back(extract_object_key());
                    continue; // prepare to parse the first value of the object.
                }
                input.advance(); // move past the closing brace.
                output.reset(Provisioner_::new_object(std::unordered_map<std::string, std::shared_ptr<typename Provisioner_::Base> >{}));
                break;

            case '-':
                if (!input.advance()) {
                    throw std::runtime_error("incomplete number starting at position " + std::to_string(start));
                }
                if (!is_digit(input.get())) {
                    throw std::runtime_error("invalid number starting at position " + std::to_string(start));
                }
                if (options.number_as_string) {
                    output.reset(Provisioner_::new_number_as_string("-" + extract_number<true>(input)));
                } else {
                    output.reset(Provisioner_::new_number(-extract_number<false>(input)));
                }
                break;

            case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                if (options.number_as_string) {
                    output.reset(Provisioner_::new_number_as_string(extract_number<true>(input)));
                } else {
                    output.reset(Provisioner_::new_number(extract_number<false>(input)));
                }
                break;

            default:
                throw std::runtime_error(std::string("unknown type starting with '") + std::string(1, current) + "' at position " + std::to_string(start));
        }

        while (1) {
            if (stack.empty()) {
                goto parse_finish; // double-break to save ourselves a conditional.
            }

            if (stack.back() == ARRAY) {
                auto& contents = array_stack.back();
                contents.emplace_back(std::move(output));

                if (!check_and_chomp(input)) {
                    throw std::runtime_error("unterminated array starting at position " + std::to_string(start));
                }

                char next = input.get();
                if (next == ',') {
                    if (!advance_and_chomp(input)) {
                        throw std::runtime_error("unterminated array starting at position " + std::to_string(start));
                    }
                    break; // prepare to parse the next entry of the array.
                }
                if (next != ']') {
                    throw std::runtime_error("unknown character '" + std::string(1, next) + "' in array at position " + std::to_string(input.position() + 1));
                }
                input.advance(); // skip the closing bracket.

                output.reset(Provisioner_::new_array(std::move(contents)));
                stack.pop_back();
                array_stack.pop_back();

            } else {
                auto& mapping = object_stack.back().mapping;
                auto& key = object_stack.back().key;
                if (mapping.find(key) != mapping.end()) {
                    throw std::runtime_error("detected duplicate keys in the object at position " + std::to_string(input.position() + 1));
                }
                mapping[std::move(key)] = std::move(output); // consuming the key here.

                if (!check_and_chomp(input)) {
                    throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                }

                char next = input.get();
                if (next == ',') {
                    if (!advance_and_chomp(input)) {
                        throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                    }
                    key = extract_object_key();
                    break; // prepare to parse the next value of the object.
                }
                if (next != '}') {
                    throw std::runtime_error("unknown character '" + std::string(1, next) + "' in array at position " + std::to_string(input.position() + 1));
                }
                input.advance(); // skip the closing brace.

                output.reset(Provisioner_::new_object(std::move(mapping)));
                stack.pop_back();
                object_stack.pop_back();
            }
        }
    }

parse_finish:;
    if (check_and_chomp(input)) {
        throw std::runtime_error("invalid JSON with trailing non-space characters at position " + std::to_string(input.position() + 1));
    }
    return output;
}
/**
 * @endcond
 */

/**
 * @brief Default methods to provision representations of JSON types.
 */
struct DefaultProvisioner {
    /**
     * Alias for the base class for all JSON representations.
     * All classes returned by `new_*` methods should be derived from this class.
     */
    typedef ::millijson::Base Base;

    /**
     * @param x Value of the boolean.
     * @return Pointer to a new JSON boolean instance.
     */
    static Boolean* new_boolean(bool x) {
        return new Boolean(x); 
    }

    /**
     * @param x Value of the number.
     * @return Pointer to a new JSON number instance.
     */
    static Number* new_number(double x) {
        return new Number(x);
    }

    /**
     * @param x Value of the number as a string.
     * @return Pointer to a new JSON number instance.
     */
    static NumberAsString* new_number_as_string(std::string x) {
        return new NumberAsString(std::move(x));
    }

    /**
     * @param x Value of the string.
     * @return Pointer to a new JSON string instance.
     */
    static String* new_string(std::string x) {
        return new String(std::move(x));
    }

    /**
     * @return Pointer to a new JSON null instance.
     */
    static Nothing* new_nothing() {
        return new Nothing;
    }

    /**
     * @param x Contents of the JSON array.
     * @return Pointer to a new JSON array instance.
     */
    static Array* new_array(std::vector<std::shared_ptr<Base> > x) {
        return new Array(std::move(x));
    }

    /**
     * @param x Contents of the JSON object.
     * @return Pointer to a new JSON object instance.
     */
    static Object* new_object(std::unordered_map<std::string, std::shared_ptr<Base> > x) {
        return new Object(std::move(x));
    }
};

/**
 * @cond
 */
template<typename Input_>
auto setup_buffered_reader(Input_& input, const ParseOptions& options) {
    std::unique_ptr<byteme::BufferedReader<char> > ptr;
    if (options.parallel) {
        ptr.reset(new byteme::ParallelBufferedReader<char, Input_*>(&input, options.buffer_size)); 
    } else {
        ptr.reset(new byteme::SerialBufferedReader<char, Input_*>(&input, options.buffer_size)); 
    }
    return ptr;
}
/**
 * @endcond
 */

/**
 * Parse a stream of input bytes for a JSON value, based on the specification at https://json.org.
 *
 * No consideration is given to floating-point overflow for arbitrarily large numbers. 
 * On systems that support IEEE754 arithmetic, overflow will manifest as infinities in `Number`, otherwise it is undefined behavior.
 * If overflow is undesirable, consider setting `ParseOptions::number_as_string` to manually control conversion after parsing.
 *
 * @tparam Provisioner_ Class that provide methods for provisioning each JSON type, see `DefaultProvisioner` for an example.
 * All types should be subclasses of the provisioner's base class (which may but is not required to be `Base`).
 * @tparam Input_ Class of the source of input bytes.
 * This should satisfy the `byteme::Reader` interface.
 
 * @param input A source of input bytes, usually from a JSON-formatted file or string.
 * @param options Further options for parsing.
 *
 * @return A pointer to a JSON value.
 */
template<class Provisioner_ = DefaultProvisioner, class Input_ = byteme::Reader>
std::shared_ptr<typename DefaultProvisioner::Base> parse(Input_& input, const ParseOptions& options) {
    auto iptr = setup_buffered_reader(input, options);
    return parse_internal<Provisioner_>(*iptr, options);
}

/**
 * Check that a string contains a valid JSON value.
 * This follows the same logic as `parse()` but is more memory-efficient.
 *
 * @tparam Input_ Any class that supplies input characters, see `parse()` for details. 
 *
 * @param input A source of input bytes, usually from a JSON-formatted file or string.
 * @param options Further options for parsing.
 *
 * @return The type of the JSON variable stored in `input`.
 * If the JSON string is invalid, an error is raised.
 */
template<class Input_ = byteme::Reader>
Type validate(Input_& input, const ParseOptions& options) {
    auto iptr = setup_buffered_reader(input, options);
    auto ptr = parse_internal<FakeProvisioner>(*iptr, options);
    return ptr->type();
}

/**
 * Parse a string containing a JSON value using `parse()`. 
 *
 * @tparam Provisioner_ Class that provide methods for provisioning each JSON type, see `DefaultProvisioner` for an example.
 * All types should be subclasses of the provisioner's base class (which may but is not required to be `Base`).
 * @param[in] ptr Pointer to an array containing a JSON string.
 * @param len Length of the array.
 * @param options Further options for parsing.
 * @return A pointer to a JSON value.
 */
template<class Provisioner_ = DefaultProvisioner>
inline std::shared_ptr<typename Provisioner_::Base> parse_string(const char* ptr, std::size_t len, const ParseOptions& options) {
    byteme::RawBufferReader input(reinterpret_cast<const unsigned char*>(ptr), len);
    return parse<Provisioner_>(input, options);
}

/**
 * Check that a string contains a valid JSON value using `validate()`. 
 *
 * @param[in] ptr Pointer to an array containing a JSON string.
 * @param len Length of the array.
 * @param options Further options for parsing.
 *
 * @return The type of the JSON variable stored in the string.
 * If the JSON string is invalid, an error is raised.
 */
inline Type validate_string(const char* ptr, std::size_t len, const ParseOptions& options) {
    byteme::RawBufferReader input(reinterpret_cast<const unsigned char*>(ptr), len);
    return validate(input, options);
}

/**
 * Parse a file containing a JSON value using `parse()`. 
 *
 * @param[in] path Pointer to an array containing a path to a JSON file.
 * @param options Further options.
 *
 * @return A pointer to a JSON value.
 */
template<class Provisioner_ = DefaultProvisioner>
std::shared_ptr<Base> parse_file(const char* path, const ParseOptions& options) {
    byteme::RawFileReader input(path, {});
    return parse(input, options);
}

/**
 * Check that a file contains a valid JSON value using `validate()`. 
 *
 * @param[in] path Pointer to an array containing a path to a JSON file.
 * @param options Further options.
 *
 * @return The type of the JSON variable stored in the file.
 * If the JSON file is invalid, an error is raised.
 */
inline Type validate_file(const char* path, const ParseOptions& options) {
    byteme::RawFileReader input(path, {});
    return validate(input, options);
}

/**
 * @cond
 */
// Back-compatibility only.
typedef ParseOptions FileReadOptions;

template<class Provisioner_ = DefaultProvisioner, class Input_>
std::shared_ptr<typename DefaultProvisioner::Base> parse(Input_& input) {
    return parse<Provisioner_>(input, {});
}

template<class Input_>
Type validate(Input_& input) {
    return validate(input, {});
}

template<class Provisioner_ = DefaultProvisioner>
inline std::shared_ptr<typename Provisioner_::Base> parse_string(const char* ptr, std::size_t len) {
    return parse_string<Provisioner_>(ptr, len, {});
}

inline Type validate_string(const char* ptr, std::size_t len) {
    return validate_string(ptr, len, {});
}
/**
 * @endcond
 */

}

#endif
