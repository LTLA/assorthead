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
 */
enum Type {
    NUMBER,
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

template<class Input_>
double extract_number(Input_& input) {
    unsigned long long start = input.position() + 1;
    double value = 0;
    bool in_fraction = false;
    bool in_exponent = false;

    // We assume we're starting from the absolute value, after removing any preceding negative sign.
    char lead = input.get();
    if (lead == '0') {
        if (!input.advance()) {
            return 0;
        }

        switch (input.get()) {
            case '.':
                in_fraction = true;
                break;
            case 'e': case 'E':
                in_exponent = true;
                break;
            case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                return value;
            default:
                throw std::runtime_error("invalid number starting with 0 at position " + std::to_string(start));
        }

    } else { // 'lead' must be a digit, as extract_number is only called when the current character is a digit.
        value += lead - '0';

        while (input.advance()) {
            char val = input.get();
            switch (input.get()) {
                case '.':
                    in_fraction = true;
                    goto integral_end;
                case 'e': case 'E':
                    in_exponent = true;
                    goto integral_end;
                case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                    goto total_end;
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    value *= 10;
                    value += val - '0';
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
        value += (val - '0') / fractional;

        while (input.advance()) {
            char val = input.get();
            switch (input.get()) {
                case 'e': case 'E':
                    in_exponent = true;
                    goto fraction_end;
                case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                    goto total_end;
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    fractional *= 10;
                    value += (val - '0') / fractional;
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

        exponent += (val - '0');

        while (input.advance()) {
            char val = input.get();
            switch (val) {
                case ',': case ']': case '}': case ' ': case '\r': case '\n': case '\t':
                    goto exponent_end;
                case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                    exponent *= 10;
                    exponent += (val - '0');
                    break;
                default:
                    throw std::runtime_error("invalid number containing '" + std::string(1, val) + "' at position " + std::to_string(start));
            }
        }

exponent_end:
        if (exponent) {
            if (negative_exponent) {
                exponent *= -1;
            }
            value *= std::pow(10.0, exponent);
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
std::shared_ptr<typename Provisioner_::Base> parse_thing(Input_& input) {
    std::shared_ptr<typename Provisioner_::Base> output;

    unsigned long long start = input.position() + 1;
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
            {
                std::vector<std::shared_ptr<typename Provisioner_::Base> > contents;

                if (!advance_and_chomp(input)) {
                    throw std::runtime_error("unterminated array starting at position " + std::to_string(start));
                }

                if (input.get() != ']') {
                    while (1) {
                        contents.push_back(parse_thing<Provisioner_>(input));

                        if (!check_and_chomp(input)) {
                            throw std::runtime_error("unterminated array starting at position " + std::to_string(start));
                        }

                        char next = input.get();
                        if (next == ']') {
                            break;
                        } else if (next != ',') {
                            throw std::runtime_error("unknown character '" + std::string(1, next) + "' in array at position " + std::to_string(input.position() + 1));
                        }

                        if (!advance_and_chomp(input)) {
                            throw std::runtime_error("unterminated array starting at position " + std::to_string(start));
                        }
                    }
                }

                output.reset(Provisioner_::new_array(std::move(contents)));
            }
            input.advance(); // skip the closing bracket.
            break;

        case '{':
            {
                std::unordered_map<std::string, std::shared_ptr<typename Provisioner_::Base> > contents;

                if (!advance_and_chomp(input)) {
                    throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                }

                if (input.get() != '}') {
                    while (1) {
                        char next = input.get();
                        if (next != '"') {
                            throw std::runtime_error("expected a string as the object key at position " + std::to_string(input.position() + 1));
                        }
                        auto key = extract_string(input);
                        if (contents.find(key) != contents.end()) {
                            throw std::runtime_error("detected duplicate keys in the object at position " + std::to_string(input.position() + 1));
                        }

                        if (!check_and_chomp(input)) {
                            throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                        }
                        if (input.get() != ':') {
                            throw std::runtime_error("expected ':' to separate keys and values at position " + std::to_string(input.position() + 1));
                        }

                        if (!advance_and_chomp(input)) {
                            throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                        }
                        contents[std::move(key)] = parse_thing<Provisioner_>(input); // consuming the key here.

                        if (!check_and_chomp(input)) {
                            throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                        }

                        next = input.get();
                        if (next == '}') {
                            break;
                        } else if (next != ',') {
                            throw std::runtime_error("unknown character '" + std::string(1, next) + "' in array at position " + std::to_string(input.position() + 1));
                        }

                        if (!advance_and_chomp(input)) {
                            throw std::runtime_error("unterminated object starting at position " + std::to_string(start));
                        }
                    }
                }

                output.reset(Provisioner_::new_object(std::move(contents)));
            }
            input.advance(); // skip the closing brace.
            break;

        case '-':
            if (!input.advance()) {
                throw std::runtime_error("incomplete number starting at position " + std::to_string(start));
            }
            if (!is_digit(input.get())) {
                throw std::runtime_error("invalid number starting at position " + std::to_string(start));
            }
            output.reset(Provisioner_::new_number(-extract_number(input)));
            break;

        case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
            output.reset(Provisioner_::new_number(extract_number(input)));
            break;

        default:
            throw std::runtime_error(std::string("unknown type starting with '") + std::string(1, current) + "' at position " + std::to_string(start));
    }

    return output;
}

template<class Provisioner_, class Input_>
std::shared_ptr<typename Provisioner_::Base> parse_thing_with_chomp(Input_& input) {
    if (!check_and_chomp(input)) {
        throw std::runtime_error("invalid JSON with no contents");
    }
    auto output = parse_thing<Provisioner_>(input);
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
 * @tparam Provisioner_ Class that provide methods for provisioning each JSON type, see `DefaultProvisioner` for an example.
 * All types should be subclasses of the provisioner's base class (which may but is not required to be `Base`).
 * @tparam Input_ Class of the source of input bytes.
 * This should satisfy the [`byteme::PerByteInterface`](https://ltla.github.io/byteme) interface with the following methods:
 *
 * - `char get() const `, which extracts a `char` from the input source without advancing the position on the byte stream.
 * - `bool valid() const`, to determine whether an input `char` can be `get()` from the input.
 * - `bool advance()`, to advance the input stream and return `valid()` at the new position.
 * - `unsigned long long position() const`, for the current position relative to the start of the byte stream.
 *
 * @param input A source of input bytes, usually from a JSON-formatted file or string.
 * @return A pointer to a JSON value.
 */
template<class Provisioner_ = DefaultProvisioner, class Input_>
std::shared_ptr<typename DefaultProvisioner::Base> parse(Input_& input) {
    return parse_thing_with_chomp<Provisioner_>(input);
}

/**
 * @tparam Input_ Any class that supplies input characters, see `parse()` for details. 
 *
 * @param input A source of input bytes, usually from a JSON-formatted file or string.
 *
 * @return The type of the JSON variable stored in `input`.
 * If the JSON string is invalid, an error is raised.
 */
template<class Input_>
Type validate(Input_& input) {
    auto ptr = parse_thing_with_chomp<FakeProvisioner>(input);
    return ptr->type();
}

/**
 * @cond
 */
class RawReader {
public:
    RawReader(const char* ptr, std::size_t len) : my_ptr(ptr), my_len(len) {}

private:
    unsigned long long my_pos = 0;
    const char * my_ptr;
    std::size_t my_len;

public:
    char get() const {
        return my_ptr[my_pos];
    }

    bool valid() const {
        return my_pos < my_len;
    }

    bool advance() {
        ++my_pos;
        return valid();
    }

    unsigned long long position() const {
        return my_pos;
    }
};
/**
 * @endcond
 */

/**
 * @tparam Provisioner_ Class that provide methods for provisioning each JSON type, see `DefaultProvisioner` for an example.
 * All types should be subclasses of the provisioner's base class (which may but is not required to be `Base`).
 * @param[in] ptr Pointer to an array containing a JSON string.
 * @param len Length of the array.
 * @return A pointer to a JSON value.
 */
template<class Provisioner_ = DefaultProvisioner>
inline std::shared_ptr<typename Provisioner_::Base> parse_string(const char* ptr, std::size_t len) {
    RawReader input(ptr, len);
    return parse<Provisioner_>(input);
}

/**
 * @param[in] ptr Pointer to an array containing a JSON string.
 * @param len Length of the array.
 *
 * @return The type of the JSON variable stored in the string.
 * If the JSON string is invalid, an error is raised.
 */
inline Type validate_string(const char* ptr, std::size_t len) {
    RawReader input(ptr, len);
    return validate(input);
}

/**
 * @cond
 */
class FileReader {
public:
    FileReader(const char* path, std::size_t buffer_size) : my_handle(std::fopen(path, "rb")), my_buffer(check_buffer_size(buffer_size)) {
        if (!my_handle) {
            throw std::runtime_error("failed to open file at '" + std::string(path) + "'");
        }
        fill();
    }

    ~FileReader() {
        std::fclose(my_handle);
    }

public:
    typedef typename std::vector<char>::size_type Size;

    static Size check_buffer_size(std::size_t buffer_size) {
        // Usually this is a no-op as the vector::size_type is a size_t.
        // But it doesn't hurt to confirm that it will fit properly.
        constexpr Size max_size = std::numeric_limits<Size>::max();
        if (buffer_size >= max_size) { // size_type should be unsigned, so at least this comparison is safe.
            return max_size;
        } else {
            return buffer_size;
        }
    }

private:
    std::FILE* my_handle;
    std::vector<char> my_buffer;
    Size my_available = 0;
    Size my_index = 0;
    unsigned long long my_overall = 0;
    bool my_finished = false;

public:
    char get() const {
        return my_buffer[my_index];
    }

    bool valid() const {
        return my_index < my_available;
    }

    bool advance() {
        ++my_index;
        if (my_index < my_available) {
            return true;
        }

        my_index = 0;
        my_overall += my_available;
        fill();
        return valid();
    }

    void fill() {
        if (my_finished) {
            my_available = 0;
            return;
        }

        my_available = std::fread(my_buffer.data(), sizeof(char), my_buffer.size(), my_handle);
        if (my_available == my_buffer.size()) {
            return;
        }

        if (std::feof(my_handle)) {
            my_finished = true;
        } else {
            throw std::runtime_error("failed to read file (error " + std::to_string(std::ferror(my_handle)) + ")");
        }
    }

    unsigned long long position() const {
        return my_overall + my_index;
    }
};
/**
 * @endcond
 */

/**
 * @brief Options for `parse_file()` and `validate_file()`.
 */
struct FileReadOptions {
    /**
     * Size of the buffer to use for reading the file.
     */
    std::size_t buffer_size = 65536;
};

/**
 * @param[in] path Pointer to an array containing a path to a JSON file.
 * @param options Further options.
 * @return A pointer to a JSON value.
 */
template<class Provisioner_ = DefaultProvisioner>
std::shared_ptr<Base> parse_file(const char* path, const FileReadOptions& options) {
    FileReader input(path, options.buffer_size);
    return parse(input);
}

/**
 * @param[in] path Pointer to an array containing a path to a JSON file.
 * @param options Further options.
 *
 * @return The type of the JSON variable stored in the file.
 * If the JSON file is invalid, an error is raised.
 */
inline Type validate_file(const char* path, const FileReadOptions& options) {
    FileReader input(path, options.buffer_size);
    return validate(input);
}

}

#endif
