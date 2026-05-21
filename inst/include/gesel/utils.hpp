#ifndef GESEL_UTILS_HPP
#define GESEL_UTILS_HPP

#include <string>
#include <limits>
#include <cstdint>

namespace gesel {

namespace internal {

inline std::string append_line_number(uint64_t line) {
    if (line == std::numeric_limits<uint64_t>::max()) {
        return " (line 18446744073709551616)";
    } else {
        return " (line " + std::to_string(line + 1) + ")";
    }
}

inline bool invalid_token_character(char x) {
    return (x < 'a' || x > 'z') && (x < '0' || x > '9') && (x != '-');
}

template<typename Value_>
bool same_vectors(const std::vector<Value_>& left, const std::vector<Value_>& right) {
    size_t num_left = left.size();
    if (num_left != right.size()) {
        return false;
    }
    for (size_t i = 0; i < num_left; ++i) {
        if (left[i] != right[i]) {
            return false;
        }
    }
    return true;
}

}

}

#endif
