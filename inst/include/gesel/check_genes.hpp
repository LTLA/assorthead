#ifndef GESEL_CHECK_GENES_HPP
#define GESEL_CHECK_GENES_HPP

#include <limits>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>

#include "byteme/byteme.hpp"

#include "parse_field.hpp"
#include "utils.hpp"

namespace gesel {

namespace internal {

inline uint64_t check_genes(const std::string& path) {
    byteme::GzipFileReader reader(path.c_str(), {});
    byteme::SerialBufferedReader<char, decltype(&reader)> pb(&reader, 65536);
    std::vector<uint64_t> output;

    bool valid = pb.valid();
    uint64_t line = 0;
    constexpr uint64_t max_line = std::numeric_limits<uint64_t>::max();
    std::unordered_set<std::string> current_names;

    while (valid) {
        if (pb.get() == '\n') {
            valid = pb.advance();
        } else {
            current_names.clear();
            do {
                auto parsed = parse_string_field<FieldType::UNKNOWN>(pb, valid, path, line);
                if (parsed.first == "") {
                    throw std::runtime_error("empty name detected in '" + path + "' " + append_line_number(line)); 
                }
                if (current_names.find(parsed.first) != current_names.end()) {
                    throw std::runtime_error("duplicated names detected in '" + path + "' " + append_line_number(line)); 
                }
                if (parsed.second) {
                    break;
                }
                current_names.insert(parsed.first);
            } while (true);
        }

        if (line == max_line) {
            throw std::runtime_error("number of lines should fit in a 32-bit integer"); 
        }
        ++line;
    }

    return line;
}

}

}

#endif
