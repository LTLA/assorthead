#ifndef GESEL_CHECK_COLLECTION_DETAILS_HPP
#define GESEL_CHECK_COLLECTION_DETAILS_HPP

#include <string>
#include <cstdint>
#include <vector>

#include "byteme/byteme.hpp"

#include "parse_field.hpp"

namespace gesel {

namespace internal {

inline void check_collection_details(const std::string& path, const std::vector<uint64_t>& ranges, const std::vector<uint64_t>& numbers) {
    byteme::RawFileReader raw_r(path.c_str(), {});
    auto gzpath = path + ".gz";
    byteme::GzipFileReader gzip_r(gzpath.c_str(), {});

    byteme::SerialBufferedReader<char, decltype(&raw_r)> raw_p(&raw_r, 65536);
    byteme::SerialBufferedReader<char, decltype(&gzip_r)> gzip_p(&gzip_r, 65536);

    bool raw_valid = raw_p.valid();
    bool gzip_valid = gzip_p.valid();
    uint64_t line = 0;
    const uint64_t num_ranges = ranges.size();

    while (raw_valid) {
        auto raw_pos = raw_p.position();
        auto title = parse_string_field<FieldType::MIDDLE>(raw_p, raw_valid, path, line);
        auto description = parse_string_field<FieldType::MIDDLE>(raw_p, raw_valid, path, line);
        auto species = parse_integer_field<FieldType::MIDDLE>(raw_p, raw_valid, path, line);
        auto maintainer = parse_string_field<FieldType::MIDDLE>(raw_p, raw_valid, path, line);
        auto source = parse_string_field<FieldType::LAST>(raw_p, raw_valid, path, line);

        if (line >= num_ranges) {
            throw std::runtime_error("number of lines in '" + path + "' exceeds that expected from its '*.ranges.gz' file " + append_line_number(line));
        }
        if (raw_p.position() - raw_pos - 1 != static_cast<size_t>(ranges[line])) {
            throw std::runtime_error("number of bytes per line in '" + path + "' is not the same as that expected from the '*.ranges.gz' file " + append_line_number(line));
        }

        if (!gzip_valid) {
            throw std::runtime_error("early termination of the Gzipped version of '" + path + "'");
        }

        auto gz_title = parse_string_field<FieldType::MIDDLE>(gzip_p, gzip_valid, path, line);
        if (gz_title != title) {
            throw std::runtime_error("different title in '" + path + "' compared to its Gzipped version " + append_line_number(line));
        }

        auto gz_description = parse_string_field<FieldType::MIDDLE>(gzip_p, gzip_valid, path, line);
        if (gz_description != description) {
            throw std::runtime_error("different description in '" + path + "' compared to its Gzipped version " + append_line_number(line));
        }

        auto gz_species = parse_integer_field<FieldType::MIDDLE>(gzip_p, gzip_valid, path, line);
        if (gz_species != species) {
            throw std::runtime_error("different species in '" + path + "' compared to its Gzipped version " + append_line_number(line));
        }

        auto gz_maintainer = parse_string_field<FieldType::MIDDLE>(gzip_p, gzip_valid, path, line);
        if (gz_maintainer != maintainer) {
            throw std::runtime_error("different maintainer in '" + path + "' compared to its Gzipped version " + append_line_number(line));
        }

        auto gz_source = parse_string_field<FieldType::MIDDLE>(gzip_p, gzip_valid, path, line);
        if (gz_source != source) {
            throw std::runtime_error("different source in '" + path + "' compared to its Gzipped version " + append_line_number(line));
        }

        auto gz_number = parse_integer_field<FieldType::LAST>(gzip_p, gzip_valid, path, line);
        if (gz_number != numbers[line]) {
            throw std::runtime_error("different number in '" + path + ".gz' compared to its '*.ranges.gz' file " + append_line_number(line));
        }

        ++line;
    }

    if (line != num_ranges) {
        throw std::runtime_error("number of lines in '" + path + "' is less than that expected from its '*.ranges.gz' file " + append_line_number(line));
    }
}

}

}

#endif
