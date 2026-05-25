#ifndef GESEL_CHECK_VERSION_HPP
#define GESEL_CHECK_VERSION_HPP

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

#include "byteme/byteme.hpp"
#include "sanisizer/sanisizer.hpp"

namespace gesel {

namespace internal {

struct Version {
    int major = 0;
    int minor = 0;
    int patch = 0;

public:
    Version() = default;
    Version(int major, int minor, int patch) : major(major), minor(minor), patch(patch) {}

public:
    bool operator==(const Version& other) const {
        return (other.major == major && other.minor == minor && other.patch == patch);
    }

    bool operator<(const Version& other) const {
        if (other.major == major) {
            if (other.minor == minor) {
                return patch < other.patch;
            } else {
                return minor < other.minor;
            }
        } else {
            return major < other.major;
        }
    }

    bool operator>(const Version& other) const {
        if (other.major == major) {
            if (other.minor == minor) {
                return patch > other.patch;
            } else {
                return minor > other.minor;
            }
        } else {
            return major > other.major;
        }
    }

    bool operator!=(const Version& other) const {
        return !(*this == other);
    }

    bool operator>=(const Version& other) const {
        return !(*this < other);
    }

    bool operator<=(const Version& other) const {
        return !(*this > other);
    }
};

inline Version check_version(const std::string& version_path) {
    byteme::RawFileReader reader(version_path.c_str(), {});
    std::vector<char> buffer(10);
    typename std::vector<char>::size_type offset = 0;
    while (1) {
        const auto remaining = sanisizer::cast<std::size_t>(buffer.size() - offset);
        const auto read = reader.read(reinterpret_cast<unsigned char*>(buffer.data() + offset), remaining);
        if (read < remaining) {
            buffer.resize(offset + read);
            break;
        }
        offset = buffer.size();
        buffer.insert(buffer.end(), 10, static_cast<char>(0));
    }

    Version output;
    typename std::vector<char>::size_type position = 0;
    const auto limit = buffer.size();

    for (int mode = 0; mode < 3; ++mode) {
        const char term = (mode < 2 ? '.' : '\n');
        auto& target = [&]() -> int& {
            if (mode == 0) {
                return output.major;
            } else if (mode == 1) {
                return output.minor;
            } else {
                return output.patch;
            }
        }();

        if (limit <= position) {
            throw std::runtime_error("premature termination of the version string at '" + version_path + "'");
        }
        const auto start = buffer[position];
        if (start == '0') {
            ++position;
            if (limit <= position) {
                throw std::runtime_error("premature termination of the version string at '" + version_path + "'");
            } else if (buffer[position] != term) {
                throw std::runtime_error("leading zeros detected in the version string at '" + version_path + "'");
            }
            ++position;
        } else if (start > '0' && start <= '9') {
            auto current = start;
            while (1) {
                target *= 10;
                target += current - '0';
                ++position;
                if (limit <= position) {
                    throw std::runtime_error("premature termination of the version string at '" + version_path + "'");
                } 
                current = buffer[position];
                if (current == term) {
                    ++position;
                    break;
                } else if (current < '0' || current > '9') {
                    throw std::runtime_error("non-digit characters detected in the version string at '" + version_path + "'");
                }
            }
        } else if (start == term) {
            throw std::runtime_error("empty field in the version string at '" + version_path + "'");
        } else {
            throw std::runtime_error("non-digit characters detected in the version string at '" + version_path + "'");
        }
    }

    if (limit != position) {
        throw std::runtime_error("additional characters after the version string at '" + version_path + "'");
    }
    return output;
}

}

}

#endif
