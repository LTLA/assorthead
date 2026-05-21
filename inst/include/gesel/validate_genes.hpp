#ifndef GESEL_VALIDATE_GENES_HPP
#define GESEL_VALIDATE_GENES_HPP

#include "check_genes.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept> 
#include <filesystem>

/**
 * @file validate_genes.hpp
 * @brief Validate gene mapping files.
 */

namespace gesel {

/**
 * Validate Gesel gene mapping files for a particular species.
 * Any invalid formatting or inconsistency between files will result in an error.
 *
 * @param prefix Prefix for the Gesel gene mapping files.
 * This should be of the form `<DIRECTORY>/<SPECIES>_`, where `<SPECIES>` is an NCBI taxonomy ID.
 * @param types Vector of gene name types, e.g., `"ensembl"`, `"symbol"`.
 * This should contain at least one value.
 *
 * @return Number of genes.
 */
inline uint64_t validate_genes(const std::string& prefix, const std::vector<std::string>& types) {
    bool first = true;
    uint64_t num_genes = 0;
    for (auto t : types) {
        auto candidate = internal::check_genes(prefix + t + ".tsv.gz");
        if (first) {
            num_genes = candidate;
            first = false;
        } else if (candidate != num_genes) {
            throw std::runtime_error("inconsistent number of genes between types (" + std::to_string(num_genes) + " for " + types.front() + ", " + std::to_string(candidate) + " for " + t + ")");
        }
    }

    if (first) {
        throw std::runtime_error("at least one gene name type should be present");
    }

    return num_genes;
}

/**
 * Overload for `validate_genes()`.
 * This will scan the directory for all files starting with `prefix` and ending with `".tsv.gz"`.
 *
 * @param prefix Prefix for the Gesel gene files.
 * This should be of the form `<DIRECTORY>/<SPECIES>_`, where `<SPECIES>` is an NCBI taxonomy ID.
 *
 * @return Number of genes.
 */
inline uint64_t validate_genes(const std::string& prefix) {
    std::vector<std::string> types;

    std::filesystem::path path(prefix);
    auto dir = path.parent_path();
    auto raw_prefix = path.filename().string();

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name.rfind(raw_prefix, 0) != 0) {
            continue;
        }
        if (name.size() < 6) {
            continue;
        }
        size_t ext_loc = name.size() - 7;
        if (name.rfind(".tsv.gz", ext_loc) != ext_loc) {
            continue;
        }
        types.push_back(name.substr(raw_prefix.size(), ext_loc - raw_prefix.size()));
    }

    return validate_genes(prefix, types);
}

}

#endif
