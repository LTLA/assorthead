#ifndef GESEL_VALIDATE_DATABASE_HPP
#define GESEL_VALIDATE_DATABASE_HPP

#include "check_collection_details.hpp"
#include "check_indices.hpp"
#include "check_set_details.hpp"
#include "load_ranges.hpp"

#include <string>
#include <cstdint>
#include <stdexcept> 
#include <vector>
#include <unordered_map>

/**
 * @file validate_database.hpp
 * @brief Validate database files.
 */

namespace gesel {

/**
 * @cond
 */
namespace internal {

inline void tokenize(uint64_t index, const std::string& text, std::unordered_map<std::string, std::vector<uint64_t> >& tokens_to_sets) {
    std::string latest;
    auto add = [&]() {
        if (latest.size()) {
            auto& vec = tokens_to_sets[latest];
            if (vec.empty() || vec.back() != index) {
                vec.push_back(index);
            }
            latest.clear();
        }
    };

    for (auto x : text) {
        x = std::tolower(x);
        if (invalid_token_character(x)) {
            add();
        } else {
            latest += x;
        }
    }

    add();
}

inline void check_tokens(const std::vector<std::string>& tokens, const std::string& path) {
    for (size_t t = 0, end = tokens.size(); t < end; ++t) {
        const auto& token = tokens[t];
        if (token.empty()) {
            throw std::runtime_error("token should not be an empty string in '" + path + "' " + append_line_number(t));
        }

        for (auto x : token) {
            if (invalid_token_character(x)) {
                throw std::runtime_error("tokens should only contain lower-case alphabetical characters, digits or a dash in '" + path + "' " + append_line_number(t));
            }
        }

        if (t && token <= tokens[t - 1]) {
            throw std::runtime_error("tokens should be unique and lexicographically sorted in '" + path + "' " + append_line_number(t));
        } 
    }
}

}
/**
 * @endcond
 */

/**
 * Validate Gesel database files for a particular species.
 * This checks all files for validity and consistency except for the gene mapping files (which are validated by `validate_genes()`).
 * Any invalid formatting or inconsistency between files will result in an error.
 *
 * @param prefix Prefix for the Gesel database files.
 * This should be of the form `<DIRECTORY>/<SPECIES>_`, where `<SPECIES>` is an NCBI taxonomy ID.
 * @param num_genes Total number of genes for this species.
 */
inline void validate_database(const std::string& prefix, uint64_t num_genes) {
    uint64_t total_sets = 0;
    {
        auto coll_info = internal::load_ranges_with_sizes(prefix + "collections.tsv.ranges.gz");
        internal::check_collection_details(prefix + "collections.tsv", coll_info.first, coll_info.second);
        constexpr uint64_t limit = std::numeric_limits<uint64_t>::max();
        for (auto x : coll_info.second) {
            if (limit - total_sets < x) {
                throw std::runtime_error("64-bit unsigned integer overflow for the sum of the number of sets in 'collections.tsv.ranges.gz'");
            }
            total_sets += x;
        }
    }

    std::vector<uint64_t> set_sizes;
    {
        auto set_info = internal::load_ranges_with_sizes(prefix + "sets.tsv.ranges.gz");
        if (static_cast<uint64_t>(set_info.first.size()) != total_sets) {
            throw std::runtime_error("total number of sets in 'sets.tsv' does not match with the reported number from 'collections.tsv.ranges.gz'");
        }
        set_sizes.swap(set_info.second);

        std::unordered_map<std::string, std::vector<uint64_t> > token_n, token_d;
        internal::check_set_details(
            prefix + "sets.tsv",
            set_info.first,
            set_sizes,
            [&](uint64_t line, const std::string& name, const std::string& description) {
                internal::tokenize(line, name, token_n);
                internal::tokenize(line, description, token_d);
            }
        );

        // Check for correct tokenization.
        for (int tt = 0; tt < 2; ++tt) {
            std::string type = (tt == 0 ? "names" : "descriptions");
            const auto& tokens = (tt == 0 ? token_n : token_d);

            auto path = "tokens-" + type + ".tsv";
            auto ranges_path = path + ".ranges.gz";
            auto tok_info = internal::load_named_ranges(prefix + ranges_path);
            internal::check_tokens(tok_info.first, ranges_path);
            if (tok_info.first.size() != tokens.size()) {
                throw std::runtime_error("different number of tokens from " + type + " between '" + ranges_path + "' and 'sets.tsv'");
            }

            internal::check_indices<false>(
                prefix + path,
                total_sets,
                tok_info.second,
                [&](uint64_t line, const std::vector<uint64_t>& indices) {
                    const auto& tok = tok_info.first[line];
                    auto tIt = tokens.find(tok);
                    if (tIt == tokens.end()) {
                        throw std::runtime_error("token '" + tok + "' in '" + ranges_path + "' is not present in " + type + " in 'sets.tsv'");
                    }
                    if (!internal::same_vectors(tIt->second, indices)) {
                        throw std::runtime_error("sets for token '" + tok + "' in '" + path + "' are inconsistent with " + type + " in 'sets.tsv'");
                    }
                }
            );
        }
    }

    // Check for correct mapping of sets to genes.
    std::vector<std::vector<uint64_t> > reverse_map(num_genes);
    {
        auto s2g_info = internal::load_ranges(prefix + "set2gene.tsv.ranges.gz");
        if (s2g_info.size() != static_cast<size_t>(total_sets)) {
            throw std::runtime_error("number of lines in 'set2gene.tsv.ranges.gz' does not match the total number of sets");
        }

        internal::check_indices<true>(
            prefix + "set2gene.tsv",
            num_genes,
            s2g_info,
            [&](uint64_t line, const std::vector<uint64_t>& indices) {
                if (static_cast<uint64_t>(indices.size()) != set_sizes[line]) {
                    throw std::runtime_error("size of set " + std::to_string(line) + " from 'sets.tsv.ranges.gz' does not match with that in 'set2gene.tsv'");
                }
                for (auto i : indices) {
                    reverse_map[i].push_back(line);
                }
            }
        );
    }

    // And making sure that the reverse mapping is consistent.
    {
        auto g2s_info = internal::load_ranges(prefix + "gene2set.tsv.ranges.gz");
        if (g2s_info.size() != static_cast<size_t>(num_genes)) {
            throw std::runtime_error("number of lines in 'gene2set.tsv.ranges.gz' does not match the total number of genes");
        }

        internal::check_indices<true>(
            prefix + "gene2set.tsv",
            total_sets,
            g2s_info,
            [&](uint64_t line, const std::vector<uint64_t>& indices) {
                if (!internal::same_vectors(reverse_map[line], indices)) {
                    throw std::runtime_error("sets for gene " + std::to_string(line) + " in 'gene2set.tsv' are inconsistent with 'set2gene.tsv'");
                }
            }
        );
    }
} 

}

#endif
