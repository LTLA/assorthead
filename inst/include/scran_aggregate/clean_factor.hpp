#ifndef SCRAN_AGGREGATE_CLEAN_FACTORS_HPP
#define SCRAN_AGGREGATE_CLEAN_FACTORS_HPP

#include "factorize/factorize.hpp"

namespace scran_aggregate {

// Preserving it here for back-compatibility.
template<typename Factor_, typename Output_>
std::vector<Factor_> clean_factor(const std::size_t n, const Factor_* const factor, Output_* const cleaned) {
    return factorize::create_factor(n, factor, cleaned);
}

}

#endif
