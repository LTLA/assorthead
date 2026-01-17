#ifndef SCRAN_AGGREGATE_COMBINE_FACTORS_HPP
#define SCRAN_AGGREGATE_COMBINE_FACTORS_HPP

#include "factorize/factorize.hpp"

namespace scran_aggregate {

// Preserving it here for back-compatibility.
template<typename Factor_, typename Combined_>
std::vector<std::vector<Factor_> > combine_factors(const std::size_t n, const std::vector<const Factor_*>& factors, Combined_* const combined) {
    return factorize::combine_to_factor(n, factors, combined);
}

template<typename Factor_, typename Number_, typename Combined_>
std::vector<std::vector<Factor_> > combine_factors_unused(const std::size_t n, const std::vector<std::pair<const Factor_*, Number_> >& factors, Combined_* const combined) {
    return factorize::combine_to_factor_unused(n, factors, combined);
}

}

#endif
