#ifndef TATAMI_STATS__UTILS_HPP
#define TATAMI_STATS__UTILS_HPP

#include <vector>
#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "sanisizer/sanisizer.hpp"
#include "tatami/tatami.hpp"
#include "quickstats/quickstats.hpp"

namespace tatami_stats {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

template<typename Value_, typename Index_>
Index_ shift_nans(Value_* const ptr, const Index_ num) {
    return quickstats::skip_values(
        num, // conversion of Index_ to a std::size_t is safe due to tatami's guarantees. 
        ptr,
        [](const std::size_t, const Value_ val) -> bool {
            return std::isnan(val);
        }
    ); 
}

template<typename Value_, class If_, class Else_>
void nanable_ifelse(bool skip_nan, If_ iffun, Else_ elsefun) {
    if constexpr(std::numeric_limits<Value_>::has_quiet_NaN) {
        if (skip_nan) {
            iffun();
            return;
        }
    }
    elsefun();
}

template<typename Value_, class If_, class Else_>
auto nanable_ifelse_with_value(bool skip_nan, If_ iffun, Else_ elsefun) {
    if constexpr(std::numeric_limits<Value_>::has_quiet_NaN) {
        if (skip_nan) {
            return iffun();
        }
    }
    return elsefun();
}

}

#endif
