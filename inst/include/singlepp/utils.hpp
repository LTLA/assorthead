#ifndef SINGLEPP_UTILS_HPP
#define SINGLEPP_UTILS_HPP

#include <type_traits>
#include <algorithm>
#include <vector>

namespace singlepp {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

template<typename First_, typename Second_>
void sort_by_first(std::vector<std::pair<First_, Second_> >& x) {
    std::sort(
        x.begin(),
        x.end(), 
        [](const std::pair<First_, Second_>& left, const std::pair<First_, Second_>& right) -> bool {
            return left.first < right.first;
        }
    );
}

template<typename Number_, typename Value_>
bool is_sorted_unique(const Number_ n, const Value_* data) {
    for (Number_ i = 1; i < n; ++i) {
        if (data[i] <= data[i-1]) {
            return false;
        }
    }
    return true;
}

}

#endif
