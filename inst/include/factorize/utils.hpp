#ifndef FACTORIZE_UTILS_HPP
#define FACTORIZE_UTILS_HPP

#include <type_traits>

namespace factorize {

template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(Input_ x) {
    return x;
}

}

#endif
