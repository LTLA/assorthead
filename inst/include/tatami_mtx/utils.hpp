#ifndef TATAMI_MTX_UTILS_HPP
#define TATAMI_MTX_UTILS_HPP

#include <type_traits>

namespace tatami_mtx {

template<typename Value_>
using I = std::remove_cv_t<std::remove_reference_t<Value_> >;

}

#endif
