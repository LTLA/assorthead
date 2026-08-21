#ifndef TATAMI_CHUNKED_UTILS_HPP
#define TATAMI_CHUNKED_UTILS_HPP

#include <type_traits>

namespace tatami_chunked {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

}

#endif
