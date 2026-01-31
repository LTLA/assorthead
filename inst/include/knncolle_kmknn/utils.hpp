#ifndef KNNCOLLE_KMKNN_UTILS_HPP
#define KNNCOLLE_KMKNN_UTILS_HPP

#include <fstream>
#include <string>
#include <type_traits>

namespace knncolle_kmknn {

template<typename Input_>
using I = std::remove_const_t<std::remove_reference_t<Input_> >;

}

#endif
