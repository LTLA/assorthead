#ifndef SCRAN_AGGREGATE_UTILS_HPP
#define SCRAN_AGGREGATE_UTILS_HPP

#include <type_traits>

namespace scran_aggregate {

template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;

}

#endif
