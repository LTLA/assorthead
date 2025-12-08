#ifndef SCRAN_BLOCKS_UTILS_HPP
#define SCRAN_BLOCKS_UTILS_HPP

#include <type_traits>

namespace scran_blocks {

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

}

#endif
