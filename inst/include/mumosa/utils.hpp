#ifndef MUMOSA_UTILS_HPP
#define MUMOSA_UTILS_HPP

#include <type_traits>

namespace mumosa {

/**
 * @cond
 */
template<typename Input_>
using I = typename std::remove_cv<typename std::remove_reference<Input_>::type>::type;
/**
 * @endcond
 */

}

#endif
