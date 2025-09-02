#ifndef SANISIZER_UTILS_HPP
#define SANISIZER_UTILS_HPP

#include <stdexcept>
#include <type_traits>

/**
 * @file utils.hpp
 * @brief General-purposes utilities. 
 */

namespace sanisizer {

/**
 * @brief Error from an integer overflow.
 *
 * This allow callers of various **sanisizer** functions to catch overflows and handle them accordingly.
 */
class OverflowError final : public std::runtime_error {
public:
    /**
     * @cond
     */
    template<typename ... Args_>
    OverflowError(Args_&&... args) : std::runtime_error(std::forward<Args_>(args)...) {}
    /**
     * @endcond
     */
};

/**
 * @cond
 */
template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(Input_ x) {
    return x;
}
/**
 * @endcond
 */

}

#endif
