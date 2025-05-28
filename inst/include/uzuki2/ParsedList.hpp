#ifndef UZUKI2_PARSED_LIST_HPP
#define UZUKI2_PARSED_LIST_HPP

#include <memory>

#include "interfaces.hpp"
#include "Version.hpp"

/**
 * @file ParsedList.hpp
 * @brief Class to hold the parsed list.
 */

namespace uzuki2 {

/**
 * @brief Results of parsing a list from file.
 */
struct ParsedList {
public:
    /**
     * @cond
     */
    ParsedList(std::shared_ptr<Base> p, Version v) : version(std::move(v)), ptr(std::move(p)) {}
    /**
     * @endcond
     */

    /**
     * Version of the **uzuki2** specification.
     */
    Version version;

    /**
     * Pointer to the `Base` object.
     */
    std::shared_ptr<Base> ptr;

public:
    /**
     * @cond
     */
    // Provided for back-compatibility only.
    Base* get() const {
        return ptr.get();
    }

    Base& operator*() const {
        return *ptr;
    }

    Base* operator->() const {
        return ptr.operator->();
    }

    operator bool() const {
        return ptr.operator bool();
    }

    template<typename ...Args_>
    void reset(Args_&& ... args) const {
        ptr.reset(std::forward<Args_>(args)...);
    }
    /**
     * @endcond
     */
};

}

#endif
