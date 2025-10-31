#ifndef RAIIGRAPH_ERROR_HPP
#define RAIIGRAPH_ERROR_HPP

#include "igraph.h"

/**
 * @file error.hpp
 * @brief Error handling for **raiigraph**.
 */

namespace raiigraph {

/**
 * @brief Error class for **igraph**-related errors.
 */
class IgraphError : public std::runtime_error {
public:
    /**
     * @param code Error code returned by **igraph** functions.
     */
    IgraphError(igraph_error_t code) : std::runtime_error(igraph_strerror(code)), my_code(code) {}

    /**
     * @return Error code returned by **igraph** functions.
     * This should be anything but `IGRAPH_SUCCESS`.
     */
    igraph_error_t code() const {
        return my_code;
    }

private:
    igraph_error_t my_code;
};

/**
 * Throw an `IgraphError` if `code` is not `IGRAPH_SUCCESS`.
 * @param code Error code returned by **igraph** functions.
 */
inline void check_code(igraph_error_t code) {
    if (code != IGRAPH_SUCCESS) {
        throw IgraphError(code);
    }
}

}

#endif
