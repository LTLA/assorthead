#ifndef RAIIGRAPH_RNG_SCOPE_HPP
#define RAIIGRAPH_RNG_SCOPE_HPP

#include "igraph.h"
#include "error.hpp"

/**
 * @file RNGScope.hpp
 * @brief Control the **igraph** RNG via RAII.
 */

namespace raiigraph {

/**
 * @brief Control the **igraph** RNG via RAII.
 *
 * When an instance of this class is created, it will replace the global default **igraph** RNG with its own.
 * When it is destroyed, it will restore the default to the RNG that was present before its construction.
 *
 * It is assumed that users have already called `igraph_setup()` before constructing a instance of this class.
 */
class RNGScope {
public:
    /**
     * Sets PCG32 as the default RNG with the specified `seed`.
     *
     * @param seed Seed for the RNG.
     */
    RNGScope(igraph_uint_t seed) : RNGScope(seed, &igraph_rngtype_pcg32) {}

    /**
     * Sets the specified RNG type as the default with the specified `seed`.
     *
     * @param seed Seed for the RNG.
     * @param type Pointer to the RNG type.
     */
    RNGScope(igraph_uint_t seed, const igraph_rng_type_t* type) {
        check_code(igraph_rng_init(&current, type));

        auto errcode = igraph_rng_seed(&current, seed);
        if (errcode != IGRAPH_SUCCESS) {
            igraph_rng_destroy(&current);
            throw IgraphError(errcode);
        }

        previous = igraph_rng_set_default(&current);
    }

    /**
     * Sets the specified RNG type as the default with its default seed.
     *
     * @param type Pointer to the RNG type.
     */
    RNGScope(const igraph_rng_type_t* type) {
        check_code(igraph_rng_init(&current, type));
        previous = igraph_rng_set_default(&current);
    }

    /**
     * Sets PCG32 as the default with its default seed.
     */
    RNGScope() : RNGScope(&igraph_rngtype_pcg32) {}

public:
    /**
     * @cond
     */
    // We shouldn't be copying or moving an RNGscope object, as this defeats the logic provided by scoping.
    RNGScope(const RNGScope&) = delete;
    RNGScope& operator=(const RNGScope&) = delete;
    RNGScope(RNGScope&&) = delete;
    RNGScope& operator=(RNGScope&&) = delete;

    ~RNGScope() {
        igraph_rng_set_default(previous);
        igraph_rng_destroy(&current);
    }
    /**
     * @endcond
     */

private:
    igraph_rng_t* previous;
    igraph_rng_t current;
};

}

#endif
