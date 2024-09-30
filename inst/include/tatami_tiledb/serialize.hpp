#ifndef TATAMI_TILEDB_SERIALIZE_HPP
#define TATAMI_TILEDB_SERIALIZE_HPP

/**
 * @file serialize.hpp
 * @brief Locking for serial access.
 */

namespace tatami_tiledb {

/**
 * Serialize a function's execution to avoid simultaneous calls to the TileDB library.
 * By default, no locking is performed as TileDB is thread-safe,
 * but enforcing serialization may be helpful on filesystems with I/O bottlenecks. 
 *
 * @tparam Function_ Function that accepts no arguments and returns no outputs.
 * @param fun Function to be run in a serial section.
 */
template<class Function_>
void serialize(Function_ fun) {
#ifdef TATAMI_TILEDB_PARALLEL_LOCK
    TATAMI_TILEDB_PARALLEL_LOCK(fun);
#else
    fun();
#endif
}

}

#endif
