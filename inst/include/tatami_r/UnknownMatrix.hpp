#ifndef TATAMI_R_UNKNOWNMATRIX_HPP
#define TATAMI_R_UNKNOWNMATRIX_HPP

#include "Rcpp.h"
#include "tatami/tatami.hpp"

#include "parallelize.hpp"
#include "dense_extractor.hpp"
#include "sparse_extractor.hpp"

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <optional>
#include <cstddef>

/**
 * @file UnknownMatrix.hpp
 * @brief **tatami** binding for an unknown R matrix.
 */

namespace tatami_r {

/**
 * @brief Options for data extraction from an `UnknownMatrix`.
 */
struct UnknownMatrixOptions {
    /**
     * Size of the cache, in bytes.
     * If not set, this is determined from `DelayedArray::getAutoBlockSize()`.
     */
    std::optional<std::size_t> maximum_cache_size;

    /**
     * Whether to automatically enforce a minimum size for the cache, regardless of `maximum_cache_size`.
     * This minimum is chosen to ensure that all chunks overlapping one row (or a slice/subset thereof) can be retained in memory,
     * so that the same chunks are not repeatedly re-read from disk when iterating over consecutive rows/columns of the matrix.
     */
    bool require_minimum_cache = true;
};

/**
 * @brief Unknown matrix-like object in R.
 *
 * @tparam Value_ Numeric type of data value for the interface.
 * @tparam Index_ Integer type for the row/column indices, for the interface.
 *
 * Pull data out of an unknown matrix-like object by calling methods from the [**DelayedArray**](https://bioconductor.org/packages/DelayedArray) package via **Rcpp**.
 * This effectively extends **tatami** to work with any abstract numeric matrix that might be consumed by an R function.
 * 
 * Instances of class should only be constructed and destroyed in a serial context, specifically on the same thread running R itself. 
 * Calls to its methods may be parallelized but some additional effort is required to serialize calls to the R API; see `executor()` for more details.
 */
template<typename Value_, typename Index_, typename CachedValue_ = Value_, typename CachedIndex_ = Index_>
class UnknownMatrix : public tatami::Matrix<Value_, Index_> {
public:
    /**
     * This constructor should only be called in a serial context, as the (default) construction of **Rcpp** objects may call the R API.
     *
     * @param seed A matrix-like R object.
     * @param opt Extraction options.
     */
    UnknownMatrix(Rcpp::RObject seed, const UnknownMatrixOptions& opt) : 
        my_original_seed(seed), 
        my_delayed_env(Rcpp::Environment::namespace_env("DelayedArray")),
        my_sparse_env(Rcpp::Environment::namespace_env("SparseArray")),
        my_dense_extractor(my_delayed_env["extract_array"]),
        my_sparse_extractor(my_sparse_env["extract_sparse_array"])
    {
        // We assume the constructor only occurs on the main thread, so we
        // won't bother locking things up. I'm also not sure that the
        // operations in the initialization list are thread-safe.

        {
            const auto base = Rcpp::Environment::base_env();
            const Rcpp::Function fun = base["dim"];
            const Rcpp::RObject output = fun(seed);
            if (output.sexp_type() != INTSXP) {
                auto ctype = get_class_name(my_original_seed);
                throw std::runtime_error("'dim(<" + ctype + ">)' should return an integer vector");
            }

            const Rcpp::IntegerVector dims(output);
            if (dims.size() != 2 || dims[0] < 0 || dims[1] < 0) {
                auto ctype = get_class_name(my_original_seed);
                throw std::runtime_error("'dim(<" + ctype + ">)' should contain two non-negative integers");
            }

            // If this cast is okay, all subsequent casts from 'int' to 'Index_' will be okay.
            // This is because all subsequent casts will involve values that are smaller than 'dims', e.g., chunk extents.
            // For example, an ArbitraryArrayGrid is restricted by the ticks, while a RegularArrayGrid must have chunkdim <= refdim.
            my_nrow = sanisizer::cast<Index_>(dims[0]);
            my_ncol = sanisizer::cast<Index_>(dims[1]);

            // Checking that we can safely create an Rcpp::IntegerVector without overfllow.
            // We do it here once, so that we don't need to check in each call to consecutive_indices() or increment_indices() or whatever.
            tatami::can_cast_Index_to_container_size<Rcpp::IntegerVector>(std::max(my_nrow, my_ncol));
        }

        {
            const Rcpp::Function fun = my_delayed_env["is_sparse"];
            const Rcpp::LogicalVector is_sparse = fun(seed);
            if (is_sparse.size() != 1) {
                auto ctype = get_class_name(my_original_seed);
                throw std::runtime_error("'is_sparse(<" + ctype + ">)' should return a logical vector of length 1");
            }
            my_sparse = (is_sparse[0] != 0);
        }

        {
            tatami::resize_container_to_Index_size(my_row_chunk_map, my_nrow);
            tatami::resize_container_to_Index_size(my_col_chunk_map, my_ncol);

            const Rcpp::Function fun = my_delayed_env["chunkGrid"];
            const Rcpp::RObject grid = fun(seed);

            if (grid == R_NilValue) {
                my_row_max_chunk_size = 1;
                my_col_max_chunk_size = 1;
                std::iota(my_row_chunk_map.begin(), my_row_chunk_map.end(), static_cast<Index_>(0));
                std::iota(my_col_chunk_map.begin(), my_col_chunk_map.end(), static_cast<Index_>(0));
                my_row_chunk_ticks.resize(sanisizer::sum<decltype(my_row_chunk_ticks.size())>(my_nrow, 1));
                std::iota(my_row_chunk_ticks.begin(), my_row_chunk_ticks.end(), static_cast<Index_>(0));
                my_col_chunk_ticks.resize(sanisizer::sum<decltype(my_col_chunk_ticks.size())>(my_ncol, 1));
                std::iota(my_col_chunk_ticks.begin(), my_col_chunk_ticks.end(), static_cast<Index_>(0));

                // Both dense and sparse inputs are implicitly column-major, so
                // if there isn't chunking information to the contrary, we'll
                // favor extraction of the columns.
                my_prefer_rows = false;

            } else {
                auto grid_cls = get_class_name(grid);

                if (grid_cls == "RegularArrayGrid") {
                    const Rcpp::IntegerVector spacings(Rcpp::RObject(grid.slot("spacings")));
                    if (spacings.size() != 2) {
                        auto ctype = get_class_name(seed);
                        throw std::runtime_error("'chunkGrid(<" + ctype + ">)@spacings' should be an integer vector of length 2 with non-negative values");
                    }

                    const auto populate = [](
                        const Index_ extent,
                        const Index_ spacing,
                        std::vector<Index_>& map,
                        std::vector<Index_>& ticks
                    ) -> void {
                        if (spacing == 0) {
                            ticks.push_back(0);
                        } else {
                            ticks.reserve((extent / spacing) + (extent % spacing > 0) + 1);
                            Index_ start = 0;
                            ticks.push_back(start);
                            while (start != extent) {
                                auto to_fill = std::min(spacing, extent - start);
                                std::fill_n(map.begin() + start, to_fill, ticks.size() - 1);
                                start += to_fill;
                                ticks.push_back(start);
                            }
                        }
                    };

                    my_row_max_chunk_size = spacings[0];
                    populate(my_nrow, my_row_max_chunk_size, my_row_chunk_map, my_row_chunk_ticks);
                    my_col_max_chunk_size = spacings[1];
                    populate(my_ncol, my_col_max_chunk_size, my_col_chunk_map, my_col_chunk_ticks);

                } else if (grid_cls == "ArbitraryArrayGrid") {
                    const Rcpp::List ticks(Rcpp::RObject(grid.slot("tickmarks")));
                    if (ticks.size() != 2) {
                        auto ctype = get_class_name(seed);
                        throw std::runtime_error("'chunkGrid(<" + ctype + ">)@tickmarks' should return a list of length 2");
                    }

                    const auto populate = [](
                        const Index_ extent,
                        const Rcpp::IntegerVector& ticks,
                        std::vector<Index_>& map,
                        std::vector<Index_>& new_ticks,
                        Index_& max_chunk_size
                    ) -> void {
                        if (ticks.size() != 0 && ticks[ticks.size() - 1] != static_cast<int>(extent)) {
                            throw std::runtime_error("invalid ticks returned by 'chunkGrid'");
                        }
                        new_ticks.resize(sanisizer::sum<decltype(new_ticks.size())>(ticks.size(), 1));
                        std::copy(ticks.begin(), ticks.end(), new_ticks.begin() + 1);

                        max_chunk_size = 0;
                        int start = 0;
                        tatami::resize_container_to_Index_size(map, extent);
                        Index_ counter = 0;

                        for (auto t : ticks) {
                            if (t < start) {
                                throw std::runtime_error("invalid ticks returned by 'chunkGrid'");
                            }
                            Index_ to_fill = t - start;
                            if (to_fill > max_chunk_size) {
                                max_chunk_size = to_fill;
                            }
                            std::fill_n(map.begin() + start, to_fill, counter);
                            ++counter;
                            start = t;
                        }
                    };

                    Rcpp::IntegerVector first(ticks[0]);
                    populate(my_nrow, first, my_row_chunk_map, my_row_chunk_ticks, my_row_max_chunk_size);
                    Rcpp::IntegerVector second(ticks[1]);
                    populate(my_ncol, second, my_col_chunk_map, my_col_chunk_ticks, my_col_max_chunk_size);

                } else {
                    auto ctype = get_class_name(seed);
                    throw std::runtime_error("instance of unknown class '" + grid_cls + "' returned by 'chunkGrid(<" + ctype + ">)");
                }

                // Choose the dimension that requires pulling out fewer chunks.
                const auto chunks_per_row = my_col_chunk_ticks.size() - 1;
                const auto chunks_per_col = my_row_chunk_ticks.size() - 1;
                my_prefer_rows = chunks_per_row <= chunks_per_col;
            }
        }

        my_require_minimum_cache = opt.require_minimum_cache;
        if (opt.maximum_cache_size.has_value()) {
            my_cache_size_in_bytes = *(opt.maximum_cache_size);
        } else {
            Rcpp::Function fun = my_delayed_env["getAutoBlockSize"];
            Rcpp::NumericVector bsize = fun();
            if (bsize.size() != 1 || bsize[0] < 0) {
                throw std::runtime_error("'getAutoBlockSize()' should return a non-negative number of bytes");
            } else if (bsize[0] > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("integer overflow from the current value of 'getAutoBlockSize()'");
            }
            my_cache_size_in_bytes = bsize[0];
        }
    }

    /**
     * This constructor overload uses the default `Options()`.
     * Again, it should only be called in a serial context, as the (default) construction of **Rcpp** objects may call the R API.
     *
     * @param seed A matrix-like R object.
     */
    UnknownMatrix(Rcpp::RObject seed) : UnknownMatrix(std::move(seed), UnknownMatrixOptions()) {}

private:
    Index_ my_nrow, my_ncol;
    bool my_sparse, my_prefer_rows;

    std::vector<Index_> my_row_chunk_map, my_col_chunk_map;
    std::vector<Index_> my_row_chunk_ticks, my_col_chunk_ticks;

    // To decide how many chunks to store in the cache, we pretend the largest
    // chunk is a good representative. This is a bit suboptimal for irregular
    // chunks but the LruSlabCache class doesn't have a good way of dealing
    // with this right now. The fundamental problem is that variable slabs will
    // either (i) all reach the maximum allocation eventually, if slabs are
    // reused, or (ii) require lots of allocations, if slabs are not reused, or
    // (iii) require manual defragmentation, if slabs are reused in a manner
    // that avoids inflation to the maximum allocation.
    Index_ my_row_max_chunk_size, my_col_max_chunk_size;

    std::size_t my_cache_size_in_bytes;
    bool my_require_minimum_cache;

    Rcpp::RObject my_original_seed;
    Rcpp::Environment my_delayed_env, my_sparse_env;
    Rcpp::Function my_dense_extractor, my_sparse_extractor;

public:
    Index_ nrow() const {
        return my_nrow;
    }

    Index_ ncol() const {
        return my_ncol;
    }

    bool is_sparse() const {
        return my_sparse;
    }

    double is_sparse_proportion() const {
        return static_cast<double>(my_sparse);
    }

    bool prefer_rows() const {
        return my_prefer_rows;
    }

    double prefer_rows_proportion() const {
        return static_cast<double>(my_prefer_rows);
    }

    bool uses_oracle(bool) const {
        return true;
    }

private:
    Index_ max_primary_chunk_length(const bool row) const {
        return (row ? my_row_max_chunk_size : my_col_max_chunk_size);
    }

    Index_ primary_num_chunks(const bool row, const Index_ primary_chunk_length) const {
        auto primary_dim = (row ? my_nrow : my_ncol);
        if (primary_chunk_length == 0) {
            return primary_dim;
        } else {
            return primary_dim / primary_chunk_length;
        }
    }

    Index_ secondary_dim(const bool row) const {
        return (row ? my_ncol : my_nrow);
    }

    const std::vector<Index_>& chunk_ticks(const bool row) const {
        if (row) {
            return my_row_chunk_ticks;
        } else {
            return my_col_chunk_ticks;
        }
    }

    const std::vector<Index_>& chunk_map(const bool row) const {
        if (row) {
            return my_row_chunk_map;
        } else {
            return my_col_chunk_map;
        }
    }

    /********************
     *** Myopic dense ***
     ********************/
private:
    template<
        bool oracle_, 
        template <bool, bool, typename, typename, typename> class FromDense_,
        template <bool, bool, typename, typename, typename, typename> class FromSparse_,
        typename ... Args_
    >
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense_internal(
        const bool row,
        const Index_ non_target_length,
        tatami::MaybeOracle<oracle_, Index_> oracle,
        Args_&& ... args
    ) const {
        std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > output;

        const Index_ max_target_chunk_length = max_primary_chunk_length(row);
        tatami_chunked::SlabCacheStats<Index_> stats(
            /* target length = */ max_target_chunk_length,
            /* non_target_length = */ non_target_length,
            /* target_num_slabs = */ primary_num_chunks(row, max_target_chunk_length),
            /* cache_size_in_bytes = */ my_cache_size_in_bytes,
            /* element_size = */ sizeof(CachedValue_),
            /* require_minimum_cache = */ my_require_minimum_cache
        );

        const auto& map = chunk_map(row);
        const auto& ticks = chunk_ticks(row);
        const bool solo = (stats.max_slabs_in_cache == 0);

#ifdef TATAMI_R_PARALLELIZE_UNKNOWN 
        // This involves some Rcpp initializations, so we lock it just in case.
        auto& mexec = executor();
        mexec.run([&]() -> void {
#endif

        if (!my_sparse) {
            if (solo) {
                output.reset(
                    new FromDense_<true, oracle_, Value_, Index_, CachedValue_>(
                        my_original_seed,
                        my_dense_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        ticks,
                        map,
                        stats
                    )
                );

            } else {
                output.reset(
                    new FromDense_<false, oracle_, Value_, Index_, CachedValue_>(
                        my_original_seed,
                        my_dense_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        ticks,
                        map,
                        stats
                    )
                );
            }

        } else {
            if (solo) {
                output.reset(
                    new FromSparse_<true, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                        my_original_seed,
                        my_sparse_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        max_target_chunk_length,
                        ticks,
                        map,
                        stats
                    )
                );

            } else {
                output.reset(
                    new FromSparse_<false, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                        my_original_seed,
                        my_sparse_extractor,
                        row,
                        std::move(oracle),
                        std::forward<Args_>(args)...,
                        max_target_chunk_length,
                        ticks,
                        map,
                        stats
                    )
                );
            }
        }

#ifdef TATAMI_R_PARALLELIZE_UNKNOWN 
        });
#endif

        return output;
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        const tatami::Options&
    ) const {
        const Index_ non_target_dim = secondary_dim(row);
        return populate_dense_internal<oracle_, DenseFull, DensifiedSparseFull>(
            row,
            non_target_dim,
            std::move(ora),
            non_target_dim
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(
        const  bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        const Index_ block_start,
        const Index_ block_length,
        const tatami::Options&
    ) const {
        return populate_dense_internal<oracle_, DenseBlock, DensifiedSparseBlock>(
            row,
            block_length,
            std::move(ora),
            block_start,
            block_length
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::DenseExtractor<oracle_, Value_, Index_> > populate_dense(
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options&
    ) const {
        const Index_ nidx = indices_ptr->size();
        return populate_dense_internal<oracle_, DenseIndexed, DensifiedSparseIndexed>(
            row,
            nidx,
            std::move(ora),
            std::move(indices_ptr)
        );
    }

public:
    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(
        const bool row,
        const tatami::Options& opt
    ) const {
        return populate_dense<false>(row, false, opt); 
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(
        const bool row,
        const Index_ block_start,
        const Index_ block_length,
        const tatami::Options& opt
    ) const {
        return populate_dense<false>(row, false, block_start, block_length, opt); 
    }

    std::unique_ptr<tatami::MyopicDenseExtractor<Value_, Index_> > dense(
        const bool row,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        return populate_dense<false>(row, false, std::move(indices_ptr), opt); 
    }

    /**********************
     *** Oracular dense ***
     **********************/
public:
    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        const bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        const tatami::Options& opt
    ) const {
        return populate_dense<true>(row, std::move(ora), opt); 
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        const bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        const Index_ block_start,
        const Index_ block_length,
        const tatami::Options& opt
    ) const {
        return populate_dense<true>(row, std::move(ora), block_start, block_length, opt); 
    }

    std::unique_ptr<tatami::OracularDenseExtractor<Value_, Index_> > dense(
        const bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        return populate_dense<true>(row, std::move(ora), std::move(indices_ptr), opt); 
    }

    /*********************
     *** Myopic sparse ***
     *********************/
public:
    template<
        bool oracle_, 
        template<bool, bool, typename, typename, typename, typename> class FromSparse_,
        typename ... Args_
    >
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse_internal(
        const bool row,
        const Index_ non_target_length, 
        tatami::MaybeOracle<oracle_, Index_> oracle, 
        const tatami::Options& opt, 
        Args_&& ... args
    ) const {
        const Index_ max_target_chunk_length = max_primary_chunk_length(row);
        tatami_chunked::SlabCacheStats<Index_> stats(
            /* target_length = */ max_target_chunk_length,
            /* non_target_length = */ non_target_length, 
            /* target_num_slabs = */ primary_num_chunks(row, max_target_chunk_length),
            /* cache_size_in_bytes = */ my_cache_size_in_bytes, 
            /* element_size = */ (opt.sparse_extract_index ? sizeof(CachedIndex_) : 0) + (opt.sparse_extract_value ? sizeof(CachedValue_) : 0),
            /* require_minimum_cache = */ my_require_minimum_cache
        );

        const auto& map = chunk_map(row);
        const auto& ticks = chunk_ticks(row);
        const bool needs_value = opt.sparse_extract_value;
        const bool needs_index = opt.sparse_extract_index;
        const bool solo = stats.max_slabs_in_cache == 0;

        std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > output;

#ifdef TATAMI_R_PARALLELIZE_UNKNOWN 
        // This involves some Rcpp initializations, so we lock it just in case.
        auto& mexec = executor();
        mexec.run([&]() -> void {
#endif

        if (solo) {
            output.reset(
                new FromSparse_<true, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                    my_original_seed,
                    my_sparse_extractor,
                    row,
                    std::move(oracle),
                    std::forward<Args_>(args)...,
                    max_target_chunk_length,
                    ticks,
                    map,
                    stats,
                    needs_value,
                    needs_index
                )
            );

        } else {
            output.reset(
                new FromSparse_<false, oracle_, Value_, Index_, CachedValue_, CachedIndex_>( 
                    my_original_seed,
                    my_sparse_extractor,
                    row,
                    std::move(oracle),
                    std::forward<Args_>(args)...,
                    max_target_chunk_length,
                    ticks,
                    map,
                    stats,
                    needs_value,
                    needs_index
                )
            );
        }

#ifdef TATAMI_R_PARALLELIZE_UNKNOWN 
        });
#endif

        return output;
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        const tatami::Options& opt
    ) const {
        const Index_ non_target_dim = secondary_dim(row);
        return populate_sparse_internal<oracle_, SparseFull>(
            row,
            non_target_dim,
            std::move(ora),
            opt,
            non_target_dim
        ); 
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        const Index_ block_start,
        const Index_ block_length,
        const tatami::Options& opt
    ) const {
        return populate_sparse_internal<oracle_, SparseBlock>(
            row,
            block_length,
            std::move(ora),
            opt,
            block_start,
            block_length
        );
    }

    template<bool oracle_>
    std::unique_ptr<tatami::SparseExtractor<oracle_, Value_, Index_> > populate_sparse(
        const bool row,
        tatami::MaybeOracle<oracle_, Index_> ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        return populate_sparse_internal<oracle_, SparseIndexed>(
            row,
            indices_ptr->size(),
            std::move(ora),
            opt,
            std::move(indices_ptr)
        );
    }

public:
    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(
        const bool row,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::FullSparsifiedWrapper<false, Value_, Index_> >(
                dense(row, opt),
                secondary_dim(row),
                opt
            );
        } else {
            return populate_sparse<false>(row, false, opt); 
        }
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(
        const bool row,
        const Index_ block_start,
        const Index_ block_length,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::BlockSparsifiedWrapper<false, Value_, Index_> >(
                dense(row, block_start, block_length, opt),
                block_start,
                block_length,
                opt
            );
        } else {
            return populate_sparse<false>(row, false, block_start, block_length, opt); 
        }
    }

    std::unique_ptr<tatami::MyopicSparseExtractor<Value_, Index_> > sparse(
        const bool row,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            auto index_copy = indices_ptr;
            return std::make_unique<tatami::IndexSparsifiedWrapper<false, Value_, Index_> >(
                dense(row, std::move(indices_ptr), opt),
                std::move(index_copy),
                opt
            );
        } else {
            return populate_sparse<false>(row, false, std::move(indices_ptr), opt); 
        }
    }

    /**********************
     *** Oracular sparse ***
     **********************/
public:
    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        const bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::FullSparsifiedWrapper<true, Value_, Index_> >(
                dense(row, std::move(ora), opt),
                secondary_dim(row),
                opt
            );
        } else {
            return populate_sparse<true>(row, std::move(ora), opt); 
        }
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        const bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        const Index_ block_start,
        const Index_ block_length,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            return std::make_unique<tatami::BlockSparsifiedWrapper<true, Value_, Index_> >(
                dense(row, std::move(ora), block_start, block_length, opt),
                block_start,
                block_length,
                opt
            );
        } else {
            return populate_sparse<true>(row, std::move(ora), block_start, block_length, opt); 
        }
    }

    std::unique_ptr<tatami::OracularSparseExtractor<Value_, Index_> > sparse(
        const bool row,
        std::shared_ptr<const tatami::Oracle<Index_> > ora,
        tatami::VectorPtr<Index_> indices_ptr,
        const tatami::Options& opt
    ) const {
        if (!my_sparse) {
            auto index_copy = indices_ptr;
            return std::make_unique<tatami::IndexSparsifiedWrapper<true, Value_, Index_> >(
                dense(row, std::move(ora), std::move(indices_ptr), opt),
                std::move(index_copy),
                opt
            );
        } else {
            return populate_sparse<true>(row, std::move(ora), std::move(indices_ptr), opt); 
        }
    }
};

}

#endif
