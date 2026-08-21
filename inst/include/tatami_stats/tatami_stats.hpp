#ifndef TATAMI_TATAMI_STATS_HPP
#define TATAMI_TATAMI_STATS_HPP

#include "count.hpp"
#include "group_median.hpp"
#include "group_sum.hpp"
#include "group_variance.hpp"
#include "median.hpp"
#include "quantile.hpp"
#include "range.hpp"
#include "sum.hpp"
#include "utils.hpp"
#include "variance.hpp"

#include "skip_nan/rss.hpp"

/**
 * @file tatami_stats.hpp
 * @brief Umbrella header for the **tatami_stats** library.
 */

/**
 * @namespace tatami_stats
 * @brief Functions to compute statistics from a `tatami::Matrix`.
 */
namespace tatami_stats {

/**
 * @namespace tatami_stats::skip_nan
 * @brief Compute statistics from a `tatami::Matrix` while skipping NaNs.
 */
namespace skip_nan {}

}

#endif
