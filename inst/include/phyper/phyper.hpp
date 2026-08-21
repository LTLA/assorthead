#ifndef PHYPER_PHYPER_HPP
#define PHYPER_PHYPER_HPP

#include <cmath>
#include <limits>

/**
 * @file phyper.hpp
 * @brief Compute hypergeometric tail probabilities.
 */

/**
 * @namespace phyper
 * @brief Compute hypergeometric tail probabilities.
 */
namespace phyper {

/**
 * @brief Options for `compute()`.
 */
struct Options {
    /**
     * Whether to report log-probabilities, which avoids underflow for very small values.
     */
    bool log = false;

    /**
     * Whether to report the upper tail, including the probability mass of the observed number of drawn white balls.
     * This allows the tail probability to be directly used as the p-value for testing enrichment.
     * If `false`, the lower tail is returned, again including the probability mass of `drawn_inside`.
     */
    bool upper_tail = true;
};

/**
 * @cond
 */
namespace internal {

template<typename Count_>
long double lfactorial(const Count_ x) {
    // Computing it exactly for small numbers, to avoid unnecessarily
    // large relative inaccuracy from the approximation. Threshold of
    // 12 is chosen more-or-less arbitrarily... but 12! is the largest
    // value that can be represented by a 32-bit int, if that helps.
    switch(x) {
        case 0: case 1: return 0;
        case 2: return std::log(2.0); 
        case 3: return std::log(6.0); 
        case 4: return std::log(24.0); 
        case 5: return std::log(120.0); 
        case 6: return std::log(720.0); 
        case 7: return std::log(5040.0); 
        case 8: return std::log(40320.0); 
        case 9: return std::log(362880.0); 
        case 10: return std::log(3628800.0); 
        case 11: return std::log(39916800.0); 
        case 12: return std::log(479001600.0); 
    }

    // For large numbers, using Ramanujan's approximation rather than R's complicated thing. 
    // Check out https://www.johndcook.com/blog/2012/09/25/ramanujans-factorial-approximation/.
    long double y = x;
    return 1.0/6.0 * std::log(y * (1 + 4 * y * (1 + 2 * y)) + 1.0/30.0) + y * std::log(y) - y + 0.5 * std::log(3.14159265358979323846);
}

}
/**
 * @endcond
 */

/**
 * Compute the tail probabilities for the hypergeometric distribution.
 * It is intended for use in quantifying gene set enrichment in a list of top marker genes.
 * The "successes" are the genes in the set, the "failures" are all other genes, and the drawing process typically involves selecting the top markers;
 * our aim is to compute the p-value for enrichment of genes in the set among the top markers.
 *
 * @tparam Count_ Integer type of the number of genes.
 *
 * @param drawn_inside Number of genes in the gene set that were selected as markers.
 * @param num_inside Total number of genes in the gene set.
 * @param num_outside Total number of genes outside the gene set.
 * @param num_drawn Number of genes that were selected as top markers.
 * @param options Further options for the calculation.
 *
 * @return Probability of randomly selecting at least `drawn_inside` genes from the gene set, if `Options::upper_tail = true`.
 * Otherwise, the probability of randomly selecting no more than `drawn_inside` genes from the gene set. 
 * These probabilities are log-transformed when `Options::log = true`.
 */
template<typename Count_>
double compute(Count_ drawn_inside, Count_ num_inside, Count_ num_outside, const Count_ num_drawn, const Options& options) {
    // Handling all the edge cases.
    if (options.upper_tail) {
        if (drawn_inside <= 0 || (num_drawn >= num_outside && drawn_inside <= num_drawn - num_outside)) {
            return (options.log ? 0 : 1);
        }
        if (drawn_inside > num_drawn || drawn_inside > num_inside) {
            return (options.log ? -std::numeric_limits<double>::infinity() : 0);
        }
    } else {
        if (drawn_inside < 0 || (num_drawn >= num_outside && drawn_inside < num_drawn - num_outside)) {
            return (options.log ? -std::numeric_limits<double>::infinity() : 0);
        }
        if (drawn_inside >= num_drawn || drawn_inside >= num_inside) {
            return (options.log ? 0 : 1);
        }
    }

    if (std::numeric_limits<Count_>::max() - num_inside < num_outside) {
        throw std::runtime_error("sum of 'num_inside' and 'num_outside' results in integer overflow");
    }
    const Count_ num_total = num_outside + num_inside;

    // Subtracting 1 to include the probably mass of 'drawn_inside' in the upper tail calculations.
    if (options.upper_tail) {
        --drawn_inside;
    }

    // We flip the problem to ensure that we're always computing the smaller tail for accuracy.
    // (Smaller in terms of the cumulative probability, and usually - but not necessarily - the number of iterations.)
    // If that's the tail that we wanted, then great; we can compute it directly without worrying about loss of precision from '1 - [some larger tail]'.
    // If it's not the tail we wanted, then we compute '1 - [this smaller tail]' and we don't have to worry about accumulation of errors from summation towards 1.
    // In addition, the smaller tail is usually faster to compute but this is a secondary effect.
    bool needs_upper = options.upper_tail;
    if (static_cast<double>(drawn_inside) * static_cast<double>(num_total) > static_cast<double>(num_drawn) * static_cast<double>(num_inside)) {
        std::swap(num_inside, num_outside);
        drawn_inside = num_drawn - drawn_inside - 1; // Guaranteed to be non-negative due to edge case protection; we already decremented drawn_inside when upper_tail = true.
        needs_upper = !needs_upper;
    }

    /*
     * Computing the cumulative sum after factorizing out the probability mass at drawn_inside.
     * This allows us to do one pass from drawn_inside to 0 to compute the probability.
     * We use long doubles to mitigate the loss of precision on these cumulative operations.
     * 
     * We can check the accuracy of our calculations with:
     * > sum(choose(num_inside, 0:drawn_inside) * choose(num_outside, num_drawn - 0:drawn_inside)) / max(choose(num_inside, num_drawn - num_outside), choose(num_outside, num_drawn)) - 1
     *
     * We start from the probability mass at drawn_inside observations, and we work our way downwards.
     * This avoids problems with floating point overflow when computing the cumulative product.
     */
    Count_ denom1a = drawn_inside, denom1b = num_inside - denom1a;
    Count_ denom2a = num_drawn - drawn_inside, denom2b = num_outside - denom2a; // be careful with the second subtraction to avoid underflow for unsigned Count_.
    const long double log_probability = 
        + internal::lfactorial(num_inside) - internal::lfactorial(denom1a) - internal::lfactorial(denom1b) // lchoose(num_inside, num_inside) 
        + internal::lfactorial(num_outside) - internal::lfactorial(denom2a) - internal::lfactorial(denom2b) // lchoose(num_outside, num_drawn - drawn_inside) 
        - internal::lfactorial(num_total) + internal::lfactorial(num_drawn) + internal::lfactorial(num_total - num_drawn); // -lchoose(num_total, num_drawn)

    long double cumulative = 0; // will add 1 via log1p.
    long double partial_probability = 1;

    for (Count_ k = drawn_inside; k > 0; --k) {
        ++denom1b;
        ++denom2a;

        long double mult = (static_cast<long double>(denom1a) * static_cast<long double>(denom2b)) / (static_cast<long double>(denom1b) * static_cast<long double>(denom2a));
        partial_probability *= mult;
        if (partial_probability == 0) { // underflow to zero, no point continuing...
            break;
        }
        cumulative += partial_probability;

        --denom1a;
        --denom2b;
    }

    const long double log_cumulative = std::log1p(cumulative) + log_probability;
    if (!needs_upper) {
        if (options.log) {
            return log_cumulative;
        } else {
            return std::exp(log_cumulative);
        }
    }

    if (options.log) {
        // Basically, we want to compute 'log(1 - exp(log_cumulative))', but need to be a bit smart about where the subtraction from 1 occurs.
        // The logic is derived from https://github.com/SurajGupta/r-source/blob/master/src/nmath/dpq.h, specifically:
        // - if 'log_cumulative' is close to zero, 'exp(log_cumulative)' will be close to 1, and thus the precision of 'expm1' is more important.
        // - if 'log_cumulative' is large and negative, 'exp(log_cumulative)' will be close to zero, and thus the precision of 'log1p' is more important.
        if (log_cumulative > -std::log(2)) {
            const auto p = -std::expm1(log_cumulative);
            return (p > 0 ? std::log(p) : -std::numeric_limits<double>::infinity());
        } else {
            const auto p = -std::exp(log_cumulative);
            return (p > -1 ? std::log1p(p) : -std::numeric_limits<double>::infinity());
        }
    } else {
        return -std::expm1(log_cumulative);
    }
}

}

#endif
