#ifndef TATAMI_R_UTILS_HPP
#define TATAMI_R_UTILS_HPP

#include "Rcpp.h"
#include <string>
#include <utility>
#include <stdexcept>
#include <memory>

#include "tatami/tatami.hpp"

namespace tatami_r { 

template<typename Input_>
using I = std::remove_reference_t<std::remove_cv_t<Input_> >;

inline std::string make_to_string(const Rcpp::RObject& str) {
    Rcpp::StringVector as_str(str);
    if (as_str.size() != 1) { 
        throw std::runtime_error("input RObject should contain a single string");
    }
    return Rcpp::as<std::string>(as_str[0]);
}

inline Rcpp::RObject get_class_object(const Rcpp::RObject& incoming) {
    if (!incoming.isObject()) {
        throw std::runtime_error("object has no 'class' attribute");
    }
    return incoming.attr("class");
}

inline std::string get_class_name(const Rcpp::RObject& incoming) {
    return make_to_string(get_class_object(incoming));
}

template<typename Index_>
Rcpp::IntegerVector increment_indices(const std::vector<Index_>& indices) {
    // Assume that we've already checked for overflow in length in the UnknownMatrix constructor.
    // We also know that there won't be any overflow in contents as we know that
    // extents fit in both int/Index_ after passing through the UnknownMatrix constructor.
    Rcpp::IntegerVector output(indices.begin(), indices.end());
    for (auto& x : output) {
        ++x;
    }
    return output;
}

template<typename Index_>
Rcpp::IntegerVector consecutive_indices(const Index_ start, const Index_ length) {
    // Assume that we've already checked for overflow in the UnknownMatrix constructor.
    Rcpp::IntegerVector output(length);
    std::iota(output.begin(), output.end(), start + 1);
    return output;
}

}

#endif
