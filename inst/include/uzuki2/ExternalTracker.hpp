#ifndef UZUKI2_EXTERNAL_TRACKER_HPP
#define UZUKI2_EXTERNAL_TRACKER_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>

namespace uzuki2 {

template<class CustomExternals_>
class ExternalTracker {
public:
    ExternalTracker(CustomExternals_ e) : my_getter(std::move(e)) {}

    void* get(size_t i) {
        my_indices.push_back(i);
        return my_getter.get(i);
    };

    size_t size() const {
        return my_getter.size();
    }

private:
    CustomExternals_ my_getter;
    std::vector<size_t> my_indices;

public:
    void validate() {
        // Checking that the external indices match up.
        size_t n = my_indices.size();
        if (n != my_getter.size()) {
            throw std::runtime_error("fewer instances of type \"external\" than expected from 'ext'");
        }

        std::sort(my_indices.begin(), my_indices.end());
        for (size_t i = 0; i < n; ++i) {
            if (i != my_indices[i]) {
                throw std::runtime_error("set of \"index\" values for type \"external\" should be consecutive starting from zero");
            }
        }
    }
};

}

#endif
