#ifndef SCRAN_MARKERS_UTILS_HPP
#define SCRAN_MARKERS_UTILS_HPP

namespace scran_markers {

template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(Input_ x) {
    return x;
}

}

#endif
