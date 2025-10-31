#ifndef RAIIGRAPH_VECTOR_HPP
#define RAIIGRAPH_VECTOR_HPP

#include "igraph.h"
#include "error.hpp"

#include <algorithm>
#include <initializer_list>
#include <iterator>

/**
 * @file Vector.hpp
 * @brief Wrapper around `igraph_vector_*_t` objects with RAII behavior.
 */

namespace raiigraph {

/**
 * @brief Wrapper around `igraph_vector_*_t` objects with RAII behavior.
 * @tparam Ns_ Structure-based namespace with static methods, internal use only.
 *
 * This class has ownership of the underlying `igraph_vector_*_t` object, handling both its initialization and destruction.
 * Users should only pass instances of this class to **igraph** functions that accept an already-initialized vector.
 * Users should not attempt to destroy the vector manually as this is done automatically when the `Vector` goes out of scope.
 *
 * It is assumed that users have already called `igraph_setup()` before constructing a instance of this class.
 */
template<class Ns_>
class Vector {
private:
    void setup(igraph_int_t size) {
        check_code(Ns_::init(&my_vector, size));
    }

public:
    /**
     * Type of the underlying **igraph** vector.
     */
    typedef typename Ns_::igraph_type igraph_type;

    /**
     * Type of the values inside the vector.
     */
    typedef typename Ns_::value_type value_type;

    /**
     * Type of the reference to values inside the vector.
     */
    typedef value_type& reference;

    /**
     * Type of a const reference to values inside the vector.
     */
    typedef const value_type& const_reference;

    /**
     * Integer type for the size of the vector.
     */
    typedef igraph_int_t size_type;

    /**
     * Integer type for differences in positions within the vector.
     */
    typedef igraph_int_t difference_type;

    /**
     * Iterator for the vector contents. 
     */
    typedef value_type* iterator;

    /**
     * Const iterator for the vector contents. 
     */
    typedef const value_type* const_iterator;

    /**
     * Reverse iterator for the vector contents.
     */
    typedef std::reverse_iterator<iterator> reverse_iterator;

    /**
     * Reverse const iterator for the vector contents.
     */
    typedef std::reverse_iterator<const_iterator> reverse_const_iterator;

public:
    /**
     * Default constructor, creates an initialized vector of length 0.
     */
    Vector() : Vector(0) {}

    /**
     * @param size Size of the vector to create.
     * @param val Value to use to fill the vector.
     */
    Vector(size_type size, const value_type& val = value_type()) {
        setup(size);
        std::fill_n(begin(), size, val);
    }

    /**
     * @param vector An initialized vector to take ownership of.
     */
    Vector(igraph_type&& vector) : my_vector(std::move(vector)) {}

    /**
     * @tparam InputIterator Iterator type that supports forward increments and subtraction.
     * @param first Iterator to the start of a range.
     * @param last Iterator to the end of a range (i.e., past the final element in the range).
     */
    template<typename InputIterator, typename = decltype(*std::declval<InputIterator>())> // use SFINAE to avoid ambiguity with other 2-argument constructors.
    Vector(InputIterator first, InputIterator last) : Vector(last - first) {
        std::copy(first, last, begin());
    }

public:
    /**
     * @param other Vector to be copy-constructed from.
     * This constructor will make a deep copy.
     */
    Vector(const Vector<Ns_>& other) {
        check_code(Ns_::copy(&my_vector, &(other.my_vector)));
    }

    /**
     * @param other Vector to be copy-assigned from.
     * This constructor will make a deep copy.
     */
    Vector<Ns_>& operator=(const Vector<Ns_>& other) {
        if (this != &other) {
            // my_vector should already be initialized before the assignment.
            check_code(Ns_::update(&my_vector, &(other.my_vector)));
        }
        return *this;
    }

    /**
     * @param other Vector to be move-constructed from.
     * This constructor will leave `other` in a valid but unspecified state.
     */
    Vector(Vector<Ns_>&& other) {
        setup(0); // we must leave 'other' in a valid state.
        std::swap(my_vector, other.my_vector);
    }

    /**
     * @param other Vector to be move-assigned from.
     * This constructor will leave `other` in a valid but unspecified state.
     */
    Vector& operator=(Vector<Ns_>&& other) {
        if (this != &other) {
            std::swap(my_vector, other.my_vector); // 'my_vector' should already be initialized, so we're leaving 'other' in a valid state.
        }
        return *this;
    }

    /**
     * Destructor.
     */
    ~Vector() {
        Ns_::destroy(&my_vector);
    }

public:
    /**
     * @return Whether the vector is empty.
     */
    igraph_bool_t empty() const {
        return Ns_::empty(&my_vector);
    }

    /**
     * @return Size of the vector.
     */
    size_type size() const {
        return Ns_::size(&my_vector);
    }

    /**
     * @return Maximum size of this vector.
     */
    constexpr size_type max_size() const { return IGRAPH_INTEGER_MAX; }

    /**
     * @return Capacity of this vector.
     */
    size_type capacity() const {
        return my_vector.stor_end - my_vector.stor_begin;
    }

    /**
     * Clear this vector, leaving it with a size of zero.
     */
    void clear() {
        Ns_::clear(&my_vector);
    }

    /**
     * Resize the vector to the specified `size`.
     * @param size New size of the vector.
     * @param val Value to use to fill the new elements, if `size` is greater than the current size.
     */
    void resize(size_type size, value_type val = value_type()) {
        auto old_size = this->size();
        check_code(Ns_::resize(&my_vector, size));
        if (old_size < size) {
            std::fill_n(begin() + old_size, size - old_size, val);
        }
    }

    /**
     * Reserve the capacity of the vector.
     * @param capacity Capacity of the vector.
     */
    void reserve(size_type capacity) {
        check_code(Ns_::reserve(&my_vector, capacity));
    }

    /**
     * Shrink the capacity of the vector to fit the contents.
     */
    void shrink_to_fit() {
        Ns_::shrink_to_fit(&my_vector);
    }

    /**
     * Add a new element to the end of the vector.
     * @param val Value to be added.
     */
    void push_back(value_type val) {
        check_code(Ns_::push_back(&my_vector, val));
    }

    /**
     * Construct and add a new element to the end of the vector.
     * @param args Arguments for the `value_type` constructor.
     */
    template<typename ... Args>
    void emplace_back(Args&& ... args) {
        // Doesn't really matter for simple types.
        push_back(value_type(std::forward<Args>(args)...));
    }

    /**
     * Remove an element from the end of the vector.
     */
    void pop_back() {
        Ns_::pop_back(&my_vector);
    }

    /**
     * Erase an element from the vector.
     * @param pos Position at which to erase the element.
     */
    iterator erase(iterator pos) {
        Ns_::remove(&my_vector, pos - begin());
        return pos;
    }

    /**
     * Erase an element from the vector.
     * @param first Start of the range of elements to remove.
     * @param last End of the range of elements to remove.
     */
    iterator erase(iterator first, iterator last) {
        auto start = begin();
        Ns_::remove_section(&my_vector, first - start, last - start);
        return first;
    }

    /**
     * Insert an element into the vector.
     * @param pos Position at which to insert the new element.
     * @param val Value to be inserted.
     * @return Iterator to the newly inserted element.
     */
    iterator insert(iterator pos, value_type val) {
        auto delta = pos - begin();
        check_code(Ns_::insert(&my_vector, delta, val));
        return begin() + delta; // recompute it as there might be a reallocation.
    }

    /**
     * Construct and insert an element into the vector.
     * @param pos Position at which to insert the new element.
     * @param args Arguments for the `value_type` constructor.
     * @return Iterator to the newly inserted element.
     */
    template<typename ... Args>
    iterator emplace(iterator pos, Args&& ... args) {
        // Doesn't really matter for simple types.
        return insert(pos, value_type(std::forward<Args>(args)...));
    }

    /**
     * Insert multiple copies of an element into the vector.
     * @param pos Position at which to insert the new elements.
     * @param n Number of values to insert.
     * @param val Value to be inserted.
     * @return Iterator to the first newly inserted element.
     */
    iterator insert(iterator pos, size_type n, value_type val) {
        auto delta = pos - begin();
        auto old_size = size();
        resize(old_size + n);

        auto new_start = begin() + delta; // recompute it as there might be a reallocation.
        std::copy(new_start, begin() + old_size, new_start + n);
        std::fill_n(new_start, n, val);
        return new_start; 
    }

    /**
     * Insert a sequence of elements into the vector.
     * @tparam InputIterator Iterator type that supports forward increments and subtraction.
     * @param pos Position at which to insert the new elements.
     * @param first Iterator to the start of a range.
     * @param last Iterator to the end of a range (i.e., past the final element in the range).
     * @return Iterator to the first newly inserted element.
     */
    template<typename InputIterator, typename = decltype(*std::declval<InputIterator>())>
    iterator insert(iterator pos, InputIterator first, InputIterator last) {
        auto delta = pos - begin();
        auto old_size = size();
        auto n = last - first;
        resize(old_size + n);

        auto new_start = begin() + delta; // recompute it as there might be a reallocation.
        std::copy(new_start, begin() + old_size, new_start + n);
        std::copy(first, last, new_start);
        return new_start; 
    }

public:
    /**
     * @param i Index on the vector.
     * @return Reference to the value at `i`.
     */
    reference operator[](igraph_int_t i) {
        return *(begin() + i);
    }

    /**
     * @param i Index on the vector.
     * @return Const reference to the value at `i`.
     */
    const_reference operator[](igraph_int_t i) const {
        return *(begin() + i);
    }

    /**
     * @return Reference to the last element in the vector.
     */
    reference back() {
        return *(end() - 1);
    }

    /**
     * @return Const reference to the last element in the vector.
     */
    const_reference back() const {
        return *(end() - 1);
    }

    /**
     * @return Reference to the last element in the vector.
     */
    reference front() {
        return *(begin());
    }

    /**
     * @return Const reference to the last element in the vector.
     */
    const_reference front() const {
        return *(begin());
    }

public:
    /**
     * @return Iterator to the start of this vector.
     */
    iterator begin() {
        return my_vector.stor_begin;
    }

    /**
     * @return Iterator to the end of this vector.
     */
    iterator end() {
        return my_vector.end;
    }

    /**
     * @return Const iterator to the start of this vector.
     */
    const_iterator begin() const {
        return cbegin();
    }

    /**
     * @return Const iterator to the end of this vector.
     */
    const_iterator end() const {
        return cend();
    }

    /**
     * @return Const iterator to the start of this vector.
     */
    const_iterator cbegin() const {
        return my_vector.stor_begin;
    }

    /**
     * @return Const iterator to the end of this vector.
     */
    const_iterator cend() const {
        return my_vector.end;
    }

    /**
     * @return Pointer to the start of this vector.
     */
    value_type* data() {
        return my_vector.stor_begin;
    }

    /**
     * @return Const pointer to the start of this vector.
     */
    const value_type* data() const {
        return my_vector.stor_begin;
    }

    /**
     * @return Reverse iterator to the last element of this vector.
     */
    reverse_iterator rbegin() {
        return std::reverse_iterator(end());
    }

    /**
     * @return Reverse iterator to a location before the start of this vector.
     */
    reverse_iterator rend() {
        return std::reverse_iterator(begin());
    }

    /**
     * @return Reverse const iterator to the last element of this vector.
     */
    reverse_const_iterator rbegin() const {
        return std::reverse_iterator(end());
    }

    /**
     * @return Reverse const iterator to a location before the start of this vector.
     */
    reverse_const_iterator rend() const {
        return std::reverse_iterator(begin());
    }

    /**
     * @return Reverse const iterator to the last element of this vector.
     */
    reverse_const_iterator crbegin() const {
        return std::reverse_iterator(cend());
    }

    /**
     * @return Reverse const iterator to a location before the start of this vector.
     */
    reverse_const_iterator crend() const {
        return std::reverse_iterator(cbegin());
    }

public:
    /**
     * @return Pointer to the underlying **igraph** vector object.
     * This is guaranteed to be non-NULL and initialized.
     */
    operator igraph_type*() {
        return &my_vector;
    }

    /**
     * @return Const pointer to the underlying **igraph** vector object.
     * This is guaranteed to be non-NULL and initialized.
     */
    operator const igraph_type*() const {
        return &my_vector;
    }

    /**
     * @return Pointer to the underlying **igraph** vector object.
     * This is guaranteed to be non-NULL and initialized.
     */
    igraph_type* get() {
        return &my_vector;
    }

    /**
     * @return Const pointer to the underlying **igraph** vector object.
     * This is guaranteed to be non-NULL and initialized.
     */
    const igraph_type* get() const {
        return &my_vector;
    }

public:
    /**
     * Swap two vectors, maintaining the validity of existing pointers to each vector and its elements.
     * @param other Vector to be swapped.
     */
    void swap(Vector<Ns_>& other)  {
        // Swapping structures entirely to ensure that iterators and pointers
        // remain valid; looks like igraph_vector_swap does the same.
        std::swap(my_vector, other.my_vector);
    }

private:
    igraph_type my_vector;
};

/**
 * @cond
 */
namespace internal {

struct Integer {
    typedef igraph_int_t value_type;
    typedef igraph_vector_int_t igraph_type;

#define RAIIGRAPH_VECTOR_SUFFIX _int
#include "fragments/vector.hpp"
#undef RAIIGRAPH_VECTOR_SUFFIX
};

struct Real {
    typedef igraph_real_t value_type;
    typedef igraph_vector_t igraph_type;

#define RAIIGRAPH_VECTOR_SUFFIX
#include "fragments/vector.hpp"
#undef RAIIGRAPH_VECTOR_SUFFIX
};

struct Bool {
    typedef igraph_bool_t value_type;
    typedef igraph_vector_bool_t igraph_type;

#define RAIIGRAPH_VECTOR_SUFFIX _bool
#include "fragments/vector.hpp"
#undef RAIIGRAPH_VECTOR_SUFFIX
};

}
/**
 * @endcond
 */

/**
 * Vector of **igraph** integers.
 */
typedef Vector<internal::Integer> IntVector;

/**
 * @cond
 */
// For back-compatibility.
typedef IntVector IntegerVector;
/**
 * @endcond
 */

/**
 * Vector of **igraph** reals.
 */
typedef Vector<internal::Real> RealVector;

/**
 * Vector of **igraph** booleans.
 */
typedef Vector<internal::Bool> BoolVector;

}


#endif
