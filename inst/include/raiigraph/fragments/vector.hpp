#define RAIIGRAPH_VECTOR_FUNCTION1(suffix, action) igraph_vector##suffix##_##action
#define RAIIGRAPH_VECTOR_FUNCTION2(suffix, action) RAIIGRAPH_VECTOR_FUNCTION1(suffix, action)
#define RAIIGRAPH_VECTOR_FUNCTION(action) RAIIGRAPH_VECTOR_FUNCTION2(RAIIGRAPH_VECTOR_SUFFIX, action)

static auto init(igraph_type* ptr, igraph_int_t size) {
    return RAIIGRAPH_VECTOR_FUNCTION(init)(ptr, size);
}

static auto copy(igraph_type* ptr, const igraph_type* other) {
    return RAIIGRAPH_VECTOR_FUNCTION(init_copy)(ptr, other);
}

static auto init_array(igraph_type* ptr, const value_type* start, igraph_int_t size) {
    return RAIIGRAPH_VECTOR_FUNCTION(init_array)(ptr, start, size);
}

static auto update(igraph_type* ptr, const igraph_type* other) {
    return RAIIGRAPH_VECTOR_FUNCTION(update)(ptr, other);
}

static auto destroy(igraph_type* ptr) {
    return RAIIGRAPH_VECTOR_FUNCTION(destroy)(ptr);
}

static auto empty(const igraph_type* ptr) {
    return RAIIGRAPH_VECTOR_FUNCTION(empty)(ptr);
}

static auto size(const igraph_type* ptr) {
    return RAIIGRAPH_VECTOR_FUNCTION(size)(ptr);
}

static auto clear(igraph_type* ptr) {
    return RAIIGRAPH_VECTOR_FUNCTION(clear)(ptr);
}

static auto resize(igraph_type* ptr, igraph_int_t size) {
    return RAIIGRAPH_VECTOR_FUNCTION(resize)(ptr, size);
}

static auto reserve(igraph_type* ptr, igraph_int_t size) {
    return RAIIGRAPH_VECTOR_FUNCTION(reserve)(ptr, size);
}

static void shrink_to_fit(igraph_type* ptr) {
    RAIIGRAPH_VECTOR_FUNCTION(resize_min)(ptr);
}

static auto insert(igraph_type* ptr, igraph_int_t pos, value_type x) {
    return RAIIGRAPH_VECTOR_FUNCTION(insert)(ptr, pos, x);
}

static void remove(igraph_type* ptr, igraph_int_t pos) {
    RAIIGRAPH_VECTOR_FUNCTION(remove)(ptr, pos);
}

static void remove_section(igraph_type* ptr, igraph_int_t first, igraph_int_t last) {
    RAIIGRAPH_VECTOR_FUNCTION(remove_section)(ptr, first, last);
}

static auto push_back(igraph_type* ptr, value_type x) {
    return RAIIGRAPH_VECTOR_FUNCTION(push_back)(ptr, x);
}

static void pop_back(igraph_type* ptr) {
    RAIIGRAPH_VECTOR_FUNCTION(pop_back)(ptr);
}

#undef RAIIGRAPH_VECTOR_FUNCTION1
#undef RAIIGRAPH_VECTOR_FUNCTION2
#undef RAIIGRAPH_VECTOR_FUNCTION
