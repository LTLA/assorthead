#define RAIIGRAPH_MATRIX_FUNCTION1(suffix, action) igraph_matrix##suffix##_##action
#define RAIIGRAPH_MATRIX_FUNCTION2(suffix, action) RAIIGRAPH_MATRIX_FUNCTION1(suffix, action)
#define RAIIGRAPH_MATRIX_FUNCTION(action) RAIIGRAPH_MATRIX_FUNCTION2(RAIIGRAPH_MATRIX_SUFFIX, action)

static auto init(igraph_type* ptr, igraph_int_t nr, igraph_int_t nc) {
    return RAIIGRAPH_MATRIX_FUNCTION(init)(ptr, nr, nc);
}

static auto copy(igraph_type* ptr, const igraph_type* other) {
    return RAIIGRAPH_MATRIX_FUNCTION(init_copy)(ptr, other);
}

static auto update(igraph_type* ptr, const igraph_type* other) {
    return RAIIGRAPH_MATRIX_FUNCTION(update)(ptr, other);
}

static auto destroy(igraph_type* ptr) {
    return RAIIGRAPH_MATRIX_FUNCTION(destroy)(ptr);
}

static auto empty(const igraph_type* ptr) {
    return RAIIGRAPH_MATRIX_FUNCTION(empty)(ptr);
}

static auto size(const igraph_type* ptr) {
    return RAIIGRAPH_MATRIX_FUNCTION(size)(ptr);
}

static auto resize(igraph_type* ptr, igraph_int_t nr, igraph_int_t nc) {
    return RAIIGRAPH_MATRIX_FUNCTION(resize)(ptr, nr, nc);
}

template<typename Vector_>
static auto get_row(const igraph_type* ptr, Vector_* vec, igraph_int_t i) {
    return RAIIGRAPH_MATRIX_FUNCTION(get_row)(ptr, vec, i);
}

template<typename Vector_>
static auto get_col(const igraph_type* ptr, Vector_* vec, igraph_int_t i) {
    return RAIIGRAPH_MATRIX_FUNCTION(get_col)(ptr, vec, i);
}

static void shrink_to_fit(igraph_type* ptr) {
    RAIIGRAPH_MATRIX_FUNCTION(resize_min)(ptr);
}

#undef RAIIGRAPH_MATRIX_FUNCTION1
#undef RAIIGRAPH_MATRIX_FUNCTION2
#undef RAIIGRAPH_MATRIX_FUNCTION
