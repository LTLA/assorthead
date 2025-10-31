#ifndef RAIIGRAPH_GRAPH_HPP 
#define RAIIGRAPH_GRAPH_HPP

#include "igraph.h"
#include "Vector.hpp"
#include "error.hpp"

/**
 * @file Graph.hpp
 * @brief Wrapper around `igraph_t` objects with RAII behavior.
 */

namespace raiigraph {

/**
 * @brief Wrapper around `igraph_t` objects with RAII behavior. 
 *
 * This class has ownership of the underlying `igraph_t` object, handling both its initialization and destruction.
 * Users should only pass instances of this class to **igraph** functions that accept an already-initialized graph.
 * Users should not attempt to destroy the graph manually as this is done automatically when the `Graph` goes out of scope.
 *
 * It is assumed that users have already called `igraph_setup()` before constructing a instance of this class.
 */
class Graph {
private:
    void setup(igraph_int_t num_vertices, igraph_bool_t directed) {
        check_code(igraph_empty(&my_graph, num_vertices, directed));
    }

public:
    /**
     * Create an empty graph, i.e., with no edges.
     *
     * @param num_vertices Number of vertices.
     * @param directed Whether the graph is directed.
     */
    Graph(igraph_int_t num_vertices = 0, igraph_bool_t directed = false) {
        setup(num_vertices, directed);
    }

    /**
     * @param edges Edges between vertices, stored as row-major matrix with two columns.
     * Each row corresponds to an edge and contains its connected vertices.
     * For example, the `i`-th edge is defined from the first vertex at `edges[2 * i]` to the second vertex at `edges[2 * i + 1]`.
     * @param num_vertices Number of vertices in the graph.
     * This should be greater than the largest index in `edges`.
     * @param directed Whether the graph is directed.
     */
    Graph(const IntVector& edges, igraph_int_t num_vertices, igraph_bool_t directed) : Graph(edges.get(), num_vertices, directed) {} 

    /**
     * @param edges Edges between vertices, stored as a vector of non-negative vertex indices of length equal to twice the number of edges.
     * The `i`-th edge is defined from the first vertex at `edges[2 * i]` to the second vertex at `edges[2 * i + 1]`.
     * @param num_vertices Number of vertices in the graph.
     * This should be greater than the largest index in `edges`.
     * @param directed Whether the graph is directed.
     */
    Graph(const igraph_vector_int_t* edges, igraph_int_t num_vertices, igraph_bool_t directed) { 
        check_code(igraph_create(&my_graph, edges, num_vertices, directed));
    }

    /**
     * @param graph An initialized graph to take ownership of.
     */
    Graph(igraph_t&& graph) : my_graph(std::move(graph)) {}

public:
    /**
     * @param other Graph to be copy-constructed from.
     */
    Graph(const Graph& other) {
        check_code(igraph_copy(&my_graph, &(other.my_graph)));
    }

    /**
     * @param other Graph to be copy-assigned from.
     */
    Graph& operator=(const Graph& other) {
        if (this != &other) {
            check_code(igraph_copy(&my_graph, &(other.my_graph)));
        }
        return *this;
    }

    /**
     * @param other Graph to be move-constructed from.
     * This constructor will leave `other` in a valid but unspecified state.
     */
    Graph(Graph&& other) {
        setup(0, false);
        std::swap(my_graph, other.my_graph);
    }    

    /**
     * @param other Graph to be move-assigned from.
     * This constructor will leave `other` in a valid but unspecified state.
     */
    Graph& operator=(Graph&& other) {
        if (this != &other) {
            std::swap(my_graph, other.my_graph);
        }
        return *this;
    }

    /**
     * Destructor.
     */
    ~Graph() {
        igraph_destroy(&my_graph);
    }

public:
    /**
     * @return Number of vertices in the graph.
     */
    igraph_int_t vcount() const {
        return igraph_vcount(&my_graph);
    }

    /**
     * @return Number of edges in the graph.
     */
    igraph_int_t ecount() const {
        return igraph_ecount(&my_graph);
    }

    /**
     * @param by_col Whether to return the edges in a column-major array.
     * @return Vector containing a matrix with two columns.
     * Each row corresponds to an edge and contains its connected vertices.
     * If `by_col = false`, this is the same as the sequence of edges used in the constructor.
     */
    IntVector get_edgelist(igraph_bool_t by_col = false) const {
        IntVector out(ecount());
        check_code(igraph_get_edgelist(&my_graph, out.get(), by_col));
        return out;
    }

public:
    /**
     * @return Whether the graph is directed.
     */
    igraph_bool_t is_directed() const {
        return igraph_is_directed(&my_graph);
    }

    /**
     * @param mode The connectedness mode, for directed graphs.
     * This can be either `IGRAPH_WEAK` or `IGRAPH_STRONG`.
     * Ignored for undirected graphs.
     * @return Whether the graph is (weakly or strongly) connected.
     */
    igraph_bool_t is_connected(igraph_connectedness_t mode = IGRAPH_WEAK) const {
        igraph_bool_t res;
        check_code(igraph_is_connected(&my_graph, &res, mode));
        return res;
    }

    /**
     * @param directed Whether to consider the directions of edges.
     * This can be either `IGRAPH_UNDIRECTED` or `IGRAPH_DIRECTED`.
     * Ignored for undirected graphs.
     * @return Whether the graph is simple, i.e., no loops or multiple edges.
     */
    igraph_bool_t is_simple(igraph_bool_t directed) const {
        igraph_bool_t res;
        check_code(igraph_is_simple(&my_graph, &res, directed));
        return res;
    }

    /**
     * @return Whether the graph contains a loop edge, i.e., from a vertex to itself.
     */
    igraph_bool_t has_loop() const {
        igraph_bool_t res;
        check_code(igraph_has_loop(&my_graph, &res));
        return res;
    }

    /**
     * @return Whether the graph contains multiple edges between the same pair of vertices.
     */
    igraph_bool_t has_multiple() const {
        igraph_bool_t res;
        check_code(igraph_has_multiple(&my_graph, &res));
        return res;
    }

    /**
     * @param loops Whether to consider directed self-loops to be mutual.
     * @return Whether the directed graph contains mutual edges, i.e., an edge from A to B and also an edge from B back to A.
     */
    igraph_bool_t has_mutual(igraph_bool_t loops = false) const {
        igraph_bool_t res;
        check_code(igraph_has_mutual(&my_graph, &res, loops));
        return res;
    }

    /**
     * @param mode Whether to test for an out-tree, an in-tree or to ignore edge directions, for directed graphs.
     * The respective possible values are `IGRAPH_OUT`, `IGRAPH_IN` and `IGRAPH_ALL`.
     * Ignored for undirected graphs.
     * @return Whether the graph is a tree, i.e., connected with no cycles.
     */
    bool is_tree(igraph_neimode_t mode = IGRAPH_ALL) const {
        igraph_bool_t res;
        check_code(igraph_is_tree(&my_graph, &res, NULL, mode));
        return res;
    }

    /**
     * @param mode Whether to test for an out-tree, an in-tree or to ignore edge directions, for directed graphs; see `is_tree()`.
     * @return Whether a graph is a forest, i.e., all connected components are trees.
     */
    bool is_forest(igraph_neimode_t mode = IGRAPH_ALL) const {
        igraph_bool_t res;
        check_code(igraph_is_forest(&my_graph, &res, NULL, mode));
        return res;
    }

    /**
     * @return Whether the graph is a directed acyclic graph.
     */
    bool is_dag() const {
        igraph_bool_t res;
        check_code(igraph_is_dag(&my_graph, &res));
        return res;
    }

    /**
     * @return Whether the graph is an acyclic graph.
     */
    bool is_acyclic() const {
        igraph_bool_t res;
        check_code(igraph_is_acyclic(&my_graph, &res));
        return res;
    }

public:
    /**
     * @return Pointer to the underlying **igraph** graph object.
     * This is guaranteed to be non-NULL and initialized.
     */
    operator igraph_t*() {
        return &my_graph;
    }

    /**
     * @return Const pointer to the underlying **igraph** graph object.
     * This is guaranteed to be non-NULL and initialized.
     */
    operator const igraph_t*() const {
        return &my_graph;
    }

    /**
     * @return Pointer to the underlying **igraph** graph.
     * This is guaranteed to be non-NULL and initialized.
     */
    igraph_t* get() {
        return &my_graph;
    }

    /**
     * @return Pointer to the underlying **igraph** graph.
     * This is guaranteed to be non-NULL and initialized.
     */
    const igraph_t* get() const {
        return &my_graph;
    }

private:
    igraph_t my_graph;
};

}

#endif
