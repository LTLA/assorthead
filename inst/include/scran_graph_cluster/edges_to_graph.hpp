#ifndef SCRAN_GRAPH_CLUSTER_EDGES_TO_GRAPH_HPP
#define SCRAN_GRAPH_CLUSTER_EDGES_TO_GRAPH_HPP

#include <cstddef>

#include "raiigraph/raiigraph.hpp"
#include "sanisizer/sanisizer.hpp"

#include "igraph.h"

/**
 * @file edges_to_graph.hpp
 * @brief Convert a list of edges to a graph.
 */

namespace scran_graph_cluster {

/**
 * Create an `raiigraph:Graph` object from an array of edges.
 * This assumes that `igraph_setup()` or `raiigraph::initialize()` has already been called.
 *
 * @tparam Vertex_ Integer type of the vertex IDs.
 *
 * @param double_edges The number of edges multiplied by two.
 * @param[in] edges Pointer to an array of length `double_edges`.
 * `edges[2*i]` and `edges[2*i+1]` define the vertices for edge `i`.
 * For directed graphs, the edge starts from the first vertex and ends at the second vertex.
 * @param num_vertices Number of vertices in the graph.
 * @param directed Whether the graph is directed.
 * This should be one of `IGRAPH_DIRECTED` or `IGRAPH_UNDIRECTED`.
 *
 * @return A graph created from `edges`.
 */
template<typename Vertex_>
raiigraph::Graph edges_to_graph(const std::size_t double_edges, const Vertex_* const edges, const std::size_t num_vertices, const igraph_bool_t directed) {
    if constexpr(std::is_same<Vertex_, igraph_int_t>::value) {
        const auto edge_view = igraph_vector_int_view(edges, sanisizer::cast<igraph_int_t>(double_edges));
        return raiigraph::Graph(&edge_view, num_vertices, directed);
    } else {
        auto tmp = sanisizer::create<raiigraph::IntegerVector>(double_edges);
        auto& payload = *(tmp.get());
        for (std::size_t x = 0; x < double_edges; ++x) {
            VECTOR(payload)[x] = edges[x];
        }
        return raiigraph::Graph(tmp, num_vertices, directed);
    }
}

}

#endif
