#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <functional>
#include <chrono>
#include <iterator>
#include <vector>
#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"
#include "andres/partition.hxx"

namespace py = pybind11;

const size_t maxInt = std::numeric_limits<size_t>::max();
const double maxDouble = std::numeric_limits<double>::max();

// hash for an unordered set of tuples
struct Hash {
    size_t operator()(const std::tuple<size_t, size_t>& tuple) const {
        const auto& [x, y] = tuple;
        size_t hash1 = std::hash<size_t>{}(x);
        size_t hash2 = std::hash<size_t>{}(y);
        return hash1 ^ (hash2 << 1); // Simple XOR hash combiner
    }
};


// class copied from andres greedy-additive
class DynamicGraph
{
public:
    DynamicGraph(size_t n) :
        vertices_(n)
    {}

    bool edgeExists(size_t a, size_t b) const
    {
        return !vertices_[a].empty() && vertices_[a].find(b) != vertices_[a].end();
    }

    std::map<size_t, double> const& getAdjacentVertices(size_t v) const
    {
        return vertices_[v];
    }

    size_t getEdgeWeight(size_t a, size_t b) const
    {
        return vertices_[a].at(b);
    }

    void removeVertex(size_t v)
    {
        for (auto& p : vertices_[v])
            vertices_[p.first].erase(v);

        vertices_[v].clear();
    }

    void updateEdgeWeight(size_t a, size_t b, double w)
    {
        vertices_[a][b] += w;
        vertices_[b][a] += w;
        if (a < b) {
            edges_.emplace(a, b);
        } else {
            edges_.emplace(b, a);
        }
    }

    std::unordered_set<std::tuple<size_t, size_t>, Hash> getEdges() {
        return edges_;
    }

private:
    std::vector<std::map<size_t, double>> vertices_;
    std::unordered_set<std::tuple<size_t, size_t>, Hash> edges_;
};

// struct copied from andres greedy-additive
struct Edge
{
    Edge(size_t _a, size_t _b, double _w)
    {
        if (_a > _b)
            std::swap(_a, _b);

        a = _a;
        b = _b;

        w = _w;
    }

    size_t a;
    size_t b;
    size_t edition;
    double w;

    bool operator <(Edge const& other) const
    {
        return w < other.w;
    }
};

size_t find(std::vector<size_t>& parent, size_t i){
    if (parent[i] == i) {
        return i;
    }
    return find(parent, parent[i]);
}

void unionSet(std::vector<size_t>& parent, std::vector<size_t>& rank, size_t x, size_t y) {
    size_t xroot = find(parent, x);
    size_t yroot = find(parent, y);

    // Attach smaller rank tree under root of high rank
    // tree (Union by Rank)
    if (rank[xroot] < rank[yroot]) {
        parent[xroot] = yroot;
    }
    else if (rank[xroot] > rank[yroot]) {
        parent[yroot] = xroot;
    }
    // If ranks are same, then make one as root and
    // increment its rank by one
    else {
        parent[yroot] = xroot;
        rank[xroot]++;
    }
}

// source: https://www.geeksforgeeks.org/boruvkas-algorithm-greedy-algo-9/
andres::graph::Graph<> boruvkaMST(size_t numVertices, DynamicGraph graph) {
    py::print("num vertices:", numVertices);

    std::vector<size_t> parent(numVertices);
    andres::graph::Graph<> MSTgraph;
    MSTgraph.insertVertices(numVertices);

    // An array to store index of the best edge of
    // subset. It store [u,v,w] for each component
    std::vector<size_t> rank(numVertices);
    std::vector<std::vector<size_t>> best(numVertices, std::vector<size_t>(3, -1));

    // Initially there are V different trees.
    // Finally there will be one tree that will be MST
    size_t numTrees = numVertices;
    size_t MSTweight = 0;

    // Create V subsets with single elements
    for (size_t node = 0; node < numVertices; node++) {
        parent[node] = node;
        rank[node] = 0;
    }

    // Keep combining components (or sets) until all
    // components are not combined into single MST
    bool changing = true;
    while (changing) {
        changing = false;

        // Traverse through all edges and update
        // best of every component
        for (const auto [u, v] : graph.getEdges()) {
            if ( u >= numVertices || v >= numVertices)
                throw std::runtime_error("node has invalid value: " + std::to_string(u) + ", " + std::to_string(v));

            size_t w = graph.getEdgeWeight(u, v);

            // ignore negative edges
            if (w < 0)
                continue;

            size_t set1 = find(parent, u),
                   set2 = find(parent, v);

            // If two corners of current edge belong to
            // same set, ignore current edge. Else check
            // if current edge is closer to previous
            // best edges of set1 and set2
            if (set1 != set2) {
                if (best[set1][2] == -1 || best[set1][2] < w) {
                    best[set1] = { u, v, w };
                }
                if (best[set2][2] == -1 || best[set2][2] < w) {
                    best[set2] = { u, v, w };
                }
            }
        }

        // Consider the above picked best edges and
        // add them to MST
        for (size_t node = 0; node < numVertices; node++) {

            // Check if best for current set exists
            if (best[node][2] != -1) {
                size_t u = best[node][0],
                       v = best[node][1],
                       w = best[node][2];
                size_t set1 = find(parent, u),
                       set2 = find(parent, v);
                if (set1 != set2) {
                    changing = true;
                    MSTweight += w;
                    unionSet(parent, rank, set1, set2);
                    if (u >= numVertices || v >= numVertices)
                        throw std::runtime_error("node has invalid value: " + std::to_string(u) + ", " + std::to_string(v));
                    MSTgraph.insertEdge(u, v);
                    numTrees--;
                }
            }
        }
        for (size_t node = 0; node < numVertices; node++) {

            // reset best array
            best[node][2] = -1;
        }
    }

    py::print("total weight", MSTweight);
    py::print("total edges", MSTgraph.numberOfEdges());

    return MSTgraph;
}

std::tuple<std::set<size_t>, float> spanningTreeEdgeContraction(const size_t numVertices, const std::vector<std::tuple<size_t, size_t, double>>& edges) {

    // constructing the graph from parameters
    andres::graph::Graph<> graph;

    graph.insertVertices(numVertices);

    std::vector<double> edge_values(edges.size());

    size_t i = 0;
    for (const auto& [node1, node2, weight] : edges) {
        graph.insertEdge(node1, node2);
        edge_values[i++] = weight;
    };

    std::vector<char> edge_labels(graph.numberOfEdges());


    // declaring objects required for the algorithm
    DynamicGraph original_graph_cp(graph.numberOfVertices());

    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    py::print("constructing dynamic graph");
    // constructing the dynamic graph that will be altered
    for (size_t i = 0; i < graph.numberOfEdges(); ++i)
    {
        auto a = graph.vertexOfEdge(i, 0);
        auto b = graph.vertexOfEdge(i, 1);

        original_graph_cp.updateEdgeWeight(a, b, edge_values[i]);

        auto e = Edge(a, b, edge_values[i]);
    };

    // initializing the partition which will define the multicut
    andres::Partition<size_t> partition(graph.numberOfVertices());

    py::print("constructing maximum spanning tree");
    // get maximum spanning tree
    andres::graph::Graph<> MSTgraph = boruvkaMST(graph.numberOfVertices(), original_graph_cp);
    py::print("finished constructing maximum spanning tree");

    py::print("eliminating conflicts");
    size_t numEdges = graph.numberOfEdges();
    // eliminate conflicts
    for (size_t i = 0; i < numEdges; ++i) {

        double weight = edge_values[i];
        if (weight >= 0)
            continue;
        auto node1 = graph.vertexOfEdge(i, 0);
        auto node2 = graph.vertexOfEdge(i, 1);

        // if path between node1 and node2 remove edge with smalles weight
        std::queue<std::tuple<size_t, size_t, size_t, double>> Q;
        Q.emplace( node1, node1, -1, maxDouble);
        bool pathExists = false;
        while (!Q.empty() && !pathExists) {
            auto& [curNode, predecessor, minEdge, minWeight] = Q.front();
            Q.pop();
            
            //for (auto it = MSTgraph.adjacenciesFromVertexBegin(curNode); it != MSTgraph.adjacenciesFromVertexEnd(curNode); ++it) {
            for (size_t i = 0; i < MSTgraph.numberOfEdgesFromVertex(curNode); i++) {
                auto adjacency = MSTgraph.adjacencyFromVertex(curNode, i);
                size_t nextNode = adjacency.vertex();
                size_t mstEdge = adjacency.edge();
                if (nextNode == predecessor)
                    continue;

                /*
                auto edgeTuple = graph.findEdge(curNode, nextNode);
                if (!std::get<0>(edgeTuple)) {
                    throw std::runtime_error("edge: " + std::to_string(curNode) + ", " + std::to_string(nextNode) + " does not exist.");
                }*/

                size_t edge = std::get<1>(graph.findEdge(curNode, nextNode));

                double weight = edge_values[edge];
                if (weight < minWeight) {
                    minEdge = mstEdge;
                    minWeight = weight;
                }

                if (nextNode == node2) {
                    MSTgraph.eraseEdge(minEdge);
                    pathExists = true;
                    break;
                }
                else {
                    Q.emplace( nextNode, curNode, minEdge, minWeight );
                }
            }
        }
    }

    py::print("finished eliminating conflicts");

    // contract edges sequantially (this can potentially be parallized)
    for (size_t i = 0; i < MSTgraph.numberOfEdges(); ++i) {
        auto node1 = MSTgraph.vertexOfEdge(i, 0);
        auto node2 = MSTgraph.vertexOfEdge(i, 1);

        if (!original_graph_cp.edgeExists(node1, node2))
            continue;

        auto stable_vertex = node1;
        auto merge_vertex = node2;

        if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
            std::swap(stable_vertex, merge_vertex);

        // update partition
        partition.merge(stable_vertex, merge_vertex);

        // update dynamic graph
        for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex)) {
            if (p.first == stable_vertex)
                continue;

            original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);
        }

        original_graph_cp.removeVertex(merge_vertex);
    }

    // end timer
    auto end = std::chrono::high_resolution_clock::now();

    // write cut labels to graph
    for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

    // construct multicut
    std::set<size_t> multicut;
    for (size_t i = 0; i < graph.numberOfEdges(); i++) {
        if (edge_labels[i] == 1) {
            multicut.insert(i);
        };
    };

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return { multicut, static_cast<float>(elapsed.count()) };
}

// altered version from andres greedy-additive
std::tuple<std::set<size_t>, float> greedyParallelAdditiveEdgeContraction(const size_t numVertices, const std::vector<std::tuple<size_t, size_t, size_t>>& edges) {
    std::chrono::milliseconds contract_elapsed(0);

    // constructing the graph from parameters
    andres::graph::Graph<> graph;

    graph.insertVertices(numVertices);

    std::vector<size_t> edge_values(edges.size());

    size_t i = 0;
    for (const auto& [node1, node2, weight] : edges) {
        graph.insertEdge(node1, node2);
        edge_values[i++] = weight;
    };

    std::vector<char> edge_labels(graph.numberOfEdges());

    py::print("start solving");

    auto start = std::chrono::high_resolution_clock::now();

    // declaring objects required for the algorithm
    std::vector<std::map<size_t, size_t>> edge_editions(graph.numberOfVertices());
    DynamicGraph original_graph_cp(graph.numberOfVertices());
    std::priority_queue<Edge> Q;
    std::vector<Edge> edge_contraction_vector;
    std::vector<Edge> skipped_edges;
    std::vector<size_t> vertex_merge_vector;
    
    py::print("constructing dynamic graph");
    // constructing the dynamic graph that will be altered
    for (size_t i = 0; i < graph.numberOfEdges(); ++i)
    {
        auto a = graph.vertexOfEdge(i, 0);
        auto b = graph.vertexOfEdge(i, 1);

        original_graph_cp.updateEdgeWeight(a, b, edge_values[i]);

        auto e = Edge(a, b, edge_values[i]);
        e.edition = ++edge_editions[e.a][e.b];

        Q.push(e);
    };

    // initializing the partition which will define the multicut
    andres::Partition<size_t> partition(graph.numberOfVertices());

    py::print("start contracting");
    // for edge in contraction set
    while (!Q.empty())
    {   
        edge_contraction_vector.clear();
        vertex_merge_vector.clear();
        skipped_edges.clear();

        // py::print("find edges to contract");

        // find edges to contract
        size_t last_stable_vertex;
        size_t m = 10;
        size_t n = 0;

        // py::print(Q.size());

        while (n < m && !Q.empty()) {
            auto edge = Q.top();
            Q.pop();
            if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                continue;

            if (edge.w < 0)
                break;
            /*
            if (vertex_merge_set.find(edge.a) == vertex_merge_set.end() && vertex_merge_set.find(edge.b) == vertex_merge_set.end()) {
                vertex_merge_set.insert(edge.a);
                vertex_merge_set.insert(edge.b);
             */
            if (!std::count(vertex_merge_vector.begin(), vertex_merge_vector.end(), edge.a) && !std::count(vertex_merge_vector.begin(), vertex_merge_vector.end(), edge.b)) {
                vertex_merge_vector.push_back(edge.a);
                vertex_merge_vector.push_back(edge.b);
                edge_contraction_vector.push_back(edge);
                n++;
            } else {
                skipped_edges.push_back(edge);
            }
        }

        if (edge_contraction_vector.empty())
            break;

        for (const Edge edge : skipped_edges) {
            Q.push(edge);
        }

        // py::print("contract found edges");

        auto contract_start = std::chrono::high_resolution_clock::now();

        // contract edges sequantially (this can potentially be parallized)
        for (const Edge edge : edge_contraction_vector) {
            if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                continue;

            auto stable_vertex = edge.a;
            auto merge_vertex = edge.b;

            if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
                std::swap(stable_vertex, merge_vertex);

            // update partition
            partition.merge(stable_vertex, merge_vertex);

            // update dynamic graph
            for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex)) {
                if (p.first == stable_vertex)
                    continue;

                original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);

                auto e = Edge(stable_vertex, p.first, original_graph_cp.getEdgeWeight(stable_vertex, p.first));
                e.edition = ++edge_editions[e.a][e.b];

                Q.push(e);
            }

            original_graph_cp.removeVertex(merge_vertex);
        }

        auto contract_end = std::chrono::high_resolution_clock::now();
        contract_elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(contract_end - contract_start);

        // py::print("update dynamic graph");

        // update dynamic graph
        /*
        for (const auto vertex : vertex_merge_vector) {
            for (auto& p : original_graph_cp.getAdjacentVertices(vertex))
            {
                if (p.first == last_stable_vertex)
                    continue;

                original_graph_cp.updateEdgeWeight(last_stable_vertex, p.first, p.second);

                auto e = Edge(last_stable_vertex, p.first, original_graph_cp.getEdgeWeight(last_stable_vertex, p.first));
                e.edition = ++edge_editions[e.a][e.b];

                Q.push(e);
            }

            original_graph_cp.removeVertex(vertex);
        }*/

    }

    py::print("finished solving");

    // write cut labels to graph
    for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

    auto end = std::chrono::high_resolution_clock::now();
    
    py::print("constructing multicut");

    // construct multicut
    std::set<size_t> multicut;
    for (size_t i = 0; i < graph.numberOfEdges(); i++) {
        if (edge_labels[i] == 1) {
            multicut.insert(i);
        };
    };

    py::print(static_cast<float>(contract_elapsed.count()));

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return { multicut, static_cast<float>(elapsed.count()) };
};

std::tuple<std::set<size_t>, float> greedyAdditiveEdgeContraction(const size_t numVertices, const std::vector<std::tuple<size_t, size_t, double>>& edges) {
    andres::graph::Graph<> graph;

    graph.insertVertices(numVertices);

    std::vector<double> weights(edges.size());

    size_t i = 0;
    for (const auto& [node1, node2, weight] : edges) {
        graph.insertEdge(node1, node2);
        weights[i++] = weight;
    };

    std::vector<char> edge_labels(graph.numberOfEdges());

    py::print(graph.numberOfEdges());

    auto start = std::chrono::high_resolution_clock::now();
    andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);
    auto end = std::chrono::high_resolution_clock::now();

    std::set<size_t> multicut;
    for (size_t i = 0; i < graph.numberOfEdges(); i++) {
        if (edge_labels[i] == 1) {
            multicut.insert(i);
        };
    };

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return { multicut, static_cast<float>(elapsed.count()) };
};

std::tuple<std::set<size_t>, float> solveFromFile(const std::string fileName) {
    std::ifstream file;

    file.open(fileName);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    };

    std::string line;
    size_t i = 0;
    size_t numVertices = 0;
    std::vector<std::tuple<size_t, size_t, double>> edges;
    while (std::getline(file, line)) {
        if (i == 0) {
            // process number of vertices
            numVertices = std::stoi(line);
        }
        else {
            // process edges
            std::istringstream iss(line);
            double weight;
            size_t j = i;
            while (iss >> weight) {
                edges.emplace_back(i - 1, j++, weight);
            }
        }
        i++;
    };

    py::print(numVertices, edges.size());

    return spanningTreeEdgeContraction(numVertices, edges);
    //return greedyAdditiveEdgeContraction(numVertices, edges);
};


PYBIND11_MODULE(edge_contraction_solver, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: edge_contraction_solver

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
    m.def("greedyParallelAdditiveEdgeContraction", &greedyParallelAdditiveEdgeContraction, R"pbdoc(Parallel greedy additive edge contraction)pbdoc");
    m.def("greedyAdditiveEdgeContraction", &greedyAdditiveEdgeContraction, R"pbdoc(Classic greedy additive edge contraction by Björn Andres)pbdoc");
    m.def("greedyAdditiveEdgeContractionFromFile", &solveFromFile, R"pbdoc(Classic greedy additive edge contraction by Björn Andres)pbdoc");
    m.def("spanningTreeEdgeContraction", &spanningTreeEdgeContraction, R"pbdoc(Spanning tree edge contraction)pbdoc");
}
