#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstddef>
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


// class copied from andres greedy-additive
class DynamicGraph
{
public:
    DynamicGraph(int n) :
        vertices_(n)
    {}

    bool edgeExists(int a, int b) const
    {
        return !vertices_[a].empty() && vertices_[a].find(b) != vertices_[a].end();
    }

    std::map<int, int> const& getAdjacentVertices(int v) const
    {
        return vertices_[v];
    }

    int getEdgeWeight(int a, int b) const
    {
        return vertices_[a].at(b);
    }

    void removeVertex(int v)
    {
        for (auto& p : vertices_[v])
            vertices_[p.first].erase(v);

        vertices_[v].clear();
    }

    void updateEdgeWeight(int a, int b, int w)
    {
        vertices_[a][b] += w;
        vertices_[b][a] += w;
    }

private:
    std::vector<std::map<int, int>> vertices_;
};

// struct copied from andres greedy-additive
struct Edge
{
    Edge(int _a, int _b, int _w)
    {
        if (_a > _b)
            std::swap(_a, _b);

        a = _a;
        b = _b;

        w = _w;
    }

    int a;
    int b;
    int edition;
    int w;

    bool operator <(Edge const& other) const
    {
        return w < other.w;
    }
};

// altered version from andres greedy-additive
std::tuple<std::set<int>, float> greedyParallelAdditiveEdgeContraction(const int numVertices, const std::vector<std::tuple<int, int, int>>& edges) {
    std::chrono::milliseconds contract_elapsed(0);

    // constructing the graph from parameters
    andres::graph::Graph<> graph;

    graph.insertVertices(numVertices);

    std::vector<double> edge_values(edges.size());

    double i = 0;
    for (const auto& [node1, node2, weight] : edges) {
        graph.insertEdge(node1, node2);
        edge_values[i++] = weight;
    };

    std::vector<char> edge_labels(graph.numberOfEdges());

    py::print("start solving");

    auto start = std::chrono::high_resolution_clock::now();

    // declaring objects required for the algorithm
    std::vector<std::map<int, int>> edge_editions(graph.numberOfVertices());
    DynamicGraph original_graph_cp(graph.numberOfVertices());
    std::priority_queue<Edge> Q;
    std::vector<Edge> edge_contraction_vector;
    std::vector<Edge> skipped_edges;
    std::vector<int> vertex_merge_vector;
    
    py::print("constructing dynamic graph");
    // constructing the dynamic graph that will be altered
    for (int i = 0; i < graph.numberOfEdges(); ++i)
    {
        auto a = graph.vertexOfEdge(i, 0);
        auto b = graph.vertexOfEdge(i, 1);

        original_graph_cp.updateEdgeWeight(a, b, edge_values[i]);

        auto e = Edge(a, b, edge_values[i]);
        e.edition = ++edge_editions[e.a][e.b];

        Q.push(e);
    };

    // initializing the partition which will define the multicut
    andres::Partition<int> partition(graph.numberOfVertices());

    py::print("start contracting");
    // for edge in contraction set
    while (!Q.empty())
    {   
        edge_contraction_vector.clear();
        vertex_merge_vector.clear();
        skipped_edges.clear();

        // py::print("find edges to contract");

        // find edges to contract
        int last_stable_vertex;
        int m = 10;
        int n = 0;

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
    for (int i = 0; i < graph.numberOfEdges(); ++i)
        edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

    auto end = std::chrono::high_resolution_clock::now();
    
    py::print("constructing multicut");

    // construct multicut
    std::set<int> multicut;
    for (double i = 0; i < graph.numberOfEdges(); i++) {
        if (edge_labels[i] == 1) {
            multicut.insert(i);
        };
    };

    py::print(static_cast<float>(contract_elapsed.count()));

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return { multicut, static_cast<float>(elapsed.count()) };
};

std::tuple<std::set<int>, float> greedyAdditiveEdgeContraction(const int numVertices, const std::vector<std::tuple<int, int, int>>& edges) {
    andres::graph::Graph<> graph;

    graph.insertVertices(numVertices);

    std::vector<double> weights(edges.size());

    double i = 0;
    for (const auto& [node1, node2, weight] : edges) {
        graph.insertEdge(node1, node2);
        weights[i++] = weight;
    };

    std::vector<char> edge_labels(graph.numberOfEdges());

    py::print(graph.numberOfEdges());

    auto start = std::chrono::high_resolution_clock::now();
    andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);
    auto end = std::chrono::high_resolution_clock::now();

    std::set<int> multicut;
    for (double i = 0; i < graph.numberOfEdges(); i++) {
        if (edge_labels[i] == 1) {
            multicut.insert(i);
        };
    };

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return { multicut, static_cast<float>(elapsed.count()) };
};

std::tuple<std::set<int>, float> solveFromFile(const std::string fileName) {
    std::ifstream file;

    file.open(fileName);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
    };

    std::string line;
    int i = 0;
    int numVertices = 0;
    std::vector<std::tuple<int, int, int>> edges;
    while (std::getline(file, line)) {
        if (i == 0) {
            // process number of vertices
            numVertices = std::stoi(line);
        }
        else {
            // process edges
            std::istringstream iss(line);
            int weight;
            int j = i;
            while (iss >> weight) {
                edges.emplace_back(i - 1, j++, weight);
            }
        }
        i++;
    };

    py::print(numVertices, edges.size());

    return greedyAdditiveEdgeContraction(numVertices, edges);
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
}
