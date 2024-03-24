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
#include <iomanip>

#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"
#include "andres/partition.hxx"

namespace py = pybind11;

const size_t maxInt = std::numeric_limits<size_t>::max();
constexpr size_t NOT_PRESENT = static_cast<size_t>(-1);
const double maxDouble = std::numeric_limits<double>::max();
const double minDouble = std::numeric_limits<double>::min();

// hash for an unordered set of tuples
struct Hash {
    size_t operator()(const std::tuple<size_t, size_t>& tuple) const {
        const auto& [x, y] = tuple;
        size_t hash1 = std::hash<size_t>{}(x);
        size_t hash2 = std::hash<size_t>{}(y);
        return hash1 ^ (hash2 << 1); // Simple XOR hash combiner
    }
};


// altered DynamicGraph class from https://github.com/bjoern-andres/graph/blob/master/include/andres/graph/multicut/greedy-additive.hxx
class DynamicGraph
{
public:
    DynamicGraph(size_t n) :
        vertices_(n)
    {}

    // Copy constructor
    DynamicGraph(const DynamicGraph& other) :
        vertices_(other.vertices_),
        edges_(other.edges_)
    {}

    bool edgeExists(size_t a, size_t b) const
    {
        return !vertices_[a].empty() && vertices_[a].find(b) != vertices_[a].end();
    }

    std::map<size_t, double> const& getAdjacentVertices(size_t v) const
    {
        return vertices_[v];
    }

    double getEdgeWeight(size_t a, size_t b) const
    {
        return vertices_[a].at(b);
    }

    void removeVertex(size_t v)
    {
        for (auto& p : vertices_[v]) {
            vertices_[p.first].erase(v);
            if (p.first < v) {
                edges_.erase({ p.first, v });
            } else {
                edges_.erase({ v, p.first });
            }
        }

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

// altered Edge struct from https://github.com/bjoern-andres/graph/blob/master/include/andres/graph/multicut/greedy-additive.hxx
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

class EdgeContractionSolver {
public:
    EdgeContractionSolver() {
        trackHistory = false;
    };

    void activateTrackHistory() {
        trackHistory = true;
    };

    std::tuple<andres::graph::Graph<>, std::vector<double>> constructGraphAndWeights() {
        py::print("constructing graph");

        andres::graph::Graph<> graph;

        graph.insertVertices(numVertices);

        std::vector<double> weights(edges.size());

        size_t i = 0;
        for (const auto& [node1, node2, weight] : edges) {
            graph.insertEdge(node1, node2);
            weights[i++] = weight;
        };

        py::print("length weights:", weights.size());
        py::print("number of edges:", graph.numberOfEdges());
        py::print("number of vertices:", graph.numberOfVertices());

        return { graph, weights };
    }

    double getScore() {
        std::vector<double> weights = std::get<1>(constructGraphAndWeights());

        double score = 0;

        for (const auto& edge : multicut) {
            score += weights[edge];
        }

        return score;
    }

    std::vector<int> getContractionHistory() {
        return contractionHistory;
    }

    void constructMulticut(std::vector<char> edge_labels) {
        multicut.clear();
        for (size_t i = 0; i < edge_labels.size(); i++) {
            if (edge_labels[i] == 1) {
                multicut.insert(i);
            };
        };

        py::print("constructed multicut of length:", multicut.size());
    }

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

    void maximumMatching() {
        if (trackHistory)
            openFile();

        auto& [graph, weights] = constructGraphAndWeights();
        std::vector<char> edge_labels(graph.numberOfEdges());
        contractionHistory.clear();

        DynamicGraph original_graph_cp(graph.numberOfVertices());

        // start timer
        auto start = std::chrono::high_resolution_clock::now();

        py::print("constructing dynamic graph");
        // constructing the dynamic graph that will be altered
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        {
            auto a = graph.vertexOfEdge(i, 0);
            auto b = graph.vertexOfEdge(i, 1);

            original_graph_cp.updateEdgeWeight(a, b, weights[i]);

            auto e = Edge(a, b, weights[i]);
        };

        // initializing the partition which will define the multicut
        andres::Partition<size_t> partition(graph.numberOfVertices());

        py::print("find handshakes");

        // repeatedly apply the one handshake algorithm until no improvement can be done
        bool contracted = true;
        while (contracted) {
            contracted = false;        
            std::unordered_set<std::tuple<size_t, size_t>, Hash> edge_contraction_set;
            
            DynamicGraph handshake_graph = original_graph_cp;

            // repeatedly search for handshakes until none can be found
            bool found_handshake = true;
            while (found_handshake) {
                found_handshake = false;
            
                std::vector<size_t> hands(graph.numberOfVertices());

                // extend hand to neighbor (can be parallelized on GPU)
                for (size_t node = 0; node < graph.numberOfVertices(); ++node) {
                    size_t strongest_neighbor = NOT_PRESENT;
                    double strongest_neighbor_weight = minDouble;
                    for (const auto& [neighbor, weight] : handshake_graph.getAdjacentVertices(node)) {
                        if (weight > 0 && weight > strongest_neighbor_weight) {
                            strongest_neighbor = neighbor;
                            strongest_neighbor_weight = weight;
                        }
                    }
                    hands[node] = strongest_neighbor;
                }

                // find handshakes (can be parallelized on GPU)
                for (const auto& [u, v] : handshake_graph.getEdges()) {
                    if (!handshake_graph.edgeExists(u, v))
                        continue;
                    if (hands[u] == v && hands[v] == u) {
                        found_handshake = true;
                        edge_contraction_set.emplace(u, v);
                        handshake_graph.removeVertex(u);
                        handshake_graph.removeVertex(v);
                    }
                }
            }
            
            py::print("contracting", edge_contraction_set.size(), "edges");

            // contract edges sequantially (this can potentially be parallized)
            for (const auto& [u, v] : edge_contraction_set) {
                if (!original_graph_cp.edgeExists(u, v))
                    throw std::runtime_error("edge: " + std::to_string(u) + ", " + std::to_string(v) + " does not exist.");

                auto stable_vertex = u;
                auto merge_vertex = v;
                contracted = true;

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

            if (trackHistory) {
                writeContractedEdgesHistory(graph, partition);
                setHistory();
            }
                

            if (edge_contraction_set.size() > 0)
                contractionHistory.push_back(edge_contraction_set.size());
        }

        // end timer
        auto end = std::chrono::high_resolution_clock::now();

        py::print("calculating multicut");

        // write cut labels to graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        // construct multicut
        constructMulticut(edge_labels);

        py::print("finished solving");

        if (trackHistory)
            closeFile();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    void maximumMatchingWithCutoff() {
        auto& [graph, weights] = constructGraphAndWeights();
        std::vector<char> edge_labels(graph.numberOfEdges());

        DynamicGraph original_graph_cp(graph.numberOfVertices());

        // start timer
        auto start = std::chrono::high_resolution_clock::now();

        py::print("constructing dynamic graph");
        // constructing the dynamic graph that will be altered
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        {
            auto a = graph.vertexOfEdge(i, 0);
            auto b = graph.vertexOfEdge(i, 1);

            original_graph_cp.updateEdgeWeight(a, b, weights[i]);

            auto e = Edge(a, b, weights[i]);
        };

        // initializing the partition which will define the multicut
        andres::Partition<size_t> partition(graph.numberOfVertices());

        py::print("find handshakes");

        // repeatedly apply the one handshake algorithm until no improvement can be done
        bool contracted = true;
        while (contracted) {
            contracted = false;
            std::unordered_set<std::tuple<size_t, size_t>, Hash> edge_contraction_set;

            DynamicGraph handshake_graph = original_graph_cp;

            // repeatedly search for handshakes until none can be found
            bool found_handshake = true;
            while (found_handshake) {
                found_handshake = false;

                std::vector<size_t> hands(graph.numberOfVertices());

                // extend hand to neighbor (can be parallelized on GPU)
                for (size_t node = 0; node < graph.numberOfVertices(); ++node) {
                    size_t strongest_neighbor = NOT_PRESENT;
                    double strongest_neighbor_weight = minDouble;
                    for (const auto& [neighbor, weight] : handshake_graph.getAdjacentVertices(node)) {
                        if (weight > 0 && weight > strongest_neighbor_weight) {
                            strongest_neighbor = neighbor;
                            strongest_neighbor_weight = weight;
                        }
                    }
                    hands[node] = strongest_neighbor;
                }

                // find handshakes (can be parallelized on GPU)
                for (const auto& [u, v] : handshake_graph.getEdges()) {
                    if (!handshake_graph.edgeExists(u, v))
                        continue;
                    if (hands[u] == v && hands[v] == u) {
                        found_handshake = true;
                        edge_contraction_set.emplace(u, v);
                        handshake_graph.removeVertex(u);
                        handshake_graph.removeVertex(v);
                    }
                }
            }

            py::print("contracting", edge_contraction_set.size(), "edges");

            // contract edges sequantially (this can potentially be parallized)
            for (const auto& [u, v] : edge_contraction_set) {
                if (!original_graph_cp.edgeExists(u, v))
                    throw std::runtime_error("edge: " + std::to_string(u) + ", " + std::to_string(v) + " does not exist.");

                auto stable_vertex = u;
                auto merge_vertex = v;
                contracted = true;

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

            if (edge_contraction_set.size() == 1)
                break;
        }

        py::print("stopped maximum matching; continueing with greedy additive");

        // altered greedy additive edge contraction from https://github.com/bjoern-andres/graph/blob/master/include/andres/graph/multicut/greedy-additive.hxx
        std::vector<std::map<size_t, size_t>> edge_editions(graph.numberOfVertices());
        std::priority_queue<Edge> Q;
        for (const auto& [u, v] : original_graph_cp.getEdges()) {
            auto weight = original_graph_cp.getEdgeWeight(u, v);
            auto e = Edge(u, v, weight);
            e.edition = ++edge_editions[e.a][e.b];

            Q.push(e);
        }

        while (!Q.empty())
        {
            auto edge = Q.top();
            Q.pop();

            if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                continue;

            if (edge.w < 0)
                break;

            auto stable_vertex = edge.a;
            auto merge_vertex = edge.b;

            if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
                std::swap(stable_vertex, merge_vertex);

            partition.merge(stable_vertex, merge_vertex);

            for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex))
            {
                if (p.first == stable_vertex)
                    continue;

                original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);

                auto e = Edge(stable_vertex, p.first, original_graph_cp.getEdgeWeight(stable_vertex, p.first));
                e.edition = ++edge_editions[e.a][e.b];

                Q.push(e);
            }

            original_graph_cp.removeVertex(merge_vertex);
        }

        // end timer
        auto end = std::chrono::high_resolution_clock::now();

        py::print("calculating multicut");

        // write cut labels to graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        // construct multicut
        constructMulticut(edge_labels);

        py::print("finished solving");

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    // an altered version of: https://www.geeksforgeeks.org/boruvkas-algorithm-greedy-algo-9/
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
        double MSTweight = 0;

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

                double weight = graph.getEdgeWeight(u, v);

                // ignore negative edges
                if (weight < 0)
                    continue;

                size_t w = weight;

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

    void spanningTreeEdgeContraction() {
        if (trackHistory)
            openFile();

        auto& [graph, weights] = constructGraphAndWeights();
        std::vector<char> edge_labels(graph.numberOfEdges());

        contractionHistory.clear();

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

            original_graph_cp.updateEdgeWeight(a, b, weights[i]);

            auto e = Edge(a, b, weights[i]);
        };

        // initializing the partition which will define the multicut
        andres::Partition<size_t> partition(graph.numberOfVertices());

        // reapeat until no edge can be contracted anymore
        bool contracted = true;
        while (contracted) {
            contracted = false;
            py::print("constructing maximum spanning tree");
            // get maximum spanning tree
            andres::graph::Graph<> MSTgraph = boruvkaMST(graph.numberOfVertices(), original_graph_cp);
            py::print("finished constructing maximum spanning tree of size:", MSTgraph.numberOfEdges());

            py::print("eliminating conflicts");
            size_t numEdges = graph.numberOfEdges();

            // eliminate conflicts
            for (const auto& [node1, node2] : original_graph_cp.getEdges()) {
                double weight = original_graph_cp.getEdgeWeight(node1, node2);
                if (weight >= 0)
                    continue;

                // if path between node1 and node2 remove edge with smalles weight
                std::queue<std::tuple<size_t, size_t, size_t, size_t, double>> Q;
                Q.emplace( node1, node1, -1, -1, maxDouble);
                bool pathExists = false;
                while (!Q.empty() && !pathExists) {
                    auto& [curNode, predecessor, minVertex1, minVertex2, minWeight] = Q.front();
                    Q.pop();
            
                    for (size_t i = 0; i < MSTgraph.numberOfEdgesFromVertex(curNode); i++) {
                        auto adjacency = MSTgraph.adjacencyFromVertex(curNode, i);
                        size_t nextNode = adjacency.vertex();
                        if (nextNode == predecessor)
                            continue;

                        double weight = original_graph_cp.getEdgeWeight(curNode, nextNode);
                        if (weight < minWeight) {
                            minVertex1 = curNode;
                            minVertex2 = nextNode;
                            minWeight = weight;
                        }
                        if (nextNode == node2) {
                            MSTgraph.eraseEdge(std::get<1>(MSTgraph.findEdge(minVertex1, minVertex2)));
                            pathExists = true;
                            break;
                        }
                        else {
                            Q.emplace(nextNode, curNode, minVertex1, minVertex2 , minWeight);
                        }
                    }
                }
            }

            py::print("finished eliminating conflicts with size:", MSTgraph.numberOfEdges());

            std::vector<size_t> parent(MSTgraph.numberOfVertices());

            for (size_t node = 0; node < MSTgraph.numberOfVertices(); node++) {
                parent[node] = node;
            }

            // contract edges sequantially (this can potentially be parallized)
            for (size_t i = 0; i < MSTgraph.numberOfEdges(); ++i) {
                auto node1 = MSTgraph.vertexOfEdge(i, 0);
                auto node2 = MSTgraph.vertexOfEdge(i, 1);

                auto stable_vertex = find(parent, node1);
                auto merge_vertex = find(parent, node2);

                if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
                    std::swap(stable_vertex, merge_vertex);

                // update partition
                partition.merge(stable_vertex, merge_vertex);

                // map merge vertex to stable vertex, because merge vertex is now included in stable vertex
                parent[merge_vertex] = stable_vertex;

                // update dynamic graph
                for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex)) {
                    if (p.first == stable_vertex)
                        continue;
                    original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);
                }

                original_graph_cp.removeVertex(merge_vertex);
                contracted = true;
            }

            if (trackHistory) {
                writeContractedEdgesHistory(graph, partition);
                setHistory();
            }
                
                

            if (MSTgraph.numberOfEdges() > 0)
                contractionHistory.push_back(MSTgraph.numberOfEdges());

            py::print("anything was contracted:", contracted);
        }

        // end timer
        auto end = std::chrono::high_resolution_clock::now();

        // write cut labels to graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        // construct multicut
        constructMulticut(edge_labels);

        py::print("finished solving");

        if (trackHistory)
            closeFile();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    // a test implementation that turned out to not be usefull
    void spanningTreeEdgeContractionContinued() {
        auto& [graph, weights] = constructGraphAndWeights();
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

            original_graph_cp.updateEdgeWeight(a, b, weights[i]);

            auto e = Edge(a, b, weights[i]);
        };

        // initializing the partition which will define the multicut
        andres::Partition<size_t> partition(graph.numberOfVertices());

        // reapeat until no edge can be contracted anymore
        bool contracted = true;
        while (contracted) {
            contracted = false;
            py::print("constructing maximum spanning tree");
            // get maximum spanning tree
            andres::graph::Graph<> MSTgraph = boruvkaMST(graph.numberOfVertices(), original_graph_cp);
            py::print("finished constructing maximum spanning tree of size:", MSTgraph.numberOfEdges());

            py::print("eliminating conflicts");
            size_t numEdges = graph.numberOfEdges();
            // eliminate conflicts
            for (size_t i = 0; i < numEdges; ++i) {

                double weight = weights[i];
                if (weight >= 0)
                    continue;
                auto node1 = graph.vertexOfEdge(i, 0);
                auto node2 = graph.vertexOfEdge(i, 1);

                // if path between node1 and node2 remove edge with smalles weight
                std::queue<std::tuple<size_t, size_t, size_t, double>> Q;
                Q.emplace(node1, node1, -1, maxDouble);
                bool pathExists = false;
                while (!Q.empty() && !pathExists) {
                    auto& [curNode, predecessor, minEdge, minWeight] = Q.front();
                    Q.pop();

                    for (size_t i = 0; i < MSTgraph.numberOfEdgesFromVertex(curNode); i++) {
                        auto adjacency = MSTgraph.adjacencyFromVertex(curNode, i);
                        size_t nextNode = adjacency.vertex();
                        size_t mstEdge = adjacency.edge();
                        if (nextNode == predecessor)
                            continue;

                        size_t edge = std::get<1>(graph.findEdge(curNode, nextNode));

                        double weight = weights[edge];
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
                            Q.emplace(nextNode, curNode, minEdge, minWeight);
                        }
                    }
                }
            }

            py::print("finished eliminating conflicts with size:", MSTgraph.numberOfEdges());

            std::vector<size_t> parent(graph.numberOfVertices());

            for (size_t node = 0; node < graph.numberOfVertices(); node++) {
                parent[node] = node;
            }

            // contract edges sequantially (this can potentially be parallized)
            for (size_t i = 0; i < graph.numberOfEdges(); ++i) {
                auto node1 = graph.vertexOfEdge(i, 0);
                auto node2 = graph.vertexOfEdge(i, 1);

                if (!std::get<0>(MSTgraph.findEdge(node1, node2)))
                    continue;

                node1 = find(parent, node1);
                node2 = find(parent, node2);

                if (!original_graph_cp.edgeExists(node1, node2))
                    continue;

                auto stable_vertex = node1;
                auto merge_vertex = node2;

                if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
                    std::swap(stable_vertex, merge_vertex);

                // update partition
                partition.merge(stable_vertex, merge_vertex);

                // map merge vertex to stable vertex, because merge vertex is now included in stable vertex
                parent[merge_vertex] = stable_vertex;

                // update dynamic graph
                for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex)) {
                    if (p.first == stable_vertex)
                        continue;
                    original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);
                }

                original_graph_cp.removeVertex(merge_vertex);
                contracted = true;
            }
        }

        // continue with greedy additive edge contraction taken from the andres repository
        std::vector<std::map<size_t, size_t>> edge_editions(graph.numberOfVertices());
        std::priority_queue<Edge> Q;
        for (const auto& [u, v] : original_graph_cp.getEdges()) {
            auto weight = original_graph_cp.getEdgeWeight(u, v);
            auto e = Edge(u, v, weight);
            e.edition = ++edge_editions[e.a][e.b];

            Q.push(e);
        }

        while (!Q.empty())
        {
            auto edge = Q.top();
            Q.pop();

            if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                continue;

            if (edge.w < 0)
                break;

            auto stable_vertex = edge.a;
            auto merge_vertex = edge.b;

            if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
                std::swap(stable_vertex, merge_vertex);

            partition.merge(stable_vertex, merge_vertex);

            for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex))
            {
                if (p.first == stable_vertex)
                    continue;

                original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);

                auto e = Edge(stable_vertex, p.first, original_graph_cp.getEdgeWeight(stable_vertex, p.first));
                e.edition = ++edge_editions[e.a][e.b];

                Q.push(e);
            }

            original_graph_cp.removeVertex(merge_vertex);
        }

        // end timer
        auto end = std::chrono::high_resolution_clock::now();

        // write cut labels to graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        // construct multicut
        constructMulticut(edge_labels);

        py::print("finished solving");

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    void greedyMatchingsEdgeContractionWithMulticutApplied(std::set<size_t> preMulticut) {
        if (trackHistory)
            openFile();

        auto& [graph, weights] = constructGraphAndWeights();
        std::vector<char> edge_labels(graph.numberOfEdges());

        contractionHistory.clear();

        std::chrono::milliseconds contract_elapsed(0);
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

            original_graph_cp.updateEdgeWeight(a, b, weights[i]);

            auto e = Edge(a, b, weights[i]);
            e.edition = ++edge_editions[e.a][e.b];

            Q.push(e);
        };

        py::print("checkpoint");

        // initializing the partition which will define the multicut
        andres::Partition<size_t> partition(graph.numberOfVertices());

        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        {   

            auto it = preMulticut.find(i);
            if (it != preMulticut.end())
                continue;

            auto a = graph.vertexOfEdge(i, 0);
            auto b = graph.vertexOfEdge(i, 1);

            if (!original_graph_cp.edgeExists(a, b))
                continue;

            auto stable_vertex = a;
            auto merge_vertex = b;

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
        };

        py::print("start contracting");
        // for edge in contraction set
        while (!Q.empty())
        {
            edge_contraction_vector.clear();
            vertex_merge_vector.clear();
            skipped_edges.clear();

            // find edges to contract
            size_t last_stable_vertex;
            size_t n = 0;

            while (n < 1 && !Q.empty()) {
                auto edge = Q.top();
                Q.pop();
                if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                    continue;

                if (edge.w < 0)
                    break;

                if (!std::count(vertex_merge_vector.begin(), vertex_merge_vector.end(), edge.a) && !std::count(vertex_merge_vector.begin(), vertex_merge_vector.end(), edge.b)) {
                    vertex_merge_vector.push_back(edge.a);
                    vertex_merge_vector.push_back(edge.b);
                    edge_contraction_vector.push_back(edge);
                    n++;
                }
                else {
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

            if (trackHistory) {
                writeContractedEdgesHistory(graph, partition);
                setHistory();
            }


            if (edge_contraction_vector.size() > 0)
                contractionHistory.push_back(edge_contraction_vector.size());

            auto contract_end = std::chrono::high_resolution_clock::now();
            contract_elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(contract_end - contract_start);

        }

        // write cut labels to graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        auto end = std::chrono::high_resolution_clock::now();

        constructMulticut(edge_labels);

        py::print("finished solving");

        if (trackHistory)
            closeFile();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    };

    // altered version from andres greedy-additive
    void greedyMatchingsEdgeContraction(size_t m = maxInt) {
        if (trackHistory)
            openFile();

        auto& [graph, weights] = constructGraphAndWeights();
        std::vector<char> edge_labels(graph.numberOfEdges());

        contractionHistory.clear();

        std::chrono::milliseconds contract_elapsed(0);
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

            original_graph_cp.updateEdgeWeight(a, b, weights[i]);

            auto e = Edge(a, b, weights[i]);
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

            // find edges to contract
            size_t last_stable_vertex;
            size_t n = 0;

            while (n < m && !Q.empty()) {
                auto edge = Q.top();
                Q.pop();
                if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                    continue;

                if (edge.w < 0)
                    break;

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

            if (trackHistory) {
                writeContractedEdgesHistory(graph, partition);
                setHistory();
            }
                

            if (edge_contraction_vector.size() > 0)
                contractionHistory.push_back(edge_contraction_vector.size());

            auto contract_end = std::chrono::high_resolution_clock::now();
            contract_elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(contract_end - contract_start);

        }

        // write cut labels to graph
        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        auto end = std::chrono::high_resolution_clock::now();

        constructMulticut(edge_labels);

        py::print("finished solving");

        if (trackHistory)
            closeFile();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    };

    void largestPositiveCostEdgeContraction() {
        if (trackHistory)
            openFile();

        auto& [graph, weights] = constructGraphAndWeights();

        std::vector<char> edge_labels(graph.numberOfEdges());

        auto start = std::chrono::high_resolution_clock::now();
        andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);
        auto end = std::chrono::high_resolution_clock::now();

        constructMulticut(edge_labels);

        py::print("finished solving");

        if (trackHistory)
            closeFile();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    };

    void kernighanLin() {
        if (trackHistory)
            openFile();

        auto& [graph, weights] = constructGraphAndWeights();

        std::vector<char> edge_labels(graph.numberOfEdges());

        auto start = std::chrono::high_resolution_clock::now();
        andres::graph::multicut::kernighanLin(graph, weights, edge_labels, edge_labels);
        auto end = std::chrono::high_resolution_clock::now();

        constructMulticut(edge_labels);

        py::print("finished solving");

        if (trackHistory)
            closeFile();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    };

    void load(int loadNumVertices, std::vector<std::tuple<int, int, double>> loadEdges) {
        py::print("Loading graph from parameters");

        edges.clear();
        numVertices = loadNumVertices;

        for (const auto& edge : loadEdges) {
            edges.emplace_back(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge));
        }

        py::print("loaded edges:", edges.size());
        py::print("loaded vertices:", numVertices);
    }

    void loadFromFile(const std::string fileName) {
        py::print("Loading graph from file", fileName);

        std::ifstream file;

        file.open(fileName);

        if (!file.is_open()) {
            std::cerr << "Error opening file" << std::endl;
        };

        std::string line;
        size_t i = 0;
        numVertices = 0;
        edges.clear();
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
        py::print("loaded edges:", edges.size());
        py::print("loaded vertices:", numVertices);
    };

    float getElapsedTime() {
        return static_cast<float>(elapsed.count());
    }

    std::set<int> getMulticut() {
        return multicut;
    }

    void writeContractedEdgesHistory(andres::graph::Graph<> graph, andres::Partition<size_t> partition) {
        std::vector<char> edge_labels(graph.numberOfEdges());

        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;

        for (size_t i = 0; i < edge_labels.size(); i++) {
            auto it = foundEdges.find(i);
            if (edge_labels[i] == 0 && it == foundEdges.end()) {
                contractedEdgesHistory.push_back(i);
                foundEdges.emplace(i);
            };
        };
    }

    void openFile() {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        foundEdges.clear();

        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S");
        filename = ss.str() + ".txt";
        file.open(filename);
        py::print("created file", filename);

    }

    void closeFile() {
        file.close();
    }

    void setHistory() {
        if (!file.is_open()) file.open(filename, std::ios::app);

        for (size_t i = 0; i < contractedEdgesHistory.size(); ++i) {
            file << contractedEdgesHistory[i];
            if (i != contractedEdgesHistory.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
        contractedEdgesHistory.clear();
    }



private:
    size_t numVertices;
    std::vector<std::tuple<size_t, size_t, double>> edges;
    std::chrono::milliseconds elapsed;
    std::set<int> multicut;
    std::vector<int> contractionHistory;
    std::vector<size_t> contractedEdgesHistory;
    std::set<size_t> foundEdges;
    std::string filename;
    std::ofstream file;
    bool trackHistory;
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
    pybind11::class_<EdgeContractionSolver>(m, "EdgeContractionSolver")
        .def(pybind11::init<>())
        .def("load", &EdgeContractionSolver::load)
        .def("load_from_file", &EdgeContractionSolver::loadFromFile)
        .def("largest_positive_cost_edge_contraction", &EdgeContractionSolver::largestPositiveCostEdgeContraction)
        .def("kernighanLin", &EdgeContractionSolver::kernighanLin)
        .def("greedy_matchings_edge_contraction", &EdgeContractionSolver::greedyMatchingsEdgeContraction, pybind11::arg("m") = maxInt)
        .def("greedy_matchings_edge_contraction_with_multicut_applied", &EdgeContractionSolver::greedyMatchingsEdgeContractionWithMulticutApplied)
        .def("spanning_tree_edge_contraction", &EdgeContractionSolver::spanningTreeEdgeContraction)
        .def("spanning_tree_edge_contraction_continued", &EdgeContractionSolver::spanningTreeEdgeContractionContinued)
        .def("maximum_matching", &EdgeContractionSolver::maximumMatching)
        .def("maximum_matching_with_cutoff", &EdgeContractionSolver::maximumMatchingWithCutoff)
        .def("get_multicut", &EdgeContractionSolver::getMulticut)
        .def("get_elapsed_time", &EdgeContractionSolver::getElapsedTime)
        .def("get_score", &EdgeContractionSolver::getScore)
        .def("get_contraction_history", &EdgeContractionSolver::getContractionHistory)
        .def("activate_track_history", &EdgeContractionSolver::activateTrackHistory);
}
