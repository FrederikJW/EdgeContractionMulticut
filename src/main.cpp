#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <algorithm>
#include <optional>
#include <map>
#include <set>
#include <vector>
#include <tuple>
#include <queue>
#include <stdexcept>
#include <limits>

namespace py = pybind11;

const bool debug = false;

struct TupleComparator {
    bool operator()(const std::tuple<int, int, int>& lhs, const std::tuple<int, int, int>& rhs) const {
        return std::get<2>(lhs) < std::get<2>(rhs);
    }
};

// Edge structure to store cost
struct Edge {
    int node1;
    int node2;
    int cost;
    int id;

    Edge() : node1(0), node2(0), cost(0), id(0) {};
    Edge(int n1, int n2, int c, int i) : node1(n1), node2(n2), cost(c), id(i) {};
};

// Graph class
class Graph {
private:
    std::map<int, Edge> allEdges;
    std::set<int> allNodes;
    std::map<int, std::map<int, Edge>> adjacencyList;
    std::map<int, std::set<int>> mergedEdges;
    
public:
    // Function to add an edge with a specified cost
    void addEdge(int v1, int v2, int cost, int id) {
        adjacencyList[v1][v2] = Edge(v1, v2, cost, id);
        adjacencyList[v2][v1] = Edge(v2, v1, cost, id);
        allEdges[id] = Edge(v1, v2, cost, id);
        mergedEdges[id].insert(id);
    };

    void removeEdge(int v1, int v2, int key, int id) {
        adjacencyList[v1].erase(v2);
        adjacencyList[v2].erase(v1);
        allEdges.erase(id);
    };

    void addNode(int id) {
        allNodes.insert(id);
    };

    void removeNode(int node) {
        std::vector<Edge> edges = getEdges(node);
        for (const auto& edge : edges) {
            allEdges.erase(edge.id);
        };

        std::map<int, Edge> nodeToKeysToEdges = adjacencyList[node];
        adjacencyList.erase(node);

        for (const auto& pair : nodeToKeysToEdges) {
            adjacencyList[pair.first].erase(node);
        };

        allNodes.erase(node);
    };

    std::vector<Edge> getAllEdges() {
        std::vector<Edge> edges;
        for (const auto& pair : allEdges) {
            edges.push_back(pair.second);
        };
        return edges;
    };

    std::vector<Edge> getEdges(int node) {
        std::vector<Edge> edges;
        for (const auto& pair1 : adjacencyList[node]) {
            edges.push_back(pair1.second);
        };
        return edges;
    };

    Edge getEdge(int node1, int node2) {
        if (!adjacencyList[node1].count(node2)) {
            throw std::runtime_error("no edge found.");
        }
        return adjacencyList[node1][node2];
    };

    std::vector<int> getAllEdgeIds() {
        std::vector<int> edges;
        for (const auto& pair : allEdges) {
            edges.push_back(pair.first);
        };
        return edges;
    };

    std::set<int> getAllNodeIds() {
        return allNodes;
    };

    std::set<int> contract(int node1, int node2) {
        py::print("contraction:", node1, node2);
        for (const auto& edge : getEdges(node2)) {
            int neighbor = edge.node2;

            if (neighbor == node1) continue;
            int newCost;

            if (adjacencyList[node1].count(neighbor)) {
                Edge existingEdge = adjacencyList[node1][neighbor];
                newCost = existingEdge.cost + edge.cost;
                std::set_union(mergedEdges[edge.id].begin(), mergedEdges[edge.id].end(), mergedEdges[existingEdge.id].begin(), mergedEdges[existingEdge.id].end(), std::inserter(mergedEdges[edge.id], mergedEdges[edge.id].begin()));
                mergedEdges.erase(existingEdge.id);
                allEdges.erase(existingEdge.id);
            } else {
                newCost = edge.cost;
            };

            adjacencyList[node1][neighbor] = Edge(node1, neighbor, newCost, edge.id);
            adjacencyList[neighbor][node1] = Edge(neighbor, node1, newCost, edge.id);
            allEdges[edge.id] = Edge(node1, neighbor, newCost, edge.id);

            adjacencyList[neighbor].erase(node2);
        };

        Edge edge = getEdge(node1, node2);
        py::print("contracting edge:", edge.id);

        allNodes.erase(node2);
        adjacencyList.erase(node2);
        adjacencyList[node1].erase(node2);
        allEdges.erase(edge.id);

        return mergedEdges[edge.id];
    };
};

class LargestPositiveCost {
private:
    Graph graph;
    std::map<int, int> nodeToComponent;
    std::map<int, std::set<int>> components;


public:
    void loadGraph(const std::vector<std::tuple<int, int, int>>& nodes, const std::vector<std::tuple<int, int, int, int, int>>& edges) {
        int n = 0;

        for (const auto& node : nodes) {
            int id = std::get<0>(node);
            graph.addNode(id);
            nodeToComponent[id] = n;
            components[n++].insert(id);
        };

        for (const auto& edge : edges) {
            graph.addEdge(std::get<0>(edge), std::get<1>(edge), std::get<3>(edge), std::get<4>(edge));
        };
    };

    std::set<int> solve() {
        py::print("start solving");

        std::vector<int> allEdges = graph.getAllEdgeIds();
        std::set<int> multicut(allEdges.begin(), allEdges.end());

        while (true) {
            // select edge to contract
            int maxEdgeCost = 0;
            Edge maxCostEdge;
            for (const auto& edge : graph.getAllEdges()) {
                if (!multicut.contains(edge.id)) continue;

                if (edge.cost > maxEdgeCost) {
                    maxEdgeCost = edge.cost;
                    maxCostEdge = edge;
                }
            };

            py::print("max edge:", maxCostEdge.id, maxEdgeCost);

            // stop algorithm if there is no edge to contract left
            if (maxEdgeCost == 0) break;

            // contract edge
            std::set<int> ids = graph.contract(maxCostEdge.node1, maxCostEdge.node2);
            std::set<int> setDifference;
            std::set_difference(multicut.begin(), multicut.end(), ids.begin(), ids.end(), std::inserter(setDifference, setDifference.begin()));
            multicut = setDifference;
        };
        
        py::print("finished solving");
        return multicut;
    };
};

/*
int main() {
    // Create a graph
    Graph g;

    // Add edges with weights
    g.addEdge(0, 1, 1, 3, 1);
    g.addEdge(0, 2, 1, 5, 2);
    g.addEdge(1, 2, 1, 2, 3);
    g.addEdge(1, 3, 1, 7, 4);
    g.addEdge(0, 1, 2, 2, 5);
    g.addEdge(1, 0, 3, 3, 6);


    // Print the graph with colors
    std::cout << "Finished\n";

    return 0;
}
*/

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
    
    pybind11::class_<LargestPositiveCost>(m, "LargestPositiveCost")
        .def(pybind11::init<>())
        .def("load_graph", &LargestPositiveCost::loadGraph)
        .def("solve", &LargestPositiveCost::solve);

}
