set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")
cmake_minimum_required(VERSION 3.4...3.18)
project(edge_contraction_solver)

add_subdirectory(pybind11)
pybind11_add_module(edge_contraction_solver src/main.cpp)

