set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
cmake_minimum_required(VERSION 3.18)
project(edge_contraction_solver)

add_subdirectory(pybind11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(edge_contraction_solver src/main.cpp)
