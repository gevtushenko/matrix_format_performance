cmake_minimum_required(VERSION 3.14)
project(matrix_format_performance)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

add_executable(matrix_format_performance main.cpp)