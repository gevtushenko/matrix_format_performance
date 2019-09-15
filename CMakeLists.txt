cmake_minimum_required(VERSION 3.14)
project(matrix_format_performance)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
add_subdirectory(external/matrix_reader)
add_subdirectory(common)
add_subdirectory(gpu)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} common)

target_link_libraries(${PROJECT_NAME} gpu)
