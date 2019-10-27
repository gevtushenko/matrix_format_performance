cmake_minimum_required(VERSION 3.14)
project(matrix_format_performance)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
add_subdirectory(external/matrix_reader)
add_subdirectory(external/cpp_itt)
add_subdirectory(external/fmt)
add_subdirectory(common)
add_subdirectory(gpu)
add_subdirectory(cpu)

include_directories(external/json)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} common gpu cpu cpp_itt fmt)
