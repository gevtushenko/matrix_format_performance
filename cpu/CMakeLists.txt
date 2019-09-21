project(cpu)

set(CPU_SOURCES
        cpu_matrix_multiplier.h
        cpu_matrix_multiplier.cpp)

add_library(cpu ${CPU_SOURCES})
target_include_directories(cpu PUBLIC . ../external/vectorclass/)
set_target_properties(cpu PROPERTIES CXX_STANDARD 17)
target_link_libraries(cpu common)
