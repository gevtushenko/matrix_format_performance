project(cpu)

set(CPU_SOURCES
        cpu_matrix_multiplier.h
        cpu_matrix_multiplier.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

add_library(cpu ${CPU_SOURCES})
target_include_directories(cpu PUBLIC . ../external/vectorclass/ /opt/intel/mkl/include)
set_target_properties(cpu PROPERTIES CXX_STANDARD 17)
target_link_libraries(cpu common
        /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
        /opt/intel/mkl/lib/intel64/libmkl_intel_thread.so
        /opt/intel/mkl/lib/intel64/libmkl_core.so
        /opt/intel/compilers_and_libraries_2019.5.281/linux/compiler/lib/intel64_lin/libiomp5.so
        pthread -lm -ldl)
