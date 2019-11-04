project(gpu)
set(CMAKE_CXX_STANDARD 17)

set(GPU_SOURCES
        reduce.cuh
        gpu_matrix_multiplier.h
        gpu_matrix_multiplier.cu
        csr_adaptive_spmv.cu
        csr_adaptive_spmv.h
        resizable_gpu_memory.h
        resizable_gpu_memory.cpp)

find_package(CUDA QUIET)

# set cuda debug flags
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -fmad=false -lineinfo -gencode arch=compute_75,code=sm_75")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -gencode arch=compute_75,code=sm_75")
endif()

# set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

cuda_add_library(gpu ${GPU_SOURCES})
target_include_directories(gpu PUBLIC . ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu common ${CUDA_cusparse_LIBRARY})
