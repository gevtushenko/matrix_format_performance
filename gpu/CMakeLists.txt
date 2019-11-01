project(gpu)
set(CMAKE_CXX_STANDARD 17)

set(GPU_SOURCES
        gpu_matrix_multiplier.h
        gpu_matrix_multiplier.cu
        resizable_gpu_memory.h
        resizable_gpu_memory.cpp)

find_package(CUDA QUIET)

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -fmad=false -lineinfo -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -gencode arch=compute_75,code=sm_75")
cuda_add_library(gpu ${GPU_SOURCES})
target_include_directories(gpu PUBLIC . ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu common ${CUDA_cusparse_LIBRARY})
