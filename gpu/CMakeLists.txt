project(gpu)

set(GPU_SOURCES
        gpu_matrix_multiplier.h
        gpu_matrix_multiplier.cu
        resizable_gpu_memory.h
        resizable_gpu_memory.cpp)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -gencode arch=compute_75,code=sm_75")
cuda_add_library(gpu ${GPU_SOURCES})
target_include_directories(gpu PUBLIC .)
target_link_libraries(gpu common)
