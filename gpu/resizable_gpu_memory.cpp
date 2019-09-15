//
// Created by egi on 9/15/19.
//

#include "resizable_gpu_memory.h"

#include <cuda_runtime.h>

void internal_free (char *& data)
{
  if (data)
    cudaFree (data);
  data = nullptr;
}

void internal_resize (char *& data, size_t new_size)
{
  internal_free (data);
  cudaMalloc (&data, new_size);
}
