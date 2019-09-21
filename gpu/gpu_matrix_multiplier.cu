//
// Created by egi on 9/15/19.
//

#include "gpu_matrix_multiplier.h"
#include "resizable_gpu_memory.h"

#include <cuda_runtime.h>

__global__ void csr_spmv_kernel (
    unsigned int n_rows,
    const unsigned int *col_ids,
    const unsigned int *row_ptr,
    const double *data,
    const double *x,
    double *y)
{
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
  {
    const int row_start = row_ptr[row];
    const int row_end = row_ptr[row + 1];

    double dot = 0;
    for (unsigned int element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }
}

__global__ void fill_vector (unsigned int n, double *vec, double value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    vec[i] = value;
}

#include <iostream>

double csr_spmv (
    const csr_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y)
{

  auto &meta = matrix.meta;

  const size_t A_size = matrix.meta.non_zero_count;
  const size_t col_ids_size = matrix.meta.non_zero_count;
  const size_t row_ptr_size = matrix.meta.rows_count + 1;
  const size_t vec_size = matrix.meta.cols_count;

  A.resize (A_size);
  col_ids.resize (col_ids_size);
  row_ptr.resize (row_ptr_size);
  x.resize (vec_size);
  y.resize (vec_size);

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ptr.get (), matrix.row_ptr.get (), row_ptr_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (vec_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (vec_size, x.get (), 1.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (meta.rows_count + block_size.x - 1) / block_size.x;

    csr_spmv_kernel<<<grid_size, block_size>>> (
        static_cast<unsigned int> (meta.rows_count),
            col_ids.get (), row_ptr.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), vec_size * sizeof (double), cudaMemcpyDeviceToHost);

  constexpr double epsilon = 1e-18;
  for (unsigned int i = 0; i < vec_size; i++)
    if (std::abs (reusable_vector[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << reusable_vector[i] << " != " << reference_y[i] << ")\n";

  return milliseconds / 1000;
}

