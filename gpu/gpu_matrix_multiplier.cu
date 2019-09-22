//
// Created by egi on 9/15/19.
//

#include "gpu_matrix_multiplier.h"
#include "resizable_gpu_memory.h"

#include <cuda_runtime.h>

constexpr double epsilon = 1e-2;

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

double gpu_csr_spmv (
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

  const size_t A_size = matrix.get_matrix_size ();
  const size_t col_ids_size = matrix.meta.non_zero_count;
  const size_t row_ptr_size = matrix.meta.rows_count + 1;
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  A.resize (A_size);
  col_ids.resize (col_ids_size);
  row_ptr.resize (row_ptr_size);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ptr.get (), matrix.row_ptr.get (), row_ptr_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (x_size, x.get (), 1.0);
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
        meta.rows_count, col_ids.get (), row_ptr.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (double), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < y_size; i++)
    if (std::abs (reusable_vector[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << reusable_vector[i] << " != " << reference_y[i] << ")\n";

  return milliseconds / 1000;
}


#define FULL_WARP_MASK 0xFFFFFFFF


template <class T>
__device__ T warp_reduce (T val)
{
  /**
   *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
   *  the value of the val variable from the thread at lane X+offset of the same warp.
   *  The data exchange is performed between registers, and more efficient than going
   *  through shared memory, which requires a load, a store and an extra register to
   *  hold the address.
   */
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync (FULL_WARP_MASK, val, offset);

  return val;
}


__global__ void csr_spmv_vector_kernel (
    unsigned int n_rows,
    const unsigned int *col_ids,
    const unsigned int *row_ptr,
    const double *data,
    const double *x,
    double *y)
{
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int lane = thread_id % 32;

  const unsigned int row = warp_id; ///< One warp per row

  double dot = 0;
  if (row < n_rows)
  {
    const unsigned int row_start = row_ptr[row];
    const unsigned int row_end = row_ptr[row + 1];

    for (unsigned int element = row_start + lane; element < row_end; element += 32)
      dot += data[element] * x[col_ids[element]];
  }

  dot = warp_reduce (dot);

  if (lane == 0 && row < n_rows)
  {
    y[row] = dot;
  }
}

double gpu_csr_vector_spmv (
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

  const size_t A_size = matrix.get_matrix_size ();
  const size_t col_ids_size = matrix.meta.non_zero_count;
  const size_t row_ptr_size = matrix.meta.rows_count + 1;
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  A.resize (A_size);
  col_ids.resize (col_ids_size);
  row_ptr.resize (row_ptr_size);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ptr.get (), matrix.row_ptr.get (), row_ptr_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (y_size, y.get (), 24.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * meta.rows_count + block_size.x - 1) / block_size.x;

    csr_spmv_vector_kernel<<<grid_size, block_size>>> (
        meta.rows_count, col_ids.get (), row_ptr.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (double), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < y_size; i++)
    if (std::abs (reusable_vector[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << reusable_vector[i] << " != " << reference_y[i] << ")\n";

  return milliseconds / 1000;
}


__global__ void ell_spmv_kernel (
    unsigned int n_rows,
    unsigned int elements_in_rows,
    const unsigned int *col_ids,
    const double *data,
    const double *x,
    double *y)
{
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
  {
    double dot = 0;
    for (unsigned int element = 0; element < elements_in_rows; element++)
    {
      const unsigned int element_offset = row + element * n_rows;
      dot += data[element_offset] * x[col_ids[element_offset]];
    }
    y[row] = dot;
  }
}

double gpu_ell_spmv (
    const ell_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y)
{

  auto &meta = matrix.meta;

  const size_t A_size = matrix.get_matrix_size ();
  const size_t col_ids_size = A_size;
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.cols_count;

  A.resize (A_size);
  col_ids.resize (col_ids_size);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (meta.rows_count + block_size.x - 1) / block_size.x;

    ell_spmv_kernel<<<grid_size, block_size>>> (
        meta.rows_count, matrix.elements_in_rows, col_ids.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (double), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < y_size; i++)
    if (std::abs (reusable_vector[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << reusable_vector[i] << " != " << reference_y[i] << ")\n";

  return milliseconds / 1000;
}


__global__ void coo_spmv_kernel (
    unsigned int n_elements,
    const unsigned int *col_ids,
    const unsigned int *row_ids,
    const double *data,
    const double *x,
    double *y)
{
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;

  if (element < n_elements)
  {
    const double dot = data[element] * x[col_ids[element]];
    atomicAdd (y + row_ids[element], dot);
  }
}

double gpu_coo_spmv (
    const coo_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y)
{

  auto &meta = matrix.meta;

  const size_t n_elements = matrix.get_matrix_size ();
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.cols_count;

  A.resize (n_elements);
  col_ids.resize (n_elements);
  row_ids.resize (n_elements);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.data.get (), n_elements * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.cols.get (), n_elements * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ids.get (), matrix.rows.get (), n_elements * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (n_elements + block_size.x - 1) / block_size.x;

    coo_spmv_kernel<<<grid_size, block_size>>> (
        n_elements, col_ids.get (), row_ids.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (double), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < y_size; i++)
    if (std::abs (reusable_vector[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << reusable_vector[i] << " != " << reference_y[i] << ")\n";

  return milliseconds / 1000;
}

/*
 * https://github.com/danghvu/cudaSpmv/blob/master/matrix/sliced_coo_kernel.h
 *
 *
 * template <typename ValueType, const uint32_t THREADS_PER_BLOCK, const uint32_t NUM_ROWS_PER_SLICE, const uint32_t LANE_SIZE>
    __global__ void
sliced_coo_kernel_32(
        const uint32_t num_rows,
        const uint32_t numPacks,
        const uint32_t * cols,
        const uint16_t * rows,
        const ValueType * V,
        const uint32_t * offsets,
        const ValueType * __restrict x,
        ValueType * y)
{
    const int thread_lane = threadIdx.x & (LANE_SIZE-1);
    const int row_lane = threadIdx.x/(LANE_SIZE);

    __shared__ ValueType sdata[NUM_ROWS_PER_SLICE][LANE_SIZE];

    const uint32_t packNo=blockIdx.x;
    const uint32_t limit = ( (packNo==numPacks-1)?((num_rows-1)%NUM_ROWS_PER_SLICE)+1:NUM_ROWS_PER_SLICE );

    const uint32_t begin = offsets[packNo];
    const uint32_t end = offsets[packNo+1];
    int index;

    for(index=row_lane; index<limit; index+=THREADS_PER_BLOCK/LANE_SIZE){
        sdata[index][thread_lane] = 0;
    }

    __syncthreads();

    for(index=begin+threadIdx.x; index<end; index+=THREADS_PER_BLOCK){
        const uint32_t col = cols[index];
        const uint16_t row = rows[index];
        const ValueType value = V[index];

#if __CUDA_ARCH__ >= 300 and 0
        const ValueType input = x[col] * value; //try to use constant cache
#else
#if __CUDA_ARCH__ >= 100
#warning "use texture"
#endif
        const ValueType input = fetch_x(col, x) * value;
#endif
        atomic_add(&sdata[row][thread_lane], input);
        //atomic_add(&sdata[row][thread_lane], input);
    }

    __syncthreads();

    for (index=row_lane; index<limit; index+=THREADS_PER_BLOCK/LANE_SIZE) {
        volatile ValueType *psdata = sdata[index];
        int tid = (thread_lane+index) & (LANE_SIZE - 1);

        if (LANE_SIZE>128 && thread_lane<128) psdata[tid]+=psdata[(tid+128) & (LANE_SIZE-1)]; __syncthreads();
        if (LANE_SIZE>64 && thread_lane<64) psdata[tid]+=psdata[(tid+64) & (LANE_SIZE-1)]; __syncthreads();
        if (LANE_SIZE>32 && thread_lane<32) psdata[tid]+=psdata[(tid+32) & (LANE_SIZE-1)]; __syncthreads();

        if (LANE_SIZE>16 && thread_lane<16) psdata[tid]+=psdata[( tid+16 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>8 && thread_lane<8) psdata[tid]+=psdata[( tid+8 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>4 && thread_lane<4) psdata[tid]+=psdata[( tid+4 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>2 && thread_lane<2) psdata[tid]+=psdata[( tid+2 ) & (LANE_SIZE-1)];
        if (LANE_SIZE>1 && thread_lane<1) psdata[tid]+=psdata[( tid+1 ) & (LANE_SIZE-1)];
    }

    __syncthreads();
    const uint32_t actualRow = packNo * NUM_ROWS_PER_SLICE;

    for(index = threadIdx.x; index < limit; index+=THREADS_PER_BLOCK){
        y[actualRow+index] = sdata[index][thread_lane];
    }
}
 */
__global__ void coo_spmv_privatization_kernel (
    unsigned int n_elements,
    const unsigned int *col_ids,
    const unsigned int *row_ids,
    const double *data,
    const double *x,
    double *y)
{
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double cache[];

  if (element < n_elements)
  {
    const double dot = data[element] * x[col_ids[element]];
    atomicAdd (y + row_ids[element], dot);
  }
}

double gpu_coo_privatization_spmv (
    const coo_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y)
{

  auto &meta = matrix.meta;

  const size_t n_elements = matrix.get_matrix_size ();
  const size_t vec_size = matrix.meta.cols_count;

  A.resize (n_elements);
  col_ids.resize (n_elements);
  row_ids.resize (n_elements);
  x.resize (vec_size);
  y.resize (vec_size);

  cudaMemcpy (A.get (), matrix.data.get (), n_elements * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.cols.get (), n_elements * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ids.get (), matrix.rows.get (), n_elements * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (vec_size + block_size.x - 1) / block_size.x;
    fill_vector<<<grid_size, block_size>>> (vec_size, x.get (), 1.0);
    fill_vector<<<grid_size, block_size>>> (vec_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (n_elements + block_size.x - 1) / block_size.x;

    coo_spmv_privatization_kernel<<<grid_size, block_size>>> (
        n_elements, col_ids.get (), row_ids.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), vec_size * sizeof (double), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < vec_size; i++)
    if (std::abs (reusable_vector[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << reusable_vector[i] << " != " << reference_y[i] << ")\n";

  return milliseconds / 1000;
}
