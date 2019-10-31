//
// Created by egi on 9/15/19.
//

#include "gpu_matrix_multiplier.h"
#include "resizable_gpu_memory.h"

#include <cuda_runtime.h>
#include <iostream>
#include <limits>

template <typename data_type>
void compare_results (unsigned int y_size, const data_type *a, const data_type *b)
{
  data_type numerator = 0.0;
  data_type denumerator = 0.0;

  for (unsigned int i = 0; i < y_size; i++)
  {
    numerator += (a[i] - b[i]) * (a[i] - b[i]);
    denumerator += b[i] * b[i];
  }

  const data_type error = numerator / denumerator;

  if (error > 1e-9)
  {
    std::cerr << "ERROR: " << error << std::endl;

    for (unsigned int i = 0; i < y_size; i++)
    {
      if (std::abs (a[i] - b[i]) > 1e-8)
      {
        std::cout << "a[i] = " << a[i] << "; b[i] = " << b[i] << std::endl;
        break;
      }
    }
  }
}

template <typename data_type>
__global__ void csr_spmv_kernel (
    unsigned int n_rows,
    const unsigned int *col_ids,
    const unsigned int *row_ptr,
    const data_type *data,
    const data_type *x,
    data_type *y)
{
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
  {
    const int row_start = row_ptr[row];
    const int row_end = row_ptr[row + 1];

    data_type dot = 0;
    for (unsigned int element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }
}

template <typename data_type>
__global__ void fill_vector (unsigned int n, data_type *vec, data_type value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    vec[i] = value;
}

template <typename data_type>
measurement_class gpu_csr_spmv (
    const csr_matrix_class<data_type > &matrix,
    resizable_gpu_memory<data_type > &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
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

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ptr.get (), matrix.row_ptr.get (), row_ptr_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
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

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "GPU CSR",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
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


template <typename data_type>
__global__ void csr_spmv_vector_kernel (
    unsigned int n_rows,
    const unsigned int *col_ids,
    const unsigned int *row_ptr,
    const data_type*data,
    const data_type*x,
    data_type*y)
{
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int lane = thread_id % 32;

  const unsigned int row = warp_id; ///< One warp per row

  data_type dot = 0;
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

template <typename data_type>
measurement_class gpu_csr_vector_spmv (
    const csr_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
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

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ptr.get (), matrix.row_ptr.get (), row_ptr_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 24.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * meta.rows_count + block_size.x - 1) / block_size.x;

    csr_spmv_vector_kernel<data_type><<<grid_size, block_size>>> (
        meta.rows_count, col_ids.get (), row_ptr.get (), A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "GPU CSR (vector)",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}


template <typename data_type>
__global__ void ell_spmv_kernel (
    unsigned int n_rows,
    unsigned int elements_in_rows,
    const unsigned int *col_ids,
    const data_type*data,
    const data_type*x,
    data_type*y)
{
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
  {
    data_type dot = 0;
    for (unsigned int element = 0; element < elements_in_rows; element++)
    {
      const unsigned int element_offset = row + element * n_rows;
      dot += data[element_offset] * x[col_ids[element_offset]];
    }
    y[row] = dot;
  }
}

template <typename data_type>
measurement_class gpu_ell_spmv (
    const ell_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
{
  auto &meta = matrix.meta;

  const size_t A_size = matrix.get_matrix_size ();
  const size_t col_ids_size = A_size;
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  A.resize (A_size);
  col_ids.resize (col_ids_size);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
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

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;

  const unsigned int n_elements = matrix.elements_in_rows * matrix.meta.rows_count;
  const size_t data_bytes = n_elements * sizeof (data_type);
  const size_t x_bytes = n_elements * sizeof (data_type);
  const size_t col_ids_bytes = n_elements * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = n_elements * 2; // + and * per element

  return measurement_class (
      "GPU ELL",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + y_bytes,
      operations_count);
}


template <typename data_type>
__global__ void coo_spmv_kernel (
    unsigned int n_elements,
    const unsigned int *col_ids,
    const unsigned int *row_ids,
    const data_type*data,
    const data_type*x,
    data_type*y)
{
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;

  if (element < n_elements)
  {
    const data_type dot = data[element] * x[col_ids[element]];
    atomicAdd (y + row_ids[element], dot);
  }
}

template <typename data_type>
measurement_class gpu_coo_spmv (
    const coo_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
{

  auto &meta = matrix.meta;

  const size_t n_elements = matrix.get_matrix_size ();
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  A.resize (n_elements);
  col_ids.resize (n_elements);
  row_ids.resize (n_elements);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.data.get (), n_elements * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (col_ids.get (), matrix.cols.get (), n_elements * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (row_ids.get (), matrix.rows.get (), n_elements * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
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

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.non_zero_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element
  return measurement_class (
      "GPU COO",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
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

template <typename data_type>
measurement_class gpu_hybrid_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A_ell,
    resizable_gpu_memory<data_type> &A_coo,
    resizable_gpu_memory<unsigned int> &ell_col_ids,
    resizable_gpu_memory<unsigned int> &coo_col_ids,
    resizable_gpu_memory<unsigned int> &coo_row_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
{
  auto &meta = matrix.meta;

  {
    const size_t A_size = matrix.ell_matrix->get_matrix_size ();
    const size_t col_ids_size = A_size;

    A_ell.resize (A_size);
    ell_col_ids.resize (col_ids_size);

    cudaMemcpyAsync (A_ell.get (), matrix.ell_matrix->data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (ell_col_ids.get (), matrix.ell_matrix->columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  }

  {
    const size_t A_size = matrix.coo_matrix->get_matrix_size ();

    A_coo.resize (A_size);
    coo_col_ids.resize (A_size);
    coo_row_ids.resize (A_size);

    cudaMemcpy (A_coo.get (), matrix.coo_matrix->data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
    cudaMemcpy (coo_col_ids.get (), matrix.coo_matrix->cols.get (), A_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy (coo_row_ids.get (), matrix.coo_matrix->rows.get (), A_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  }

  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  x.resize (x_size);
  y.resize (y_size);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  /// ELL Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (meta.rows_count + block_size.x - 1) / block_size.x;

    ell_spmv_kernel<<<grid_size, block_size>>> (
        meta.rows_count, matrix.ell_matrix->elements_in_rows, ell_col_ids.get (), A_ell.get (), x.get (), y.get ());
  }

  /// COO Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    const auto n_elements = matrix.coo_matrix->get_matrix_size ();
    grid_size.x = (n_elements + block_size.x - 1) / block_size.x;

    coo_spmv_kernel<<<grid_size, block_size>>> (
        n_elements, coo_col_ids.get (), coo_row_ids.get (), A_coo.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;
  return measurement_class (
      "GPU Hybrid (TODO)",
      elapsed,
      1,
      1);
}

template <typename data_type>
__global__ void hybrid_spmv_kernel (
    unsigned int n_rows,
    unsigned int n_elements,
    unsigned int elements_in_rows,
    const unsigned int *ell_col_ids,
    const unsigned int *col_ids,
    const unsigned int *row_ids,
    const data_type*ell_data,
    const data_type*coo_data,
    const data_type*x,
    data_type*y)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_rows)
  {
    const unsigned int row = idx;

    data_type dot = 0;
    for (unsigned int element = 0; element < elements_in_rows; element++)
    {
      const unsigned int element_offset = row + element * n_rows;
      dot += ell_data[element_offset] * x[ell_col_ids[element_offset]];
    }
    atomicAdd (y + row, dot);
  }

  for (unsigned int element = idx; element < n_elements; element += blockDim.x * gridDim.x)
  {
    const data_type dot = coo_data[element] * x[col_ids[element]];
    atomicAdd (y + row_ids[element], dot);
  }
}

template <typename data_type>
measurement_class gpu_hybrid_atomic_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A_ell,
    resizable_gpu_memory<data_type> &A_coo,
    resizable_gpu_memory<unsigned int> &ell_col_ids,
    resizable_gpu_memory<unsigned int> &coo_col_ids,
    resizable_gpu_memory<unsigned int> &coo_row_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
{
  auto &meta = matrix.meta;

  {
    const size_t A_size = matrix.ell_matrix->get_matrix_size ();
    const size_t col_ids_size = A_size;

    A_ell.resize (A_size);
    ell_col_ids.resize (col_ids_size);

    cudaMemcpyAsync (A_ell.get (), matrix.ell_matrix->data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
    cudaMemcpyAsync (ell_col_ids.get (), matrix.ell_matrix->columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  }

  {
    const size_t A_size = matrix.coo_matrix->get_matrix_size ();

    A_coo.resize (A_size);
    coo_col_ids.resize (A_size);
    coo_row_ids.resize (A_size);

    cudaMemcpy (A_coo.get (), matrix.coo_matrix->data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
    cudaMemcpy (coo_col_ids.get (), matrix.coo_matrix->cols.get (), A_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy (coo_row_ids.get (), matrix.coo_matrix->rows.get (), A_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  }

  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  x.resize (x_size);
  y.resize (y_size);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  {
    const unsigned int n_elements = matrix.coo_matrix->get_matrix_size ();
    const unsigned int n_rows = meta.rows_count;

    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (n_rows + block_size.x - 1) / block_size.x;

    hybrid_spmv_kernel<<<grid_size, block_size>>> (
        meta.rows_count, n_elements, matrix.ell_matrix->elements_in_rows, ell_col_ids.get (), coo_col_ids.get (), coo_row_ids.get (), A_ell.get (), A_coo.get (), x.get (), y.get ());
  }

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;
  return measurement_class (
      "GPU Hybrid (atomic TODO)",
      elapsed,
      1,
      1);
}

/// Perform y = y + x
template <typename data_type>
__global__ void vec_add (
  unsigned int n_rows,
  const data_type*x,
  data_type*y)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_rows)
    y[idx] += x[idx];
}

template <typename data_type>
measurement_class gpu_hybrid_cpu_coo_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A_ell,
    resizable_gpu_memory<unsigned int> &ell_col_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    resizable_gpu_memory<data_type> &tmp,

    data_type*cpu_y,
    data_type*reusable_vector,
    const data_type*reference_y)
{
  using namespace std;
  auto &meta = matrix.meta;

  const size_t A_size = matrix.ell_matrix->get_matrix_size ();
  const size_t col_ids_size = A_size;
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  A_ell.resize (A_size);
  ell_col_ids.resize (col_ids_size);
  x.resize (x_size);
  y.resize (y_size);
  tmp.resize (y_size);

  std::unique_ptr<data_type[]> cpu_x (new data_type[x_size]);
  std::fill_n (cpu_x.get (), x_size, 1.0);
  std::fill_n (cpu_y, y_size, 0.0);

  cudaMemcpy (A_ell.get (), matrix.ell_matrix->data.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (ell_col_ids.get (), matrix.ell_matrix->columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  cudaStream_t ell_mult_stream, coo_send_stream;
  cudaStreamCreate (&ell_mult_stream);
  cudaStreamCreate (&coo_send_stream);

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  /// ELL Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (meta.rows_count + block_size.x - 1) / block_size.x;

    ell_spmv_kernel<<<grid_size, block_size, 0, ell_mult_stream>>> (
        meta.rows_count, matrix.ell_matrix->elements_in_rows, ell_col_ids.get (), A_ell.get (), x.get (), y.get ());
  }

  /// COO Part
  {
    const auto coo_col_ids = matrix.coo_matrix->cols.get ();
    const auto coo_row_ids = matrix.coo_matrix->rows.get ();
    const auto coo_data = matrix.coo_matrix->data.get ();

    for (unsigned int element = 0; element < matrix.coo_matrix->get_matrix_size (); element++)
      cpu_y[coo_row_ids[element]] += coo_data[element] * cpu_x[coo_col_ids[element]];

    cudaMemcpyAsync (tmp.get (), cpu_y, y_size * sizeof (data_type), cudaMemcpyHostToDevice, coo_send_stream);

    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;

    vec_add<<<grid_size, block_size>>> (y_size, tmp.get (), y.get ());
  }

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaStreamDestroy (ell_mult_stream);
  cudaStreamDestroy (coo_send_stream);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;
  return measurement_class (
      "GPU Hybrid (CPU COO TODO)",
      elapsed,
      1,
      1);
}

#define INSTANTIATE(data_type)                                                 \
  template measurement_class gpu_csr_spmv<data_type>(                          \
      const csr_matrix_class<data_type> &matrix,                               \
      resizable_gpu_memory<data_type> &A,                                      \
      resizable_gpu_memory<unsigned int> &col_ids,                             \
      resizable_gpu_memory<unsigned int> &row_ptr,                             \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);               \
  template measurement_class gpu_csr_vector_spmv<data_type>(                   \
      const csr_matrix_class<data_type> &matrix,                               \
      resizable_gpu_memory<data_type> &A,                                      \
      resizable_gpu_memory<unsigned int> &col_ids,                             \
      resizable_gpu_memory<unsigned int> &row_ptr,                             \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);               \
  template measurement_class gpu_ell_spmv<data_type>(                          \
      const ell_matrix_class<data_type> &matrix,                               \
      resizable_gpu_memory<data_type> &A,                                      \
      resizable_gpu_memory<unsigned int> &col_ids,                             \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);               \
  template measurement_class gpu_coo_spmv<data_type>(                          \
      const coo_matrix_class<data_type> &matrix,                               \
      resizable_gpu_memory<data_type> &A,                                      \
      resizable_gpu_memory<unsigned int> &col_ids,                             \
      resizable_gpu_memory<unsigned int> &row_ids,                             \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);               \
  template measurement_class gpu_hybrid_spmv<data_type>(                       \
      const hybrid_matrix_class<data_type> &matrix,                            \
      resizable_gpu_memory<data_type> &A_ell,                                  \
      resizable_gpu_memory<data_type> &A_coo,                                  \
      resizable_gpu_memory<unsigned int> &ell_col_ids,                         \
      resizable_gpu_memory<unsigned int> &coo_col_ids,                         \
      resizable_gpu_memory<unsigned int> &coo_row_ids,                         \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);               \
  template measurement_class gpu_hybrid_atomic_spmv<data_type>(                \
      const hybrid_matrix_class<data_type> &matrix,                            \
      resizable_gpu_memory<data_type> &A_ell,                                  \
      resizable_gpu_memory<data_type> &A_coo,                                  \
      resizable_gpu_memory<unsigned int> &ell_col_ids,                         \
      resizable_gpu_memory<unsigned int> &coo_col_ids,                         \
      resizable_gpu_memory<unsigned int> &coo_row_ids,                         \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);               \
  template measurement_class gpu_hybrid_cpu_coo_spmv<data_type>(               \
      const hybrid_matrix_class<data_type> &matrix,                            \
      resizable_gpu_memory<data_type> &A_ell,                                  \
      resizable_gpu_memory<unsigned int> &ell_col_ids,                         \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      resizable_gpu_memory<data_type> &tmp, data_type *cpu_y,                  \
      data_type *reusable_vector, const data_type *reference_y);

INSTANTIATE (float)
INSTANTIATE (double)
#undef INSTANTIATE
