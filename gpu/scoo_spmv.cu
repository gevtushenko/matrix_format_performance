#include "scoo_spmv.h"

#include <cuda_runtime.h>

template <typename data_type>
__global__ void fill_vector (unsigned int n, data_type *vec, data_type value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    vec[i] = value;
}

template<class T>
struct shared_memory
{
  __device__ inline operator T *()
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template<>
struct shared_memory<double>
{
  __device__ inline operator double *()
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <typename data_type>
__global__ void scoo_spmv_kernel (
    const unsigned int n_rows,
    const unsigned int n_slices,
    const unsigned int slice_size,
    const unsigned int lane_size,
    const unsigned int *c_index,
    const unsigned int *r_index,
    const data_type *values,
    const unsigned int *slices_ptr,
    const data_type *x,
    data_type *y)
{
  const unsigned int thread_lane = threadIdx.x % lane_size;
  const unsigned int row_lane = threadIdx.x / lane_size;

  data_type *cache = shared_memory<data_type> ();
  // __shared__ data_type cache[slice_size][lane_size];

  const unsigned int slice_id = blockIdx.x;
  const unsigned int slice_row_begin = slice_id * slice_size;
  const unsigned int limit = (slice_id != n_slices - 1) ? slice_size : ((n_rows - 1) % slice_size) + 1;

  const unsigned int begin = slices_ptr[slice_id];
  const unsigned int end = slices_ptr[slice_id + 1];

  const unsigned int threads_per_row = blockDim.x / lane_size;

  /// Prepare cache
  for (unsigned int index = row_lane; index < limit; index += threads_per_row)
    cache[index * lane_size + thread_lane] = 0.0;
  __syncthreads ();

  /// Cache data
  for (unsigned int index = begin + threadIdx.x; index < end; index += blockDim.x)
  {
    const unsigned int col = c_index[index];
    const unsigned int row = r_index[index] - slice_id * slice_size;
    const data_type value = values[index];

    atomicAdd (cache + row * lane_size + thread_lane, value * x[col]);
  }
  __syncthreads ();

  if (threads_per_row > 1)
  {
    /// Reduce step 1 - gather values in threads_per_row lanes
    if (row_lane < limit)
    {
      for (unsigned int lane = thread_lane + threads_per_row; lane < lane_size; lane += threads_per_row)
      {
        cache[row_lane * lane_size + thread_lane] += cache[row_lane * lane_size + lane];
      }
    }
    __syncthreads ();

      /// Reduce step 2 - tree reduction
    for (unsigned int s = min (threads_per_row, lane_size) / 2; s > 0; s /= 2)
    {
      if (row_lane < limit)
      {
        if (thread_lane < s)
          cache[row_lane * lane_size + thread_lane] += cache[row_lane * lane_size + thread_lane + s];
      }
      __syncthreads ();
    }

    /// Write reduction results
    if (row_lane < limit)
      if (slice_row_begin + row_lane < n_rows)
        y[slice_row_begin + row_lane] = cache[row_lane * lane_size + 0];
  }
  else
  {
    /// Write results
    for (unsigned int index = threadIdx.x; index < limit; index += blockDim.x)
    {
      data_type sum = 0.0;
      for (unsigned int i = 0; i < lane_size; i++)
        sum += cache[index * lane_size + i];
      y[slice_row_begin + index] = sum;
    }
  }
}

unsigned int find_next_multiple_of (unsigned int number, unsigned int multiple)
{
  if (multiple == 0)
    return number;

  unsigned int remainder = number % multiple;

  if (remainder == 0)
    return number;

  return number + multiple - remainder;
}

template <typename data_type>
measurement_class gpu_scoo_spmv (
    bool print_diff,
    const scoo_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &r_index,
    resizable_gpu_memory<unsigned int> &c_index,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y)
{
  auto &meta = matrix.meta;

  const size_t A_size = matrix.meta.non_zero_count;
  const size_t col_ids_size = matrix.meta.non_zero_count;
  const size_t row_ptr_size = matrix.meta.non_zero_count;
  const size_t x_size = matrix.meta.cols_count;
  const size_t y_size = matrix.meta.rows_count;

  A.resize (A_size);
  c_index.resize (col_ids_size);
  r_index.resize (row_ptr_size);
  x.resize (x_size);
  y.resize (y_size);

  cudaMemcpy (A.get (), matrix.values.get (), A_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (c_index.get (), matrix.c_index.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (r_index.get (), matrix.r_index.get (), row_ptr_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, x.get (), 1.0);

    grid_size.y = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  unsigned int *d_slices_ptr {};
  cudaMalloc (&d_slices_ptr, (matrix.slices_count + 1) * sizeof (unsigned int));
  cudaMemcpy (d_slices_ptr, matrix.slices_ptr.get (), sizeof (unsigned int) * (matrix.slices_count + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  const unsigned int slice_size = matrix.slice_size;
  const unsigned int lane_size = matrix.lane_size;
  dim3 block_size = 1024u;
  dim3 grid_size {};

  grid_size.x = matrix.slices_count;

  cudaDeviceSynchronize ();
  cudaEventRecord (start);
  {
    scoo_spmv_kernel<<<grid_size, block_size, slice_size * lane_size * sizeof (data_type)>>> (
        meta.rows_count, matrix.slices_count, slice_size, lane_size, c_index.get (), r_index.get (), A.get (), d_slices_ptr, x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
  cudaFree (d_slices_ptr);

  if (print_diff)
    compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;

  return measurement_class ("GPU SCOO", elapsed, 0, 0);
}

template
measurement_class gpu_scoo_spmv (
    bool print_diff,
    const scoo_matrix_class<float> &matrix,
    resizable_gpu_memory<float> &A,
    resizable_gpu_memory<unsigned int> &r_index,
    resizable_gpu_memory<unsigned int> &c_index,
    resizable_gpu_memory<float> &x,
    resizable_gpu_memory<float> &y,

    float *reusable_vector,
    const float *reference_y);

template
measurement_class gpu_scoo_spmv (
    bool print_diff,
    const scoo_matrix_class<double> &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &r_index,
    resizable_gpu_memory<unsigned int> &c_index,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y);
