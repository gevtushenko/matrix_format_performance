//
// Created by egi on 11/3/19.
//

#include "csr_adaptive_spmv.h"
#include "reduce.cuh"

#define NNZ_PER_WG 128u ///< Should be equal to warpSize

template <typename data_type>
__global__ void fill_vector (unsigned int n, data_type *vec, data_type value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    vec[i] = value;
}

template <typename data_type>
__global__ void csr_adaptive_spmv_kernel (
    const unsigned int n_rows,
    const unsigned int *col_ids,
    const unsigned int *row_ptr,
    const unsigned int *row_blocks,
    const data_type *data,
    const data_type *x,
    data_type *y)
{
  const unsigned int block_row_begin = row_blocks[blockIdx.x];
  const unsigned int block_row_end = row_blocks[blockIdx.x + 1];
  const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin];

  __shared__ data_type cache[NNZ_PER_WG];

  if (block_row_end - block_row_begin > 1)
  {
    /// CSR-Stream case
    const unsigned int i = threadIdx.x;
    const unsigned int block_data_begin = row_ptr[block_row_begin];
    const unsigned int thread_data_begin = block_data_begin + i;

    if (i < nnz)
      cache[i] = data[thread_data_begin] * x[col_ids[thread_data_begin]];
    __syncthreads ();

    if ((block_row_begin + i) < block_row_end)
    {
      data_type dot = 0.0;

      // TODO Implement reduce
      for (unsigned int j = row_ptr[block_row_begin + i] - block_data_begin;
           j < row_ptr[block_row_begin + i + 1] - block_data_begin;
           j++)
      {
        dot += cache[j];
      }

      y[block_row_begin + i] = dot;
    }
  }
  else
  {
    const unsigned int row = block_row_begin;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;

    data_type dot = 0;

    if (nnz <= 32)
    {
      /// CSR-Vector case
      if (row < n_rows)
      {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];
        unsigned int element = row_start + lane;

        if (element < row_end)
          dot = data[element] * x[col_ids[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0 && warp_id == 0 && row < n_rows)
        y[row] = dot;
    }
    else
    {
      /// CSR-VectorL case
      if (row < n_rows)
      {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
          dot += data[element] * x[col_ids[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0)
        cache[warp_id] = dot;
      __syncthreads ();

      if (warp_id == 0)
      {
        dot = 0.0;

        for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
          dot += cache[element];

        dot = warp_reduce (dot);

        if (lane == 0 && row < n_rows)
          y[row] = dot;
      }
    }
  }
}

unsigned int
fill_row_blocks (
    unsigned int rows_count,
    const unsigned int *row_ptr,
    unsigned int *row_blocks
)
{
  row_blocks[0] = 0;

  int last_i = 0;
  int current_wg = 1;
  unsigned int nnz_sum = 0;
  for (int i = 1; i <= rows_count; i++)
  {
    nnz_sum += row_ptr[i] - row_ptr[i - 1];

    if (nnz_sum == NNZ_PER_WG)
    {
      last_i = i;
      row_blocks[current_wg++] = i;
      nnz_sum = 0;
    }
    else if (nnz_sum > NNZ_PER_WG)
    {
      if (i - last_i > 1)
      {
        row_blocks[current_wg++] = i - 1;
        i--;
      }
      else
      {
        row_blocks[current_wg++] = i;
      }

      last_i = i;
      nnz_sum = 0;
    }
  }

  row_blocks[current_wg] = rows_count;

  return current_wg;
}

template <typename data_type>
measurement_class gpu_csr_adaptive_spmv (
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

    grid_size.y = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, y.get (), 0.0);
  }

  // fill delimiters
  std::unique_ptr<unsigned int[]> row_blocks(new unsigned int[meta.rows_count + 1]);

  const unsigned int blocks_count = fill_row_blocks (meta.rows_count, matrix.row_ptr.get (), row_blocks.get ());

  unsigned int *d_row_blocks {};
  cudaMalloc (&d_row_blocks, (meta.rows_count + 1) * sizeof (unsigned int));
  cudaMemcpy (d_row_blocks, row_blocks.get (), sizeof (unsigned int) * (meta.rows_count + 1), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);
  {
    dim3 block_size = dim3 (NNZ_PER_WG);
    dim3 grid_size {};

    grid_size.x = blocks_count; // (meta.non_zero_count + block_size.x - 1) / block_size.x;

    csr_adaptive_spmv_kernel<<<grid_size, block_size>>> (
        meta.rows_count, col_ids.get (), row_ptr.get (), d_row_blocks, A.get (), x.get (), y.get ());
  }
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);

  cudaMemcpy (reusable_vector, y.get (), y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
  cudaFree (d_row_blocks);

  compare_results (y_size, reusable_vector, reference_y);

  const double elapsed = milliseconds / 1000;

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "GPU CSR-Adaptive",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}


#define INSTANTIATE(data_type)                                                 \
  template measurement_class gpu_csr_adaptive_spmv<data_type>(                 \
      const csr_matrix_class<data_type> &matrix,                               \
      resizable_gpu_memory<data_type> &A,                                      \
      resizable_gpu_memory<unsigned int> &col_ids,                             \
      resizable_gpu_memory<unsigned int> &row_ptr,                             \
      resizable_gpu_memory<data_type> &x, resizable_gpu_memory<data_type> &y,  \
      data_type *reusable_vector, const data_type *reference_y);


INSTANTIATE (float)
INSTANTIATE (double)
#undef INSTANTIATE
