//
// Created by egi on 9/21/19.
//

#include "cpu_matrix_multiplier.h"
#include "matrix_converter.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <array>

#include <immintrin.h>

using namespace std;

template<typename data_type>
measurement_class cpu_csr_spmv_single_thread_naive (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  auto begin = chrono::system_clock::now ();

  for (unsigned int row = 0; row < matrix.meta.rows_count; row++)
  {
    const auto row_start = row_ptr[row];
    const auto row_end = row_ptr[row + 1];

    data_type dot = 0;
    for (auto element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }

  auto end = chrono::system_clock::now ();
  const double elapsed = chrono::duration<double> (end - begin).count ();

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "CPU CSR",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}

#include "mkl_spblas.h"
#include "mkl.h"

measurement_class cpu_csr_spmv_mkl (
    const csr_matrix_class<float> &matrix,
    float *x,
    float *y,
    const float *reference_y)
{
  const auto &meta = matrix.meta;
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  struct matrix_descr descr_A;
  sparse_matrix_t csr_A;
  const float alpha = 1.0, beta = 0.0;

  auto avx2_enabled = mkl_enable_instructions (MKL_ENABLE_AVX2);

  mkl_sparse_s_create_csr (
      &csr_A,
      SPARSE_INDEX_BASE_ZERO, meta.rows_count, meta.cols_count,
      reinterpret_cast<int *> (row_ptr), reinterpret_cast<int *> (row_ptr + 1),
      reinterpret_cast<int *> (col_ids), data);

  descr_A.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_optimize (csr_A);

  auto begin = chrono::system_clock::now ();

  mkl_sparse_s_mv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csr_A, descr_A, x, beta, y);

  auto end = chrono::system_clock::now ();
  const double elapsed = chrono::duration<double> (end - begin).count ();

  compare_results (meta.rows_count, y, reference_y);

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (float);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (float);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (float);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "CPU CSR (mkl)",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}

measurement_class cpu_csr_spmv_mkl (
    const csr_matrix_class<double> &matrix,
    double *x,
    double *y,
    const double *reference_y)
{
  const auto &meta = matrix.meta;
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  struct matrix_descr descr_A;
  sparse_matrix_t csr_A;
  const double alpha = 1.0, beta = 0.0;

  auto avx2_enabled = mkl_enable_instructions (MKL_ENABLE_AVX2);

  mkl_sparse_d_create_csr (
      &csr_A,
      SPARSE_INDEX_BASE_ZERO, meta.rows_count, meta.cols_count,
      reinterpret_cast<int *> (row_ptr), reinterpret_cast<int *> (row_ptr + 1),
      reinterpret_cast<int *> (col_ids), data);

  descr_A.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_optimize (csr_A);

  auto begin = chrono::system_clock::now ();

  mkl_sparse_d_mv (SPARSE_OPERATION_NON_TRANSPOSE, alpha, csr_A, descr_A, x, beta, y);

  auto end = chrono::system_clock::now ();
  const double elapsed = chrono::duration<double> (end - begin).count ();

  compare_results (meta.rows_count, y, reference_y);

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (double);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (double);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (double);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "CPU CSR (mkl)",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}

template<typename data_type>
measurement_class cpu_csr_spmv_single_thread_naive_with_reduce_order (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  auto begin = chrono::system_clock::now ();

  std::array<data_type, 32> dots {};
  dots.fill (0.0);

  for (unsigned int row = 0; row < matrix.meta.rows_count; row++)
  {
    const auto row_start = row_ptr[row];
    const auto row_end = row_ptr[row + 1];

    // calc
    for (unsigned int lane = 0; lane < 32; lane++)
    {
      dots[lane] = 0.0;
      for (unsigned int element = row_start + lane; element < row_end; element += 32)
        dots[lane] += data[element] * x[col_ids[element]];
    }

    // reduce
    for (int offset = 16; offset > 0; offset /= 2)
    {
      for (unsigned int lane = 0; lane < 16; lane++)
      {
        dots[lane] += dots[lane + offset];
      }
    }

    y[row] = dots[0];
  }

  auto end = chrono::system_clock::now ();
  const double elapsed = chrono::duration<double> (end - begin).count ();
  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "CPU CSR Reduce order",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}

template<typename data_type>
measurement_class cpu_csr_spmv_multi_thread_naive_kernel (
    const data_type *x,
    data_type *y,
    const unsigned int *row_ptr,
    const unsigned int *col_ids,
    const data_type *data,
    const unsigned int thread_begin,
    const unsigned int thread_end)
{
  for (unsigned int row = thread_begin; row < thread_end; row++)
  {
    const auto row_start = row_ptr[row];
    const auto row_end = row_ptr[row + 1];

    data_type dot = 0;
    for (auto element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }
}

template<typename data_type>
measurement_class cpu_csr_spmv_multi_thread_naive (
    const csr_matrix_class<data_type > &matrix,
    data_type *x,
    data_type *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  const unsigned int threads_count = thread::hardware_concurrency ();
  unique_ptr<double[]> times (new double[threads_count]);
  vector<thread> threads;

  std::atomic<unsigned int> threads_started;
  threads_started.store (0);

  for (unsigned int thread = 0; thread < threads_count; thread++)
    threads.emplace_back ([&, thread] () {
      const unsigned int rows_per_thread = matrix.meta.rows_count / threads_count;
      const unsigned int thread_begin = rows_per_thread * thread;
      const unsigned int thread_end = thread == threads_count - 1 ? matrix.meta.rows_count : (thread + 1) * rows_per_thread;

      threads_started.fetch_add (1, std::memory_order_relaxed);
      while (threads_started.load (std::memory_order_relaxed) < threads_count)
        _mm_pause ();

      auto begin = chrono::system_clock::now ();
      cpu_csr_spmv_multi_thread_naive_kernel (x, y, row_ptr, col_ids, data, thread_begin, thread_end);
      auto end = chrono::system_clock::now ();

      times[thread] = chrono::duration<double> (end - begin).count ();
    });

  for (auto &thread: threads)
    thread.join ();

  double max_time = 0.0;
  for (unsigned int i = 0; i < threads_count; i++)
    max_time = std::max (max_time, times[i]);

  const double elapsed = max_time;
  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (unsigned int);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (unsigned int);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
      "CPU CSR Parallel",
      elapsed,
      data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
      operations_count);
}

template<typename data_type>
void cpu_ell_spmv_multi_thread_naive_kernel (
    unsigned int n_rows,
    unsigned int elements_in_rows,
    const data_type *x,
    data_type *y,
    const unsigned int *col_ids,
    const data_type *data,
    const unsigned int thread_begin,
    const unsigned int thread_end)
{
  for (unsigned int row = thread_begin; row < thread_end; row++)
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

template<typename data_type>
measurement_class cpu_ell_spmv_multi_thread_naive (
    const ell_matrix_class<data_type > &matrix,
    data_type *x,
    data_type *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  vector<thread> threads;

  auto begin = chrono::system_clock::now ();

  const unsigned int threads_count = thread::hardware_concurrency ();
  for (unsigned int thread = 0; thread < threads_count; thread++)
    threads.emplace_back ([&, thread] () {
      const unsigned int rows_per_thread = matrix.meta.rows_count / threads_count;
      const unsigned int thread_begin = rows_per_thread * thread;
      const unsigned int thread_end = thread == threads_count - 1 ? matrix.meta.rows_count : (thread + 1) * rows_per_thread;

      cpu_ell_spmv_multi_thread_naive_kernel (matrix.meta.rows_count, matrix.elements_in_rows, x, y, col_ids, data, thread_begin, thread_end);
    });

  for (auto &thread: threads)
    thread.join ();

  auto end = chrono::system_clock::now ();
  const double elapsed = chrono::duration<double> (end - begin).count ();

  return measurement_class (
      "TODO",
      elapsed,
      1,
      1);
}

#define INSTANCTIATE(TYPE) \
  template measurement_class cpu_csr_spmv_single_thread_naive<TYPE> (const csr_matrix_class<TYPE> &matrix, TYPE *x, TYPE *y); \
  template measurement_class cpu_csr_spmv_single_thread_naive_with_reduce_order<TYPE> (const csr_matrix_class<TYPE> &matrix, TYPE *x, TYPE *y); \
  template measurement_class cpu_csr_spmv_multi_thread_naive<TYPE> (const csr_matrix_class<TYPE> &matrix, TYPE *x, TYPE *y);

INSTANCTIATE(float)
INSTANCTIATE(double)
#undef INSTANTIATE
