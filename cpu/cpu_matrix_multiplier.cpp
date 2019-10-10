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

#include <immintrin.h>

using namespace std;

double cpu_csr_spmv_single_thread_naive (
    const csr_matrix_class &matrix,
    double *x,
    double *y)
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

    double dot = 0;
    for (auto element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }

  auto end = chrono::system_clock::now ();
  return chrono::duration<double> (end - begin).count ();
}

void cpu_csr_spmv_multi_thread_naive_kernel (
    const double *x,
    double *y,
    const unsigned int *row_ptr,
    const unsigned int *col_ids,
    const double *data,
    const unsigned int thread_begin,
    const unsigned int thread_end)
{
  for (unsigned int row = thread_begin; row < thread_end; row++)
  {
    const auto row_start = row_ptr[row];
    const auto row_end = row_ptr[row + 1];

    double dot = 0;
    for (auto element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }
}

double cpu_csr_spmv_multi_thread_naive (
    const csr_matrix_class &matrix,
    double *x,
    double *y)
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
  return max_time;
}

#include "vectorclass.h"

void cpu_ell_spmv_multi_thread_naive_kernel (
    unsigned int n_rows,
    unsigned int elements_in_rows,
    const double *x,
    double *y,
    const unsigned int *col_ids,
    const double *data,
    const unsigned int thread_begin,
    const unsigned int thread_end)
{
  for (unsigned int row = thread_begin; row < thread_end; row++)
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

double cpu_ell_spmv_multi_thread_naive (
    const ell_matrix_class &matrix,
    double *x,
    double *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

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
      cpu_ell_spmv_multi_thread_naive_kernel (matrix.meta.rows_count, matrix.elements_in_rows, x, y, col_ids, data, thread_begin, thread_end);
      auto end = chrono::system_clock::now ();

      times[thread] = chrono::duration<double> (end - begin).count ();
    });

  for (auto &thread: threads)
    thread.join ();

  double max_time = 0.0;
  for (unsigned int i = 0; i < threads_count; i++)
    max_time = std::max (max_time, times[i]);
  return max_time;
}

void cpu_ell_spmv_multi_thread_avx2_kernel (
    unsigned int n_rows,
    unsigned int elements_in_rows,
    const double *x,
    double *y,
    const uint64_t *col_ids,
    const double *data,
    const unsigned int thread_begin,
    const unsigned int thread_end)
{
  const auto vector_size = 4;
  const auto regular_part = (thread_end / vector_size) * vector_size;

  unsigned int row = thread_begin;
  for (; row < regular_part; row += vector_size)
  {
    Vec4d vec_dot (0.0);

    for (unsigned int element = 0; element < elements_in_rows; element++)
    {
      const unsigned int element_offset = row + element * n_rows;

      Vec4d vec_data;
      vec_data.load (data + element_offset);

      Vec4uq vec_cols;
      vec_cols.load (col_ids + element_offset);

      Vec4d vec_x;
      vec_x = lookup<1024> (vec_cols, x);

      vec_dot += vec_data * vec_x;
    }

    vec_dot.store (y + row);
  }

  for (; row < thread_end; row++)
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

double cpu_ell_spmv_multi_thread_avx2 (
    const ell_matrix_class &matrix,
    double *x,
    double *y,
    const double *reference_y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  const unsigned int threads_count = thread::hardware_concurrency ();
  unique_ptr<double[]> times (new double[threads_count]);
  vector<thread> threads;

  unique_ptr<uint64_t[]> large_col_ids (new uint64_t[matrix.get_matrix_size ()]);
  copy_n (col_ids, matrix.get_matrix_size (), large_col_ids.get ());

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
      cpu_ell_spmv_multi_thread_avx2_kernel (matrix.meta.rows_count, matrix.elements_in_rows, x, y, large_col_ids.get (), data, thread_begin, thread_end);
      auto end = chrono::system_clock::now ();

      times[thread] = chrono::duration<double> (end - begin).count ();
    });

  for (auto &thread: threads)
    thread.join ();

  constexpr double epsilon = 1e-12;
  for (unsigned int i = 0; i < matrix.meta.rows_count; i++)
    if (std::abs (y[i] - reference_y[i]) > epsilon)
      std::cout << "Y'[" << i << "] != Y[" << i << "] (" << y[i] << " != " << reference_y[i] << ")\n";

  double max_time = 0.0;
  for (unsigned int i = 0; i < threads_count; i++)
    max_time = std::max (max_time, times[i]);
  return max_time;
}