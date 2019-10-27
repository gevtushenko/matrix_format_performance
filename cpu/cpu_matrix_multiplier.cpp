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

template<typename data_type>
double cpu_csr_spmv_single_thread_naive (
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
  return chrono::duration<double> (end - begin).count ();
}

template<typename data_type>
void cpu_csr_spmv_multi_thread_naive_kernel (
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
double cpu_csr_spmv_multi_thread_naive (
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
  return max_time;
}

#include "vectorclass.h"

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
double cpu_ell_spmv_multi_thread_naive (
    const ell_matrix_class<data_type > &matrix,
    data_type *x,
    data_type *y)
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

#define INSTANCTIATE(TYPE) \
  template double cpu_csr_spmv_single_thread_naive<TYPE> (const csr_matrix_class<TYPE> &matrix, TYPE *x, TYPE *y); \
  template double cpu_csr_spmv_multi_thread_naive<TYPE> (const csr_matrix_class<TYPE> &matrix, TYPE *x, TYPE *y);

INSTANCTIATE(float)
INSTANCTIATE(double)
#undef INSTANTIATE
