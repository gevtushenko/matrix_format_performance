//
// Created by egi on 9/15/19.
//

#include "matrix_converter.h"

#include <iostream>
#include <limits>

csr_matrix_class::csr_matrix_class (matrix_market::matrix_class &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  data.reset (new double [meta.non_zero_count]);
  columns.reset (new unsigned int[meta.non_zero_count]);
  row_ptr.reset (new unsigned int[meta.rows_count + 1]);
  std::fill_n (row_ptr.get (), meta.rows_count + 1, 0u);

  auto src_rows = matrix.get_row_ids ();
  auto src_cols = matrix.get_col_ids ();
  auto src_data = matrix.get_dbl_data ();

  for (unsigned int i = 0; i < meta.non_zero_count; i++)
    row_ptr[src_rows[i]]++;

  unsigned int ptr = 0;
  for (unsigned int row = 0; row < meta.rows_count + 1; row++)
  {
    const unsigned int tmp = row_ptr[row];
    row_ptr[row] = ptr;
    ptr += tmp;
  }

  std::unique_ptr<unsigned int[]> row_element_id (new unsigned int[meta.rows_count]);
  std::fill_n (row_element_id.get (), meta.rows_count, 0u);

  for (unsigned int i = 0; i < meta.non_zero_count; i++)
  {
    const unsigned int row = src_rows[i];
    const unsigned int element_offset = row_ptr[row] + row_element_id[row]++;
    data[element_offset] = src_data[i];
    columns[element_offset] = src_cols[i];
  }
}

size_t csr_matrix_class::get_matrix_size () const
{
  return meta.non_zero_count;
}

struct matrix_rows_statistic
{
  unsigned int min_elements_in_rows {};
  unsigned int max_elements_in_rows {};
  unsigned int avg_elements_in_rows {};
};

matrix_rows_statistic get_rows_statistics (
    const matrix_market::matrix_class::matrix_meta &meta,
    const unsigned int *row_ptr)
{
  matrix_rows_statistic statistic {};
  statistic.min_elements_in_rows = std::numeric_limits<unsigned int>::max () - 1;

  unsigned int sum_elements_in_rows = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];

    if (elements_in_row > statistic.max_elements_in_rows)
      statistic.max_elements_in_rows = elements_in_row;

    if (elements_in_row < statistic.min_elements_in_rows)
      statistic.min_elements_in_rows = elements_in_row;

    sum_elements_in_rows += elements_in_row;
  }

  statistic.avg_elements_in_rows = sum_elements_in_rows / meta.rows_count;
  return statistic;
}

ell_matrix_class::ell_matrix_class (csr_matrix_class &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  auto [min_elements, max_elements, avg_elements] = get_rows_statistics (meta, row_ptr);
  elements_in_rows = max_elements;

  std::cout << "ELL: " << elements_in_rows
            << " elements in rows (min: " << min_elements
            << "; avg: " << avg_elements<< ")" << std::endl;

  const unsigned int elements_count = elements_in_rows * meta.rows_count;
  data.reset (new double [elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[row + (element - start) * meta.rows_count] = matrix.data[element];
      columns[row + (element - start) * meta.rows_count] = col_ptr[element];
    }
  }
}

ell_matrix_class::ell_matrix_class (csr_matrix_class &matrix, unsigned int elements_in_row_arg)
  : meta (matrix.meta)
  , elements_in_rows (elements_in_row_arg)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  const unsigned int elements_count = get_matrix_size ();
  data.reset (new double [elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    /// Skip extra elements
    for (auto element = start; element < std::min (start + elements_in_row_arg, end); element++)
    {
      data[row + (element - start) * meta.rows_count] = matrix.data[element];
      columns[row + (element - start) * meta.rows_count] = col_ptr[element];
    }
  }
}

size_t ell_matrix_class::get_matrix_size () const
{
  return meta.rows_count * elements_in_rows;
}

coo_matrix_class::coo_matrix_class(csr_matrix_class &matrix)
  : meta (matrix.meta)
  , elements_count (meta.non_zero_count)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  data.reset (new double [meta.non_zero_count]);
  cols.reset (new unsigned int[meta.non_zero_count]);
  rows.reset (new unsigned int[meta.non_zero_count]);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[id] = matrix.data[element];
      cols[id] = col_ptr[element];
      rows[id] = row;
      id++;
    }
  }
}

coo_matrix_class::coo_matrix_class (csr_matrix_class &matrix, unsigned int element_start)
  : meta (matrix.meta)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
      if (element - start >= element_start)
      elements_count++;
  }

  data.reset (new double [get_matrix_size ()]);
  cols.reset (new unsigned int[get_matrix_size ()]);
  rows.reset (new unsigned int[get_matrix_size ()]);

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      if (element > element_start)
      {
        data[id] = matrix.data[element];
        cols[id] = col_ptr[element];
        rows[id] = row;
        id++;
      }
    }
  }
}

size_t coo_matrix_class::get_matrix_size () const
{
  return meta.non_zero_count;
}

hybrid_matrix_class::hybrid_matrix_class (csr_matrix_class &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  const auto row_ptr = matrix.row_ptr.get ();

  auto [_1, _2, avg_elements] = get_rows_statistics (meta, row_ptr);

  ell_matrix = std::make_unique<ell_matrix_class> (matrix, avg_elements); /// Don't use more than avg elements in an ELL row
  coo_matrix = std::make_unique<coo_matrix_class> (matrix, avg_elements); /// Don't use elements before avg elements in an COO row
}
