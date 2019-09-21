//
// Created by egi on 9/15/19.
//

#include "matrix_converter.h"

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

ell_matrix_class::ell_matrix_class (csr_matrix_class &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];

    if (elements_in_row > elements_in_rows)
      elements_in_rows = elements_in_row;
  }

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

size_t ell_matrix_class::get_matrix_size () const
{
  return meta.rows_count * elements_in_rows;
}
