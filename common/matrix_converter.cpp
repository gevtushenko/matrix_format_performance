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
