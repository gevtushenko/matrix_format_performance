//
// Created by egi on 9/15/19.
//

#include "matrix_converter.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <chrono>

template <typename data_type>
csr_matrix_class<data_type>::csr_matrix_class (const matrix_market::matrix_class &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  data.reset (new data_type [meta.non_zero_count]);
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

template <typename data_type>
size_t csr_matrix_class<data_type>::get_matrix_size () const
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

template <typename data_type>
ell_matrix_class<data_type>::ell_matrix_class (csr_matrix_class<data_type> &matrix)
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
  data.reset (new data_type[elements_count]);
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

template <typename data_type>
ell_matrix_class<data_type>::ell_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int elements_in_row_arg)
  : meta (matrix.meta)
  , elements_in_rows (elements_in_row_arg)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  const unsigned int elements_count = get_matrix_size ();
  data.reset (new data_type[elements_count]);
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

template <typename data_type>
size_t ell_matrix_class<data_type>::get_matrix_size () const
{
  return meta.rows_count * elements_in_rows;
}

template <typename data_type>
coo_matrix_class<data_type>::coo_matrix_class(csr_matrix_class<data_type> &matrix)
  : meta (matrix.meta)
  , elements_count (meta.non_zero_count)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  data.reset (new data_type[meta.non_zero_count]);
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

template <typename data_type>
coo_matrix_class<data_type>::coo_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int element_start)
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

  data.reset (new data_type[get_matrix_size ()]);
  cols.reset (new unsigned int[get_matrix_size ()]);
  rows.reset (new unsigned int[get_matrix_size ()]);

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      if (element - start >= element_start)
      {
        data[id] = matrix.data[element];
        cols[id] = col_ptr[element];
        rows[id] = row;
        id++;
      }
    }
  }
}

template <typename data_type>
size_t coo_matrix_class<data_type>::get_matrix_size () const
{
  return elements_count;
}

class elements_sort
{
  unsigned int *keys_1 {};
  unsigned int *keys_2 {};
public:
  elements_sort (unsigned int *keys_1_arg, unsigned int *keys_2_arg)
    : keys_1 (keys_1_arg)
    , keys_2 (keys_2_arg)
  { }

  bool operator () (unsigned int i, unsigned int j) const
  {
    if (keys_1[i] == keys_1[j])
      return keys_2[i] < keys_2[j];

    return keys_1[i] < keys_1[j];
  }
};

scoo_matrix_class::scoo_matrix_class (coo_matrix_class<double> &matrix)
  : meta (matrix.meta)
{
  const size_t coo_matrix_size = matrix.get_matrix_size ();
  c_index.reset (new unsigned int[coo_matrix_size]);
  r_index.reset (new unsigned int[coo_matrix_size]);
  values.reset (new double[coo_matrix_size]);

  unsigned int *src_cols = matrix.cols.get ();
  unsigned int *src_rows = matrix.rows.get ();

  using namespace std;
  auto begin = chrono::system_clock::now ();

  unique_ptr<unsigned int[]> permutation_indexes (new unsigned int[coo_matrix_size]);
  iota (permutation_indexes.get (), permutation_indexes.get () + coo_matrix_size, 0u);
  sort (permutation_indexes.get (), permutation_indexes.get () + coo_matrix_size, elements_sort (src_cols, src_rows));

  auto end = chrono::system_clock::now ();
  const double elapsed = chrono::duration<double> (end - begin).count ();

  cout << "Sorting data for SCOO took: " << elapsed << "s\n";

  double *src_data = matrix.data.get ();

  for (unsigned int element = 0; element < coo_matrix_size; element++)
  {
    const unsigned int new_position = permutation_indexes[element];
    c_index[new_position] = src_cols[element];
    r_index[new_position] = src_rows[element];
    values[new_position] = src_data[element];
  }
}

size_t scoo_matrix_class::get_matrix_size () const
{
  return 0;
}

template <typename data_type>
hybrid_matrix_class<data_type>::hybrid_matrix_class (csr_matrix_class<data_type> &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");
}

template <typename data_type>
void hybrid_matrix_class<data_type>::allocate(csr_matrix_class<data_type> &matrix, double percent)
{
  const auto row_ptr = matrix.row_ptr.get ();

  auto [_1, max_elements, avg_elements] = get_rows_statistics (meta, row_ptr);
  const unsigned int elements_per_ell = avg_elements + (max_elements - avg_elements) * percent;

  ell_matrix = std::make_unique<ell_matrix_class<data_type>> (matrix, elements_per_ell); /// Don't use more than avg elements in an ELL row
  coo_matrix = std::make_unique<coo_matrix_class<data_type>> (matrix, elements_per_ell); /// Don't use elements before avg elements in an COO row

  std::cout << "ELL elements per row: " << elements_per_ell << "; "
            << "COO elements: " << coo_matrix->get_matrix_size () << "; ELL elements: " << ell_matrix->get_matrix_size () << "; "
            << "COO/ELL Ratio: " << static_cast<double> (coo_matrix->get_matrix_size ()) / ell_matrix->get_matrix_size () << std::endl;
}

template class csr_matrix_class<float>;
template class csr_matrix_class<double>;

template class ell_matrix_class<float>;
template class ell_matrix_class<double>;

template class coo_matrix_class<float>;
template class coo_matrix_class<double>;

template class hybrid_matrix_class<float>;
template class hybrid_matrix_class<double>;
