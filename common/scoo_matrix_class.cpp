//
// Created by egi on 11/5/19.
//

#include "scoo_matrix_class.h"
#include "matrix_converter.h"

#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>

template <typename data_type>
unsigned int calculate_slices_count (
    unsigned int n_rows,
    unsigned int sm_count,
    size_t shared_mem_size)
{
  const unsigned int max_rows_per_block = shared_mem_size / sizeof (data_type);

  unsigned int multiplier = 1;

  while (max_rows_per_block < (n_rows / (sm_count * multiplier)))
    multiplier++;

  return sm_count * multiplier;
}

template <typename data_type>
scoo_matrix_class<data_type>::scoo_matrix_class (
    unsigned int sm_count,
    size_t shared_mem_size,
    csr_matrix_class<data_type> &matrix)
  : meta (matrix.meta)
  , slices_count (calculate_slices_count<data_type> (meta.rows_count, sm_count, shared_mem_size))
  , slice_size ((meta.rows_count + slices_count - 1) / slices_count)
  , lane_size (std::min (static_cast<size_t> (16), shared_mem_size / (slice_size * sizeof (data_type))))
{
  auto row_ptr = matrix.row_ptr.get ();

  /// Copy data
  values = std::make_unique<data_type[]> (meta.non_zero_count);
  r_index = std::make_unique<unsigned int[]> (meta.non_zero_count);
  c_index = std::make_unique<unsigned int[]> (meta.non_zero_count);
  slices_ptr = std::make_unique<unsigned int[]> (slices_count + 1);

  size_t offset = 0;

  unsigned int slice_id = 0;
  unsigned int last_slice_row = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    if (row - last_slice_row >= slice_size || slice_id == 0)
    {
      slices_ptr[slice_id++] = offset;
      last_slice_row = row;
    }

    const auto row_begin = row_ptr[row];
    const auto row_end = row_ptr[row + 1];
    const auto n_elements = row_end - row_begin;

    for (unsigned int element = 0; element < n_elements; element++)
    {
      values[offset] = matrix.data[row_begin + element];
      c_index[offset] = matrix.columns[row_begin + element];
      r_index[offset] = row;
      offset++;
    }
  }
  slices_ptr[slice_id] = offset;

  const bool sort_cols = false;

  if (sort_cols)
  {
    std::vector<data_type> tmp_data;
    std::vector<unsigned int> tmp_rows;
    std::vector<unsigned int> tmp_cols;
    std::vector<unsigned int> permutation;

    for (slice_id = 0; slice_id < slices_count; slice_id++)
    {
      const auto slice_begin = slices_ptr[slice_id];
      const auto slice_end = slices_ptr[slice_id + 1];
      const auto elements_in_slice = slice_end - slice_begin;

      tmp_data.resize (elements_in_slice);
      tmp_cols.resize (elements_in_slice);
      tmp_rows.resize (elements_in_slice);
      permutation.resize (elements_in_slice);
      std::iota (permutation.begin (), permutation.end (), 0);
      std::sort (permutation.begin (), permutation.end (), [&] (const unsigned int &l, const unsigned int &r) {
        return c_index[slice_begin + l] < c_index[slice_begin + r];
      });

      std::copy_n (values.get () + slice_begin, elements_in_slice, tmp_data.begin ());
      std::copy_n (r_index.get () + slice_begin, elements_in_slice, tmp_rows.begin ());
      std::copy_n (c_index.get () + slice_begin, elements_in_slice, tmp_cols.begin ());

      for (unsigned int element = slice_begin; element < slice_end; element++)
      {
        const unsigned int local_element = element - slice_begin;
        values[element] = tmp_data[permutation[local_element]];
        r_index[element] = tmp_rows[permutation[local_element]];
        c_index[element] = tmp_cols[permutation[local_element]];
      }
    }
  }
}

template class scoo_matrix_class<float>;
template class scoo_matrix_class<double>;
