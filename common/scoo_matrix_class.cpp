//
// Created by egi on 11/5/19.
//

#include "scoo_matrix_class.h"
#include "matrix_converter.h"

#include <iostream>
#include <algorithm>
#include <numeric>

template <typename data_type>
scoo_matrix_class<data_type>::scoo_matrix_class (csr_matrix_class<data_type> &matrix)
  : meta (matrix.meta)
{
  std::unique_ptr<unsigned int[]> permutation (new unsigned int[meta.rows_count]);
  std::iota (permutation.get (), permutation.get () + meta.rows_count, 0);

  auto row_ptr = matrix.row_ptr.get ();
  auto get_nnz_in_row = [row_ptr] (unsigned int row) { return row_ptr[row + 1] - row_ptr[row]; };

  std::sort (permutation.get (), permutation.get () + meta.rows_count, [&get_nnz_in_row] (const unsigned int &left, const unsigned int &right)
  {
    return get_nnz_in_row (left) > get_nnz_in_row (right);
  });

  /// Copy data
  values = std::make_unique<data_type[]> (meta.non_zero_count);
}

template class scoo_matrix_class<float>;
template class scoo_matrix_class<double>;
