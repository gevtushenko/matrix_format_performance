//
// Created by egi on 9/21/19.
//

#include "cpu_matrix_multiplier.h"
#include "matrix_converter.h"

using namespace std;

void cpu_csr_spmv_single_thread_naive (
    const csr_matrix_class &matrix,
    double *x,
    double *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  for (unsigned int row = 0; row < matrix.meta.rows_count; row++)
  {
    const auto row_start = row_ptr[row];
    const auto row_end = row_ptr[row + 1];

    double dot = 0;
    for (auto element = row_start; element < row_end; element++)
      dot += data[element] * x[col_ids[element]];
    y[row] = dot;
  }
}
