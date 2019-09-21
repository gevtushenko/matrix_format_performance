#include <iostream>
#include <fstream>

#include "matrix_market_reader.h"
#include "resizable_gpu_memory.h"
#include "matrix_converter.h"

#include "gpu_matrix_multiplier.h"
#include "cpu_matrix_multiplier.h"

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " /path/to/mtx" << endl;
    return 1;
  }

  ifstream is (argv[1]);
  matrix_market::reader reader (is);

  csr_matrix_class csr_matrix (reader.matrix ());

  // CPU
  std::unique_ptr<double[]> reference_answer (new double[csr_matrix.meta.cols_count]);
  std::unique_ptr<double[]> x (new double[csr_matrix.meta.cols_count]);

  cpu_csr_spmv_single_thread_naive (csr_matrix, x.get (), reference_answer.get ());

  /// GPU Reusable memory
  resizable_gpu_memory<double> A, x_gpu, y;
  resizable_gpu_memory<unsigned int> col_ids, row_ptr;

  /// GPU
  {
    csr_spmv (csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
  }

  return 0;
}