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

  cout << "Start loading" << endl;

  ifstream is (argv[1]);
  matrix_market::reader reader (is);

  csr_matrix_class csr_matrix (reader.matrix ());

  // CPU
  std::unique_ptr<double[]> reference_answer (new double[csr_matrix.meta.cols_count]);
  std::unique_ptr<double[]> x (new double[csr_matrix.meta.cols_count]);

  cout << "Complete loading" << endl;
  auto cpu_naive_time = cpu_csr_spmv_single_thread_naive (csr_matrix, x.get (), reference_answer.get ());
  cout << "CPU: " << cpu_naive_time << endl;

  /// GPU Reusable memory
  resizable_gpu_memory<double> A, x_gpu, y;
  resizable_gpu_memory<unsigned int> col_ids, row_ptr;

  /// GPU
  {
    auto gpu_time = gpu_csr_spmv (csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
    cout << "GPU CSR: " << gpu_time << " (S = " << cpu_naive_time / gpu_time << ")" << endl;
  }

  {
    ell_matrix_class ell_matrix (csr_matrix);
    auto gpu_time = gpu_ell_spmv (ell_matrix, A, col_ids, x_gpu, y, x.get (), reference_answer.get ());
    cout << "GPU ELL: " << gpu_time << " (S = " << cpu_naive_time / gpu_time << ")" << endl;
  }

  return 0;
}