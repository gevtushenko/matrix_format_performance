#include <iostream>
#include <fstream>

#include "matrix_market_reader.h"
#include "resizable_gpu_memory.h"
#include "matrix_converter.h"

#include "gpu_matrix_multiplier.h"

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

  resizable_gpu_memory<double> A, x, y;
  resizable_gpu_memory<unsigned int> col_ids, row_ptr;

  csr_spmv (csr_matrix, A, col_ids, row_ptr, x, y);

  return 0;
}