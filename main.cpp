#include <iostream>
#include <fstream>

#include "matrix_market_reader.h"
#include "matrix_converter.h"

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

  return 0;
}