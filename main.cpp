#include <iostream>
#include <fstream>

#include "matrix_market_reader.h"
#include "resizable_gpu_memory.h"
#include "matrix_converter.h"

#include "gpu_matrix_multiplier.h"
#include "cpu_matrix_multiplier.h"

#include "cpp_itt.h"

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " /path/to/mtx" << endl;
    return 1;
  }

  cout << "Start loading" << endl;

  unique_ptr<csr_matrix_class> csr_matrix;
  unique_ptr<ell_matrix_class> ell_matrix;
  unique_ptr<coo_matrix_class> coo_matrix;

  {
    cpp_itt::quiet_region region;

    ifstream is (argv[1]);
    matrix_market::reader reader (is);
    cout << "Complete loading" << endl;

    csr_matrix = make_unique<csr_matrix_class> (reader.matrix ());
    cout << "Complete converting to CSR" << endl;

    ell_matrix = make_unique<ell_matrix_class> (*csr_matrix);
    cout << "Complete converting to ELL" << endl;

    coo_matrix = make_unique<coo_matrix_class> (*csr_matrix);
    cout << "Complete converting to COO" << endl;
  }

  // CPU
  std::unique_ptr<double[]> reference_answer (new double[csr_matrix->meta.cols_count]);
  std::unique_ptr<double[]> cpu_y (new double[csr_matrix->meta.rows_count]);
  std::unique_ptr<double[]> x (new double[csr_matrix->meta.cols_count]);

  double cpu_naive_time {};

  {
    auto duration = cpp_itt::create_event_duration ("cpu_csr_spmv_single_thread_naive");
    cpu_naive_time = cpu_csr_spmv_single_thread_naive (*csr_matrix, x.get (), reference_answer.get ());
    cout << "CPU: " << cpu_naive_time << endl;
  }

  double cpu_parallel_naive_time {};

  {
    auto duration = cpp_itt::create_event_duration ("cpu_csr_spmv_multi_thread_naive");
    cpu_parallel_naive_time = cpu_csr_spmv_multi_thread_naive (*csr_matrix, x.get (), cpu_y.get ());
    cout << "CPU Parallel: " << cpu_parallel_naive_time << " (SSCPU = " << cpu_naive_time / cpu_parallel_naive_time << ")" << endl;
  }

  if (0)
  {
    auto duration = cpp_itt::create_event_duration ("cpu_ell_spmv_multi_thread_naive");
    auto cpu_parallel_naive_ell_time = cpu_ell_spmv_multi_thread_naive (*ell_matrix, x.get (), cpu_y.get ());
    cout << "CPU Parallel ELL: " << cpu_parallel_naive_ell_time << " (SSCPU = " << cpu_naive_time / cpu_parallel_naive_ell_time << ")" << endl;
  }

  if (0)
  {
    auto duration = cpp_itt::create_event_duration ("cpu_ell_spmv_multi_thread_naive");
    auto cpu_parallel_naive_ell_time = cpu_ell_spmv_multi_thread_avx2 (*ell_matrix, x.get (), cpu_y.get (), reference_answer.get ());
    cout << "CPU Parallel ELL (AVX2): " << cpu_parallel_naive_ell_time << " (SSCPU = " << cpu_naive_time / cpu_parallel_naive_ell_time << ")" << endl;
  }

  /// GPU Reusable memory
  resizable_gpu_memory<double> A, x_gpu, y;
  resizable_gpu_memory<unsigned int> col_ids, row_ptr;

  /// GPU
  {
    cpp_itt::quiet_region region;

    {
      auto gpu_time = gpu_csr_spmv (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
      cout << "GPU CSR: " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
    }

    {
      auto gpu_time = gpu_csr_vector_spmv (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
      cout << "GPU CSR (vector): " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
    }

    {
      auto gpu_time = gpu_ell_spmv (*ell_matrix, A, col_ids, x_gpu, y, x.get (), reference_answer.get ());
      cout << "GPU ELL: " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
    }

    {
      auto gpu_time = gpu_coo_spmv (*coo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
      cout << "GPU COO: " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
    }
    
    {
      scoo_matrix_class scoo_matrix (*coo_matrix);
    }

    resizable_gpu_memory<double> A_coo;
    resizable_gpu_memory<unsigned int> col_ids_coo;

    hybrid_matrix_class hybrid_matrix (*csr_matrix);

    for (double percent = 0.0; percent <= 1.0; percent += 0.35)
    {
      cout << "\n";
      hybrid_matrix.reallocate (*csr_matrix, percent);

      {
        auto gpu_time = gpu_hybrid_spmv (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
        cout << "GPU HYBRID (" << percent << "): " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
      }

      {
        auto gpu_time = gpu_hybrid_atomic_spmv (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
        cout << "GPU HYBRID (atomic, percent " << percent << "): " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
      }

      {
        resizable_gpu_memory<double> tmp;
        auto gpu_time = gpu_hybrid_cpu_coo_spmv (hybrid_matrix, A, col_ids, x_gpu, y, tmp, cpu_y.get (), x.get (), reference_answer.get ());
        cout << "GPU HYBRID (CPU COO, percent " << percent << "): " << gpu_time << " (SSCPU = " << cpu_naive_time / gpu_time << "; SMPCU = " << cpu_parallel_naive_time / gpu_time << ")" << endl;
      }
    }
  }

  return 0;
}