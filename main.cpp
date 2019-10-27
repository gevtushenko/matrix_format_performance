#include <cuda_runtime.h>
#include <optional>
#include <iostream>
#include <fstream>

#include "matrix_market_reader.h"
#include "resizable_gpu_memory.h"
#include "matrix_converter.h"

#include "gpu_matrix_multiplier.h"
#include "cpu_matrix_multiplier.h"

#include "cpp_itt.h"

#include "fmt/format.h"
#include "fmt/color.h"
#include "fmt/core.h"

using namespace std;

class time_printer
{
  double reference {};
  optional<double> parallel_reference;

  /// Settings
  const unsigned int time_width = 20;
  const unsigned int time_precision = 6;

public:
  explicit time_printer (
      double reference_time,
      std::optional<double> parallel_ref = nullopt)
   : reference (reference_time)
   , parallel_reference (move (parallel_ref))
  {
  }

  void add_time (double time, fmt::color color) const
  {
    fmt::print (fmt::fg (color), "{2:<{0}.{1}g}   ", time_width, time_precision, time);
  }

  void print_time (const std::string &label, double time) const
  {
    fmt::print (fmt::fg (fmt::color::yellow), "{0:<25}", label);
    fmt::print (":  ");
    add_time (time, fmt::color::white);
    add_time (speedup (time), fmt::color::green);
    if (parallel_reference)
      add_time (parallel_speedup (time), fmt::color::green_yellow);
    fmt::print ("\n");
  }

  double speedup (double time) const
  {
    return reference / time;
  }

  double parallel_speedup (double time) const
  {
    return *parallel_reference / time;
  }
};

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " /path/to/mtx" << endl;
    return 1;
  }

  cudaSetDevice (0);

  fmt::print ("Start loading matrix\n");

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
  }

  time_printer single_core_timer (cpu_naive_time);
  single_core_timer.print_time ("CPU CSR", cpu_naive_time);

  double cpu_parallel_naive_time {};

  {
    auto duration = cpp_itt::create_event_duration ("cpu_csr_spmv_multi_thread_naive");
    cpu_parallel_naive_time = cpu_csr_spmv_multi_thread_naive (*csr_matrix, x.get (), cpu_y.get ());
    single_core_timer.print_time ("CPU CSR Parallel", cpu_parallel_naive_time);
  }

  time_printer multi_core_timer (cpu_naive_time, cpu_parallel_naive_time);

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
      multi_core_timer.print_time ("GPU CSR", gpu_time);
    }

    {
      auto gpu_time = gpu_csr_vector_spmv (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
      multi_core_timer.print_time ("GPU CSR (vector)", gpu_time);
    }

    {
      auto gpu_time = gpu_ell_spmv (*ell_matrix, A, col_ids, x_gpu, y, x.get (), reference_answer.get ());
      multi_core_timer.print_time ("GPU ELL", gpu_time);
    }

    {
      auto gpu_time = gpu_coo_spmv (*coo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
      multi_core_timer.print_time ("GPU COO", gpu_time);
    }

    if (0)
    {
      scoo_matrix_class scoo_matrix (*coo_matrix);
    }

    resizable_gpu_memory<double> A_coo;
    resizable_gpu_memory<unsigned int> col_ids_coo;

    hybrid_matrix_class hybrid_matrix (*csr_matrix);

    for (double percent = 0.0; percent <= 1.0; percent += 0.35)
    {
      hybrid_matrix.allocate(*csr_matrix, percent);

      std::string percent_str = std::to_string ((int)(percent * 100));

      {
        auto gpu_time = gpu_hybrid_spmv (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
        multi_core_timer.print_time ("GPU HYBRID " + percent_str, gpu_time);
      }

      {
        auto gpu_time = gpu_hybrid_atomic_spmv (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
        multi_core_timer.print_time ("GPU HYBRID ATOMIC " + percent_str, gpu_time);
      }

      {
        resizable_gpu_memory<double> tmp;
        auto gpu_time = gpu_hybrid_cpu_coo_spmv (hybrid_matrix, A, col_ids, x_gpu, y, tmp, cpu_y.get (), x.get (), reference_answer.get ());
        multi_core_timer.print_time ("GPU HYBRID CPU COO " + percent_str, gpu_time);
      }
    }
  }

  return 0;
}