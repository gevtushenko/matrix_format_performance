#include <cuda_runtime.h>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <utility>

#include "json.hpp"

#include "measurement_class.h"
#include "scoo_matrix_class.h"
#include "matrix_market_reader.h"
#include "resizable_gpu_memory.h"
#include "matrix_converter.h"

#include "csr_adaptive_spmv.h"
#include "gpu_matrix_multiplier.h"
#include "cpu_matrix_multiplier.h"
#include "scoo_spmv.h"

#include "cpp_itt.h"

#include "fmt/format.h"
#include "fmt/color.h"
#include "fmt/core.h"

#define CHECK_CUSP 0
#define CHECK_CPU 0

using namespace nlohmann;
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

  void print_time (const measurement_class &measurement) const
  {
    const double time = measurement.get_elapsed ();
    fmt::print (fmt::fg (fmt::color::yellow), "{0:<25}", measurement.get_format ());
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

template <typename data_type>
vector<measurement_class> perform_measurement (
    const string &mtx,
    const matrix_market::reader &reader,
    size_t free_memory,
    int sm_count,
    size_t shared_mem_size)
{
  vector<measurement_class> measurements;

  unique_ptr<csr_matrix_class<data_type>> csr_matrix;
  unique_ptr<ell_matrix_class<data_type>> ell_matrix;
  unique_ptr<coo_matrix_class<data_type>> coo_matrix;

  {
    cpp_itt::quiet_region region;

    csr_matrix = make_unique<csr_matrix_class<data_type>> (reader.matrix ());
    cout << "Complete converting to CSR" << endl;

    if (!CHECK_CPU)
    {
      const size_t csr_matrix_size = csr_matrix->get_matrix_size ();
      const size_t ell_matrix_size = ell_matrix_class<data_type>::estimate_size (*csr_matrix);

      const size_t vec_size = std::max (reader.matrix ().meta.rows_count, reader.matrix ().meta.cols_count) * sizeof (data_type);
      const size_t matrix_size = std::max (csr_matrix_size, ell_matrix_size) * sizeof (data_type);
      const size_t estimated_size = matrix_size + 5 * vec_size;

      if (estimated_size * sizeof (data_type) > free_memory * 0.9)
        return {};

      ell_matrix = make_unique<ell_matrix_class<data_type>> (*csr_matrix);
      cout << "Complete converting to ELL" << endl;

      coo_matrix = make_unique<coo_matrix_class<data_type>> (*csr_matrix);
      cout << "Complete converting to COO" << endl;
    }
  }

  // CPU
  std::unique_ptr<data_type[]> reference_answer (new data_type[csr_matrix->meta.rows_count]);
  std::unique_ptr<data_type[]> reference_answer_for_reduce_order (new data_type[csr_matrix->meta.rows_count]);
  std::unique_ptr<data_type[]> cpu_y (new data_type[csr_matrix->meta.rows_count]);
  std::unique_ptr<data_type[]> x (new data_type[std::max (csr_matrix->meta.rows_count, csr_matrix->meta.cols_count)]);

  double cpu_naive_time {};

  {
    auto duration = cpp_itt::create_event_duration ("cpu_csr_spmv_single_thread_naive");
    auto cpu_naive = cpu_csr_spmv_single_thread_naive (*csr_matrix, x.get (), reference_answer.get ());
    measurements.push_back (cpu_naive);
    cpu_naive_time = cpu_naive.get_elapsed ();
  }
  time_printer single_core_timer (cpu_naive_time);
  single_core_timer.print_time (measurements.back ());

  auto measure_multiple_times = [&] (const std::function<measurement_class(bool)> &action)
  {
    measurement_class result;
    const unsigned int measurements_count = 20;
    for (unsigned int measurement_id = 0; measurement_id < measurements_count; measurement_id++)
      result += action (measurement_id == 0);
    result.finalize ();

    measurements.push_back (result);
    return result;
  };

  double cpu_parallel_naive_time {};

  {
    auto duration = cpp_itt::create_event_duration ("cpu_csr_spmv_multi_thread_naive");
    auto cpu_parallel_naive = measure_multiple_times ([&](bool) { return cpu_csr_spmv_multi_thread_naive (*csr_matrix, x.get (), cpu_y.get ()); });
    cpu_parallel_naive_time = cpu_parallel_naive.get_elapsed ();
    single_core_timer.print_time (cpu_parallel_naive);
  }

  {
    auto duration = cpp_itt::create_event_duration ("cpu_csr_spmv_mkl");
    auto cpu_time = measure_multiple_times ([&](bool) { return cpu_csr_spmv_mkl (*csr_matrix, x.get (), cpu_y.get (), reference_answer.get ()); });
    single_core_timer.print_time (cpu_time);
  }

  if (CHECK_CPU)
    return measurements;

  time_printer multi_core_timer (cpu_naive_time, cpu_parallel_naive_time);

  /// GPU Reusable memory
  resizable_gpu_memory<data_type> A, x_gpu, y;
  resizable_gpu_memory<unsigned int> col_ids, row_ptr;

  /// GPU
  {
    cpp_itt::quiet_region region;

    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_csr_spmv<data_type> (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    {
      cpu_csr_spmv_single_thread_naive_with_reduce_order (*csr_matrix, x.get (), reference_answer_for_reduce_order.get ());
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_csr_vector_spmv<data_type> (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer_for_reduce_order.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_csr_adaptive_spmv<data_type> (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_csr_cusparse_spmv<data_type> (*csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    if (CHECK_CUSP)
    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_csr_cusp_spmv<data_type> (mtx, *csr_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_ell_spmv<data_type> (*ell_matrix, A, col_ids, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    if (CHECK_CUSP)
    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_ell_cusp_spmv<data_type> (mtx, *ell_matrix, A, col_ids, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_coo_spmv<data_type> (*coo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    {
      scoo_matrix_class scoo_matrix (sm_count, shared_mem_size, *csr_matrix);
      fmt::print ("SCOO: slice_size={}; lane_size={}; slices_count={}\n", scoo_matrix.slice_size, scoo_matrix.lane_size, scoo_matrix.slices_count);
      auto gpu_time = measure_multiple_times ([&](bool print_diff) { return gpu_scoo_spmv<data_type> (print_diff, scoo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    if (CHECK_CUSP)
    {
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_coo_cusp_spmv<data_type> (mtx, *coo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
    }

    resizable_gpu_memory<data_type> A_coo;
    resizable_gpu_memory<unsigned int> col_ids_coo;

    {
      hybrid_matrix_class<data_type> hybrid_matrix (*csr_matrix);
      hybrid_matrix.allocate(*csr_matrix, 0.2);
      auto gpu_time = measure_multiple_times ([&](bool) { return gpu_hybrid_spmv<data_type> (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      auto gpu_time_atomic = measure_multiple_times ([&](bool) { return gpu_hybrid_atomic_spmv<data_type> (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ()); });
      multi_core_timer.print_time (gpu_time);
      multi_core_timer.print_time (gpu_time_atomic);
    }

    if (0)
    {
      hybrid_matrix_class<data_type> hybrid_matrix (*csr_matrix);

      for (double percent = 0.0; percent <= 1.0; percent += 0.35)
      {
        hybrid_matrix.allocate(*csr_matrix, percent);

        std::string percent_str = std::to_string ((int)(percent * 100));

        {
          // const string label = "GPU HYBRID " + percent_str;
          auto gpu_time = gpu_hybrid_spmv<data_type> (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
          multi_core_timer.print_time (gpu_time);
          measurements.push_back (gpu_time);
        }

        {
          // const string label = "GPU HYBRID ATOMIC " + percent_str;
          auto gpu_time = gpu_hybrid_atomic_spmv<data_type> (hybrid_matrix, A, A_coo, col_ids, col_ids_coo, row_ptr, x_gpu, y, x.get (), reference_answer.get ());
          multi_core_timer.print_time (gpu_time);
          measurements.push_back (gpu_time);
        }

        {
          // const string label = "GPU HYBRID CPU COO " + percent_str;
          resizable_gpu_memory<data_type> tmp;
          auto gpu_time = gpu_hybrid_cpu_coo_spmv<data_type> (hybrid_matrix, A, col_ids, x_gpu, y, tmp, cpu_y.get (), x.get (), reference_answer.get ());
          multi_core_timer.print_time (gpu_time);
          measurements.push_back (gpu_time);
        }
      }
    }
  }

  return measurements;
}

string get_filename (const string &path)
{
  size_t i = path.rfind ('/', path.length ());
  if (i != string::npos)
    return (path.substr (i + 1, path.length () - i));
  return path;
}

#include "nvml.h"

class gpu_frequency_controller
{
  nvmlDevice_t nvmlDeviceId;

public:
  gpu_frequency_controller ()
  {
    nvmlReturn_t nvml_error = nvmlInit ();
  }

  void fix_frequency ()
  {
    int active_cuda_device = 0;
    cudaGetDevice (&active_cuda_device);

    //1. Get device properties of active CUDA device
    cudaDeviceProp activeCUDAdeviceProp;
    cudaGetDeviceProperties (&activeCUDAdeviceProp, active_cuda_device);

    //2. Get number of NVML devices
    unsigned int nvmlDeviceCount = 0;
    nvmlDeviceGetCount ( &nvmlDeviceCount );

    //3. Loop over all NVML devices
    for ( unsigned int nvmlDeviceIdx = 0;
          nvmlDeviceIdx < nvmlDeviceCount;
          ++nvmlDeviceIdx )
    {
      //4. Obtain NVML device Id
      nvmlDeviceGetHandleByIndex ( nvmlDeviceIdx, &nvmlDeviceId );

      //5. Query PCIe Info of the NVML device
      nvmlPciInfo_t nvmPCIInfo;
      nvmlDeviceGetPciInfo ( nvmlDeviceId, &nvmPCIInfo );

      //6. Compare NVML device PCI-E info with CUDA device properties
      if ( static_cast<unsigned int>(activeCUDAdeviceProp.pciBusID)
           == nvmPCIInfo.bus &&
           static_cast<unsigned int>(activeCUDAdeviceProp.pciDeviceID)
           == nvmPCIInfo.device &&
           static_cast<unsigned int>(activeCUDAdeviceProp.pciDomainID)
           == nvmPCIInfo.domain )
        break;
    }

    //Query current application clock setting
    unsigned int appSMclock = 0;
    unsigned int appMemclock = 0;
    nvmlDeviceGetApplicationsClock ( nvmlDeviceId,
                                     NVML_CLOCK_SM,
                                     &appSMclock );
    nvmlDeviceGetApplicationsClock ( nvmlDeviceId,
                                     NVML_CLOCK_MEM,
                                     &appMemclock );

//Query maximum application clock setting
    unsigned int maxSMclock = 0;
    unsigned int maxMemclock = 0;
    nvmlDeviceGetMaxClockInfo ( nvmlDeviceId,
                                NVML_CLOCK_SM,
                                &maxSMclock );
    nvmlDeviceGetMaxClockInfo ( nvmlDeviceId,
                                NVML_CLOCK_MEM,
                                &maxMemclock );

    fmt::print ("Max SM Clock: {}", maxSMclock);

    nvmlEnableState_t isRestricted;
    nvmlDeviceGetAPIRestriction ( nvmlDeviceId,
                                  NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
                                  &isRestricted);

    if ( NVML_FEATURE_DISABLED == isRestricted )
    {
      // auto return_code = nvmlDeviceSetApplicationsClocks (nvmlDeviceId, maxMemclock, maxSMclock); This function is not supported on RTX...
      auto return_code = nvmlDeviceSetGpuLockedClocks (nvmlDeviceId, maxSMclock, maxSMclock);
      if (return_code != NVML_SUCCESS)
        fmt::print ("Can't fix gpu frequencies\n");
    }
  }

  ~gpu_frequency_controller()
  {
    nvmlDeviceResetGpuLockedClocks (nvmlDeviceId);
    // nvmlDeviceResetApplicationsClocks (nvmlDeviceId);
    nvmlShutdown ();
  }
};

/**
 * To run this tests you should:
 *
 * 1) Set cpu power mode:      sudo cpupower frequency-set --governor performance
 * 2) Fix cpu frequencies:     sudo cpupower frequency-set --min 4500000
 * 3) Check gpu frequencies:   sudo nvidia-smi -ac
 * 4) Enable persistence mode: sudo nvidia-smi -pm ENABLED -i 0
 * 5) Fix mem,gpu frequencies: sudo nvidia-smi -ac 7000,2100
 *
 * Optionally restore defaults after benchmarks: sudo nvidia-smi -ac 6800,1515
 */
int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " /path/to/mtx_list" << endl;
    return 1;
  }

  size_t free_gpu_mem, total_gpu_mem;
  int sm_count = 0;
  size_t shared_mem_size = 0;

  int gpu_id = 0;

  gpu_frequency_controller gpu_frequency;

  if (!CHECK_CPU)
  {
    cudaSetDevice (gpu_id);
    gpu_frequency.fix_frequency ();

    cudaDeviceProp prop {};
    cudaGetDeviceProperties (&prop, gpu_id);
    sm_count = prop.multiProcessorCount;
    shared_mem_size = prop.sharedMemPerBlock;

    auto status = cudaMemGetInfo (&free_gpu_mem, &total_gpu_mem);
    if (status != cudaSuccess)
    {
      cerr << "CUDA Can't get free memory!\n";
      return 1;
    }
  }

  ifstream list (argv[1]);

  string mtx;
  json measurements;
  json effective_bandwidth;
  json computational_throughput;

  json matrices_info;

  while (getline (list, mtx))
  {
    fmt::print ("Start loading matrix {}\n", mtx);

    ifstream is (mtx);
    matrix_market::reader reader (is);
    auto &meta = reader.matrix ().meta;
    fmt::print ("Complete loading (rows: {}; cols: {}; nnz: {}; nnzpr: {})\n", meta.rows_count, meta.cols_count, meta.non_zero_count, meta.non_zero_count / meta.rows_count);

    unordered_map<string, vector<measurement_class>> results;
    results["float"] = perform_measurement<float> (mtx, reader, free_gpu_mem, sm_count, shared_mem_size);
    results["double"] = perform_measurement<double> (mtx, reader, free_gpu_mem, sm_count, shared_mem_size);

    mtx = get_filename (mtx);

    if (results["float"].empty () || results["double"].empty ())
      continue; // Don't store result for matrices that couldn't be computed on GPU

    for (auto &[type, result]: results)
    {
      for (auto &measurement: result)
      {
        measurements[type][mtx][measurement.get_format ()] = measurement.get_elapsed ();
        effective_bandwidth[type][mtx][measurement.get_format ()] = measurement.get_effective_bandwidth ();
        computational_throughput[type][mtx][measurement.get_format ()] = measurement.get_computational_throughput ();
      }
    }

    matrices_info[mtx]["nnz"] = reader.matrix ().meta.non_zero_count;
    matrices_info[mtx]["rows"] = reader.matrix ().meta.rows_count;
    matrices_info[mtx]["cols"] = reader.matrix ().meta.cols_count;
  }

  ofstream mi_os ("matrices_info.json");
  mi_os << matrices_info.dump (2);

  for (auto &precision: { "float", "double" })
  {
    {
      ofstream os (std::string (precision) + ".json");
      os << measurements[precision].dump (2);
    }

    {
      ofstream os (std::string (precision) + "_effective_bandwidth.json");
      os << effective_bandwidth[precision].dump (2);
    }

    {
      ofstream os (std::string (precision) + "_computational_throughput.json");
      os << computational_throughput[precision].dump (2);
    }
  }

  // inf -> /home/egi/Documents/data/matrices/matrix_market/unco/raw/lp_scsd8/lp_scsd8.mtx
  return 0;
}