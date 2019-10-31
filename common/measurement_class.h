//
// Created by egi on 10/31/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H
#define MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H

#include <string>

class measurement_class
{
  const size_t giga = 1E+9;

public:
  measurement_class (
      std::string format,
      double elapsed,
      double load_store_bytes,
      double operations_count);

  double get_elapsed () const { return elapsed; }
  double get_effective_bandwidth () const { return effective_bandwidth; }
  double get_computational_throughput () const { return computational_throughput; }

  const std::string &get_format () const { return matrix_format; }

private:
  double elapsed {};
  double effective_bandwidth {};
  double computational_throughput {};
  std::string matrix_format;
};

#endif // MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H
