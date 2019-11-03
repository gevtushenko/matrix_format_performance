//
// Created by egi on 10/31/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H
#define MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H

#include <string>
#include <iostream>

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

template <typename data_type>
void compare_results (unsigned int y_size, const data_type *a, const data_type *b)
{
  data_type numerator = 0.0;
  data_type denumerator = 0.0;

  for (unsigned int i = 0; i < y_size; i++)
  {
    numerator += (a[i] - b[i]) * (a[i] - b[i]);
    denumerator += b[i] * b[i];
  }

  const data_type error = numerator / denumerator;

  if (error > 1e-9)
  {
    std::cerr << "ERROR: " << error << std::endl;

    for (unsigned int i = 0; i < y_size; i++)
    {
      if (std::abs (a[i] - b[i]) > 1e-8)
      {
        std::cerr << "a[" << i << "] = " << a[i] << "; b[" << i << "] = " << b[i] << std::endl;
        break;
      }
    }
  }

  std::cerr.flush ();
}


#endif // MATRIX_FORMAT_PERFORMANCE_MEASUREMENT_CLASS_H
