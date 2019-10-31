//
// Created by egi on 10/31/19.
//

#include "measurement_class.h"

measurement_class::measurement_class (
    std::string format,
    double elapsed_arg,
    double load_store_bytes,
    double operations_count)
  : elapsed (elapsed_arg)
  , effective_bandwidth (load_store_bytes / (elapsed * giga))
  , computational_throughput (operations_count / (elapsed * giga))
  , matrix_format (move (format))
{ }
