//
// Created by egi on 11/6/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_SCOO_SPMV_H
#define MATRIX_FORMAT_PERFORMANCE_SCOO_SPMV_H

#include "measurement_class.h"
#include "resizable_gpu_memory.h"
#include "scoo_matrix_class.h"

template <typename data_type>
measurement_class gpu_scoo_spmv (
    bool print_diff,
    const scoo_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &r_index,
    resizable_gpu_memory<unsigned int> &c_index,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

#endif // MATRIX_FORMAT_PERFORMANCE_SCOO_SPMV_H
