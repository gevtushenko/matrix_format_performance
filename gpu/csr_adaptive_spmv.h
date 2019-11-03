//
// Created by egi on 11/3/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_CSR_ADAPTIVE_SPMV_H
#define MATRIX_FORMAT_PERFORMANCE_CSR_ADAPTIVE_SPMV_H

#include "matrix_market_reader.h"
#include "resizable_gpu_memory.h"
#include "measurement_class.h"
#include "matrix_converter.h"

template <typename data_type>
measurement_class gpu_csr_adaptive_spmv (
    const csr_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);


#endif // MATRIX_FORMAT_PERFORMANCE_CSR_ADAPTIVE_SPMV_H
