//
// Created by egi on 9/21/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H

#include "measurement_class.h"

template <typename data_type>
class csr_matrix_class;

template <typename data_type>
class ell_matrix_class;

/// Perform y = Ax
template<typename data_type>
measurement_class cpu_csr_spmv_single_thread_naive (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

measurement_class cpu_csr_spmv_mkl (
    const csr_matrix_class<float> &matrix,
    float *x,
    float *y,
    const float *reference_y);

measurement_class cpu_csr_spmv_mkl (
    const csr_matrix_class<double> &matrix,
    double *x,
    double *y,
    const double *reference_y);

template<typename data_type>
measurement_class cpu_csr_spmv_single_thread_naive_with_reduce_order (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

template<typename data_type>
measurement_class cpu_csr_spmv_multi_thread_naive (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

template<typename data_type>
measurement_class cpu_ell_spmv_multi_thread_naive (
    const ell_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

#endif // MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
