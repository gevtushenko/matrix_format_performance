//
// Created by egi on 9/21/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H

template <typename data_type>
class csr_matrix_class;

template <typename data_type>
class ell_matrix_class;

/// Perform y = Ax
template<typename data_type>
double cpu_csr_spmv_single_thread_naive (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

template<typename data_type>
double cpu_csr_spmv_single_thread_naive_with_reduce_order (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

template<typename data_type>
double cpu_csr_spmv_multi_thread_naive (
    const csr_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

template<typename data_type>
double cpu_ell_spmv_multi_thread_naive (
    const ell_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y);

template<typename data_type>
double cpu_ell_spmv_multi_thread_avx2 (
    const ell_matrix_class<data_type> &matrix,
    data_type *x,
    data_type *y,
    const data_type *reference_y);

#endif // MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
