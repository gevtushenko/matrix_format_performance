//
// Created by egi on 9/21/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H

class csr_matrix_class;
class ell_matrix_class;

/// Perform y = Ax
double cpu_csr_spmv_single_thread_naive (
    const csr_matrix_class &matrix,
    double *x,
    double *y);

double cpu_csr_spmv_multi_thread_naive (
    const csr_matrix_class &matrix,
    double *x,
    double *y);

double cpu_ell_spmv_multi_thread_naive (
    const ell_matrix_class &matrix,
    double *x,
    double *y);

double cpu_ell_spmv_multi_thread_avx2 (
    const ell_matrix_class &matrix,
    double *x,
    double *y,
    const double *reference_y);

#endif // MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
