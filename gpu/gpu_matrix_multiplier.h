//
// Created by egi on 9/15/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H

#include "matrix_converter.h"

template <typename T>
class resizable_gpu_memory;

/// Perform y = Ax
double gpu_csr_spmv (
    const csr_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y);

double gpu_csr_vector_spmv (
    const csr_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y);

double gpu_ell_spmv (
    const ell_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y);

double gpu_coo_spmv (
    const coo_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y);

double gpu_coo_privatization_spmv (
    const coo_matrix_class &matrix,
    resizable_gpu_memory<double> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<double> &x,
    resizable_gpu_memory<double> &y,

    double *reusable_vector,
    const double *reference_y);

#endif // MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
