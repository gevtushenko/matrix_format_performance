//
// Created by egi on 9/15/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H

#include "matrix_converter.h"

template <typename T>
class resizable_gpu_memory;

/// Perform y = Ax
template <typename data_type>
double gpu_csr_spmv (
    const csr_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

template <typename data_type>
double gpu_csr_vector_spmv (
    const csr_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ptr,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

template <typename data_type>
double gpu_ell_spmv (
    const ell_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

template <typename data_type>
double gpu_coo_spmv (
    const coo_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

template <typename data_type>
double gpu_hybrid_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A_ell,
    resizable_gpu_memory<data_type> &A_coo,
    resizable_gpu_memory<unsigned int> &ell_col_ids,
    resizable_gpu_memory<unsigned int> &coo_col_ids,
    resizable_gpu_memory<unsigned int> &coo_row_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

template <typename data_type>
double gpu_hybrid_atomic_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A_ell,
    resizable_gpu_memory<data_type> &A_coo,
    resizable_gpu_memory<unsigned int> &ell_col_ids,
    resizable_gpu_memory<unsigned int> &coo_col_ids,
    resizable_gpu_memory<unsigned int> &coo_row_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    data_type*reusable_vector,
    const data_type*reference_y);

template <typename data_type>
double gpu_hybrid_cpu_coo_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    resizable_gpu_memory<data_type> &A_ell,
    resizable_gpu_memory<unsigned int> &ell_col_ids,
    resizable_gpu_memory<data_type> &x,
    resizable_gpu_memory<data_type> &y,

    resizable_gpu_memory<data_type> &tmp,

    data_type*cpu_y,
    data_type*reusable_vector,
    const data_type*reference_y);

#endif // MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
