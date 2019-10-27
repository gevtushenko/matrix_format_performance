//
// Created by egi on 9/15/19.
//

#ifndef MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H
#define MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H

#include "matrix_market_reader.h"

#include <memory>

template <typename data_type>
class csr_matrix_class
{
public:
  csr_matrix_class () = delete;
  explicit csr_matrix_class (const matrix_market::matrix_class &matrix);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;
  std::unique_ptr<unsigned int[]> row_ptr;
};

template <typename data_type>
class ell_matrix_class
{
public:
  ell_matrix_class () = delete;
  explicit ell_matrix_class (csr_matrix_class<data_type> &matrix);
  ell_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int elements_in_row_arg);

  static size_t estimate_size (csr_matrix_class<data_type> &matrix);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;

  unsigned int elements_in_rows = 0;
};

template <typename data_type>
class coo_matrix_class
{
public:
  coo_matrix_class () = delete;
  explicit coo_matrix_class (csr_matrix_class<data_type> &matrix);
  coo_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int element_start);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> cols;
  std::unique_ptr<unsigned int[]> rows;

private:
  size_t elements_count {};
};

/// Sliced COO Format
class scoo_matrix_class
{
public:
  scoo_matrix_class () = delete;
  explicit scoo_matrix_class (coo_matrix_class<double> &matrix);

  const matrix_market::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<double[]> values;
  std::unique_ptr<unsigned int[]> c_index;
  std::unique_ptr<unsigned int[]> r_index;
  std::unique_ptr<unsigned int[]> index;

private:
  size_t elements_count {};
};

template <typename data_type>
class hybrid_matrix_class
{
public:
  hybrid_matrix_class () = delete;
  explicit hybrid_matrix_class (csr_matrix_class<data_type> &matrix);

  void allocate (csr_matrix_class<data_type> &matrix, double percent);

  const matrix_market::matrix_class::matrix_meta meta;

  std::unique_ptr<ell_matrix_class<data_type>> ell_matrix;
  std::unique_ptr<coo_matrix_class<data_type>> coo_matrix;
};

#endif // MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTER_H
