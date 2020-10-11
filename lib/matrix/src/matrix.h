/*
  MATRIX CLASS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 21.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _MATRIX_H_
#define _MATRIX_H_

// standard c++ headers
#include <iostream>
#include <memory>

// own headers
#include "common.h"

// pybind11 headers + namespace
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

//_________________________________________________________________________________________________
// helper functions
class matrix;
void output_device_matrix(matrix& mat);
void output_host_matrix  (matrix& mat);
void output_matrix       (matrix& mat, bool flag_host);

//_________________________________________________________________________________________________
// matrix headers
class matrix {
private :
  // data members
  size_t _rows;
  size_t _cols;
  size_t _size;

  // flag if something is allocated on host or device
  bool flag_device_alloc;
  bool flag_host_alloc;

  // fucntions to allocate memory
  void alloc_host  (void);
  void alloc_device(void);

public :

  // constructors
  matrix(void);
  matrix(size_t rows, size_t cols);
  matrix(py::array_t<double, py::array::c_style | py::array::forcecast> array);

  // copy constructor
  matrix(const matrix& m);

  // destructor
  ~matrix() {};

  // getters
  size_t rows(void) const { return _rows; };
  size_t cols(void) const { return _cols; };
  size_t size(void) const { return _size; };
  size_t get_flag_device_alloc(void) const { return flag_device_alloc; };
  size_t get_flag_host_alloc(void) const { return flag_host_alloc; };


  // pointers to data
  std::shared_ptr<double> data_device;
  std::shared_ptr<double> data_host;

  // public allocator functions
  void alloc(void);
  void alloc_if_not_allocated(size_t rows, size_t cols);

  // copy functions for matrix
  void copy_device_to_host(void);
  void copy_host_to_device(void);

  // operator overloading
  friend std::ostream& operator <<(std::ostream& out, const matrix& mat);
  double& operator [](const int idx);
  const double& operator [](const int idx) const;
};
#endif // _MATRIX_H_
