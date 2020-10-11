/*
  TEST MATRIX HEADER
  
  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 
  TO-DO   : 
  CAUTION : 
*/

#ifndef _TEST_MATRIX_CPP_H_
#define _TEST_MATRIX_CPP_H_

#include <tuple>
#include <algorithm>
#include "matrix.h"

// funtions to compare 2 matrices
std::tuple <bool, double, double, double> compare_matrices_cpp(matrix& lhs, matrix& rhs, bool flag_host, double eps);

// create simple matrices
matrix create_one_matrix   (int rows, int cols, double value);
matrix create_random_matrix(int rows, int cols, double min, double max);
matrix create_i_matrix     (int rows);

#endif // _TEST_MATRIX_CPP_H_
