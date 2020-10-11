/*
  TEST MATRICES CPP
  
  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 
  TO-DO   : 
  CAUTION : 
*/

#include "common.h"
#include "test_matrices_cpp.h"
#include "test_matrix_operator.h"

//_______________________________________________________________________________________________
// comapare two matrices error
std::tuple <bool, double, double, double> compare_matrices_cpp(matrix& lhs,
							       matrix& rhs,
							       bool    flag_host,
							       double  eps)
{
  // calculate the min, max, and mean relative error
  double min = 0, max = 0, rel = 0;
  
  // check the matrix dimensions
  if (lhs.rows() != rhs.rows() or
      lhs.cols() != rhs.cols() or
      lhs.size() != rhs.size())
    return std::make_tuple(false, min, max, rel);
  
  // move the matrices around, as the [] operator only acts on host elements
  if (!flag_host) {
    lhs.copy_device_to_host();
    rhs.copy_device_to_host();
  }
  
  // compute the min, max and mean relative error
  for (int idx = 0; idx < lhs.size(); idx++) {
    if (std::abs(lhs[idx]) != 0.) {
      rel += std::abs((lhs[idx] - rhs[idx]) / lhs[idx]);
    } else {
      rel += std::abs(lhs[idx] - rhs[idx]);
    }
    // setting min and max
    if (rel < min)
      min = rel;
    if (rel > max)
      max = rel;
  }
  
  // compute the relative error
  if (rel > eps)
    return std::make_tuple(false, min, max, rel);
  else
    return std::make_tuple(true, min, max, rel);
}

//_______________________________________________________________________________________________
// create a identity matrix
matrix create_i_matrix(int rows)
{
  matrix i(rows, rows);
  i.alloc();
  for (int row = 0; row < i.rows(); row++)
    for (int col = 0; col < i.cols(); col++)
      if (row == col)
	i[row*i.cols()+col] = 1.;
      else
	i[row*i.cols()+col] = 0.;
  i.copy_host_to_device();
  return i;
}

//_______________________________________________________________________________________________
// create ONE matrix
matrix create_one_matrix(int    rows,
			 int    cols,
			 double value)
{
  matrix one(rows, cols);
  one.alloc();
  for (int idx = 0; idx < one.size(); idx++)
    one[idx] = value;
  one.copy_host_to_device();
  return one;
}

//_______________________________________________________________________________________________
// create a random matrix within min/max
matrix create_random_matrix(int    rows,
			    int    cols,
			    double min,
			    double max)
{
  matrix ran(rows, cols);
  ran.alloc();
  for (int idx = 0; idx < ran.size(); idx++)
    ran[idx] = random_double(min, max);
  ran.copy_host_to_device();
  return ran;
}
