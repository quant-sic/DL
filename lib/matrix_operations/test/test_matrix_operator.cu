/*
  RUN FILE TO TEST MATRIX OPERATIONS

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 11.08.2020
  TO-DO   :
  CAUTION : 1. remove unnecessary code aka this script
*/

// standard c headers
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

// own c headers
#include "matrix_operator.h"
#include "matrix_operator_gpu.h"
#include "test_matrix_operator.h"
#include "common.h"

//________________________________________________________________________________________________
// check if the operator of host and device produce the same result
// CAUTION :
void compare_host_device_operator(const char* func_name,
				  void (f_host)  (const double* lhs,
						  const double* rhs,
						  int     rows_lhs,
						  int     cols_rhs,
						  int     cols_lhs,
						  double* res,
						  int     threads_block),
				  void (f_device)(const double* lhs,
						  const double* rhs,
						  int     rows_lhs,
						  int     cols_rhs,
						  int     cols_lhs,
						  double* res,
						  int     threads_block),
				  int rows_lhs,
				  int cols_rhs,
				  int cols_lhs,
				  int threads_block)
{
  // parameters for random number generation
  double min = -100.;
  double max = +100.;

  // calculating sizes
  int size_lhs = rows_lhs * cols_lhs;
  int size_rhs = cols_rhs * cols_lhs;
  int size_res = rows_lhs * cols_rhs;

  // allocating arrays
  double* lhs   = (double*) malloc(size_lhs*sizeof(double));
  double* rhs   = (double*) malloc(size_rhs*sizeof(double));
  double* h_res = (double*) malloc(size_res*sizeof(double));
  double* d_res = (double*) malloc(size_res*sizeof(double));

  // fill matrices with data
  create_random_matrix(lhs, size_lhs, min, max);
  create_random_matrix(rhs, size_rhs, min, max);

  memset(h_res, 0, size_res*sizeof(h_res[0]));
  memset(d_res, 0, size_res*sizeof(d_res[0]));

  // calculate solutions and comapre them
  f_host(lhs, rhs, rows_lhs, cols_rhs, cols_lhs, h_res, threads_block);
  f_device(lhs, rhs, rows_lhs, cols_rhs, cols_lhs, d_res, threads_block);
  compare_matrices_error(func_name, h_res, d_res, rows_lhs, cols_rhs);
}

//________________________________________________________________________________________________
// time a matrix operation
// repeats the operation a number of rep times, for a matrix with the size rows, cols
void time_matrix_operator_host(const char* func_name,
			       void (f_matrix_operation) (const double* lhs,
							  const double* rhs,
							  int     rows_lhs,
							  int     cols_rhs,
							  int     cols_lhs,
							  double* res,
							  int     max_threads_block),
			       int         rows_lhs,
			       int         cols_rhs,
			       int         cols_lhs,
			       int         rep,
			       int         max_threads_block)
{
  // min and max for the random matrix creation
  double min = -1.;
  double max =  1.;

  // calculating sizes
  int size_lhs = rows_lhs * cols_lhs;
  int size_rhs = cols_rhs * cols_lhs;
  int size_res = rows_lhs * cols_rhs;

  // ALLOCATING THE 3 MATRICES
  double* lhs = (double*) malloc(size_lhs*sizeof(double));
  double* rhs = (double*) malloc(size_rhs*sizeof(double));
  double* res = (double*) malloc(size_res*sizeof(double));

  // repeat the matrix operation a number of times
  double sum_time = 0., time_start, time_min, time_max, time;
  for (int r = 0; r < rep; r++) {
    // create 2 random matrices for the matrix operation and reset resulting matrix
    create_random_matrix(lhs, size_lhs, min, max);
    create_random_matrix(rhs, size_rhs, min, max);
    memset(res, 0, size_res*sizeof(res[0]));

    // time the matrix operation
    time_start = seconds();
    f_matrix_operation(lhs, rhs, rows_lhs, cols_rhs, cols_lhs, res, max_threads_block);
    time = seconds() - time_start;

    // deciding for min/max time and adding sum for mean
    if (r == 0) {
      time_min = time;
      time_max = time;
    } else {
      if (time < time_min)
	time_min = time;
      if (time > time_max)
	time_max = time;
    }
    sum_time += time;
  }

  // freeing up memory from the 3 matrices
  free(lhs);
  free(rhs);
  free(res);

  // print the timing information
  printf("_______________________________________________________________________________________\n");
  printf("function     : %s\n"      , func_name);
  printf("repetitions  : %d\n"      , rep);
  printf("size matrix  : %d\n"      , size_res);
  printf("min/max time : %fs/ %fs\n", time_min, time_max);
  printf("total time   : %f\n"      , sum_time);
  printf("mean time    : %f\n"      , sum_time / (double) rep);
}

//________________________________________________________________________________________________
// function to time matrix operator on device
void time_matrix_operator_device(const char* func_name,
				 void (f_matrix_operation) (const double* lhs,
							    const double* rhs,
							    int     rows_lhs,
							    int     cols_rhs,
							    int     cols_lhs,
							    double* res,
							    int     threads_block),
				 int         rows_lhs,
				 int         cols_rhs,
				 int         cols_lhs,
				 int         rep,
				 int         max_threads_block)
{
  // min and max for the random matrix creation
  double min = -1.;
  double max =  1.;

  // calculating sizes
  int size_lhs = rows_lhs * cols_lhs;
  int size_rhs = cols_rhs * cols_lhs;
  int size_res = rows_lhs * cols_rhs;

  // ALLOCATING THE 3 MATRICES
  double* lhs = (double*) malloc(size_lhs*sizeof(double));
  double* rhs = (double*) malloc(size_rhs*sizeof(double));
  double* res = (double*) malloc(size_res*sizeof(double));

  // writing output start up
  printf("_______________________________________________________________________________________\n");
  printf("function          : %s\n", func_name);
  printf("repetitions       : %d\n", rep);
  printf("size matrix       : %d\n", size_res);
  printf("max_threads_block : %d\n", max_threads_block);
  printf("\n");
  printf("threads_per_block | mean time[s]\n");
  printf("--------------------------------\n");

  // repeat the matrix operation a number of times
  double sum_time = 0., time_start;
  for (int threads_block = 1; threads_block <= max_threads_block; threads_block *= 2) {

    // reset the total time amount
    sum_time = 0;

    for (int r = 0; r < rep; r++) {
      // create 2 random matrices for the matrix operation and reset resulting matrix
      create_random_matrix(lhs, size_lhs, min, max);
      create_random_matrix(rhs, size_rhs, min, max);
      memset(res, 0, size_res*sizeof(res[0]));

      // time the matrix operation
      time_start = seconds();
      f_matrix_operation(lhs, rhs, rows_lhs, cols_rhs, cols_lhs, res, threads_block);
      sum_time += seconds() - time_start;
    }
    printf("%17d |%11f \n", threads_block, sum_time / (double) rep);
  }

  // freeing up memory from the 3 matrices
  free(lhs);
  free(rhs);
  free(res);
}

//________________________________________________________________________________________________
// comapre the error of 2 matrices
// res has to be of size 4
// res structure : 1. min error
//                 2. max error
//                 3. mean error
//                 4. mean relative error
void compare_matrices_error_printless(double* res,
				      double* lhs,
				      double* rhs,
				      int     size)
{
  // calculating the important values
  double min = DBL_MAX, max = DBL_MIN;
  double sum_error = 0., sum_rel_error = 0., error, rel_error;
  for (int idx = 0; idx < size; idx++) {
    // error calculation
    error      = fabs(lhs[idx] - rhs[idx]);
    sum_error += error;

    // relative error
    rel_error      = error / fabs(lhs[idx]);
    sum_rel_error += rel_error;

    // min / max
    if (rel_error > max)
      max = rel_error;
    if (rel_error < min)
      min = rel_error;
  }

  // setting the result information
  res[0] = min;
  res[1] = max;
  res[2] = sum_error / size;
  res[3] = sum_rel_error / size;
}

//________________________________________________________________________________________________
// calculate the mean error between matrix elements
void compare_matrices_error(const char* comp_name,
			    double*     lhs,
			    double*     rhs,
			    int         rows,
			    int         cols)
{
  // calculation of the mean error per element
  int size = rows * cols;
  double sum_error = 0., sum_rel_error = 0., error;
  for (int idx = 0; idx < size; idx++) {
    error = fabs(lhs[idx] - rhs[idx]);
    sum_error     += error;
    sum_rel_error += error / fabs(lhs[idx]);
  }

  // printing the error information
  printf("________________________\n");
  printf("comparison name : %s\n", comp_name);
	printf("Dimensions      : %d, %d\n", rows,cols);
  printf("sum error       : %e\n", sum_error);
  printf("mean error      : %e\n", sum_error     / (double) size);
  printf("sum rel. error  : %e\n", sum_rel_error);
  printf("mean rel. error : %e\n", sum_rel_error / (double) size);
}

//________________________________________________________________________________________________
// create a random matrix with size as the number of elements, where all values are between
// min and max
// CAUTION : MATRIX MEMORY SHOULD BE ALLOCATED BEFOREHAND
void create_random_matrix(double* mat,
			  int     size,
			  double  min,
			  double  max)
{
  for (int idx = 0; idx < size; idx++)
    mat[idx] = random_double(min, max);
}

//________________________________________________________________________________________________
// Creates a const matrix where all entries are the same
// CAUTION : MATRIX MEMORY SHOULD BE ALLOCATED BEFOREHAND
void create_unit_matrix(double* mat,
			int     size,
			double  value)
{
  for (int idx = 0; idx < size; idx++)
    mat[idx] = value;
}

//________________________________________________________________________________________________
// Creates a scaled One matrix
// CAUTION : MATRIX MEMORY SHOULD BE ALLOCATED BEFOREHAND
void ONE_Matrix(double *mat,int dim,double value)
{
    memset(mat,0,dim*dim*sizeof(double));

    for(int j=0;j<dim;j++)
    {
      mat[j*dim+j]= value ;
    }
}

//________________________________________________________________________________________________
// returns a random double number between min and max
double random_double(double min,
		     double max)
{
  return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

//________________________________________________________________________________________________
// print a matrix to the console
// maybe do some padding or other stuff here
void print_matrix(double* mat,
		  int     rows,
		  int     cols)
{
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++)
      printf("%f ", mat[row * cols + col]);
    printf("\n");
  }
  printf("\n");
}

//___________________________________________________________________________________________________
// function wrappers for hadamard and add
void matrix_hadamard_gpu_wrapper(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block)
{
  matrix_hadamard_gpu(res, lhs, rhs, rows_lhs*cols_rhs, threads_block);
}

void matrix_add_gpu_wrapper(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block)
{
  matrix_add_gpu(res, lhs, rhs, rows_lhs*cols_rhs, threads_block);
}

void matrix_hadamard_cpu_wrapper(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block)
{
  matrix_hadamard_cpu(res, lhs, rhs, rows_lhs*cols_rhs);
}

void matrix_add_cpu_wrapper(const double* lhs, const double* rhs, int rows_lhs, int cols_rhs, int cols_lhs, double* res, int threads_block)
{
  matrix_add_cpu(res, lhs, rhs, rows_lhs*cols_rhs);
}
