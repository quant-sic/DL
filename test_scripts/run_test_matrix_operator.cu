/*
  RUN TEST FOR THE MATRIX OPERATORS HADAMARD / ADD / MULLADD_DIRECT / ADD_ALON_COL_DIRECT GPU VERSIONS

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 12.08.2020
  TO-DO   : 
  CAUTION : 
*/

#include <float.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include "matrix_operator.h"
#include "test_matrix_operator.h"

//_________________________________________________________________________________________________
// main function to run tests on host and device functions
int main(int agrc, char** argv) {
  // parameters for random number generation
  int max_rows = 100;
  int max_cols = 100;
  int dummy_threads_block = 64; // compatability, not used in dev functions
  double min = -1.;
  double max = +1.;
    
  // seeding number generator
  srand(time(NULL));

  // result pointer for matrix comparison
  double* result = (double*) malloc(4*sizeof(double));
  
  // setting up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Testing HADAMARD, ADD, MULADD, MULADD_DIRECT AND ADD_ALONG_COL_DIRECT for \n");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
  
  // HADAMARD
  int size = 0;
  for (int rows = 1; rows < max_rows; rows *= 2) {
    for (int cols = 1; cols < max_cols; cols *= 2) {
      // calculating size
      size = rows * cols;
      
      // allocating arrays host
      double* h_lhs = (double*) malloc(size*sizeof(double));
      double* h_rhs = (double*) malloc(size*sizeof(double));
      double* h_res = (double*) malloc(size*sizeof(double));
      
      // allocating arrays device
      double* d_lhs;
      double* d_rhs;
      double* d_res;
      CHECK(cudaMalloc((void**)&d_lhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_rhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_res, size*sizeof(double)));
      
      // allocate comparison array
      double* c_res = (double*) malloc(size*sizeof(double));
      
      // filling in lhs and rhs and result
      create_random_matrix(h_lhs, size, min, max);
      create_random_matrix(h_rhs, size, min, max);
      memset(h_res, 0, size*sizeof(h_res[0]));
      
      // copying arrays to device
      copy_host_to_device_double(h_lhs, d_lhs, size);
      copy_host_to_device_double(h_rhs, d_rhs, size);
      copy_host_to_device_double(h_res, d_res, size);
            
      // calculating host and device result
      matrix_hadamard_cpu  (h_res, h_lhs, h_rhs, size);
      matrix_hadamard_onDev(d_res, d_lhs, d_rhs, size, dummy_threads_block);
      
      // copying result back to host
      copy_device_to_host_double(d_res, c_res, size);
            
      // check for error
      if (!double_equal(h_res, c_res, size, sqrt(2)*DBL_EPSILON)) {
	printf("For rows : %d | cols : %d HADAMARD and 0 not compatible\n", rows, cols);
	return EXIT_FAILURE;
      }
            
      // freeing up the memory
      free(h_lhs);
      free(h_rhs);
      free(h_res);
      free_double_memory_device(d_lhs);
      free_double_memory_device(d_rhs);
      free_double_memory_device(d_res);
      free(c_res);
    }
  }
  printf("HADARMARD : SUCCESS\n");
  
  // ADD
  for (int rows = 1; rows < max_rows; rows *= 2) {
    for (int cols = 1; cols < max_cols; cols *= 2) {
      // calculating size
      size = rows * cols;
      
      // allocating arrays host
      double* h_lhs = (double*) malloc(size*sizeof(double));
      double* h_rhs = (double*) malloc(size*sizeof(double));
      double* h_res = (double*) malloc(size*sizeof(double));
      
      // allocating arrays device
      double* d_lhs;
      double* d_rhs;
      double* d_res;
      CHECK(cudaMalloc((void**)&d_lhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_rhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_res, size*sizeof(double)));
      
      
      // allocate comparison array
      double* c_res = (double*) malloc(size*sizeof(double));
      
      // filling in lhs and rhs and result
      create_random_matrix(h_lhs, size, min, max);
      create_random_matrix(h_rhs, size, min, max);
      memset(h_res, 0, size*sizeof(h_res[0]));
      
      // copying arrays to device
      copy_host_to_device_double(h_lhs, d_lhs, size);
      copy_host_to_device_double(h_rhs, d_rhs, size);
      copy_host_to_device_double(h_res, d_res, size);
            
      // calculating host and device result
      matrix_add_cpu  (h_res, h_lhs, h_rhs, size);
      matrix_add_onDev(d_res, d_lhs, d_rhs, size, dummy_threads_block);
      
      // copying result back to host
      copy_device_to_host_double(d_res, c_res, size);
            
      // check for error
      if (!double_equal(h_res, c_res, size, sqrt(2)*DBL_EPSILON)) {
	printf("For rows : %d | cols : %d ADD and 0 not compatible\n", rows, cols);
	return EXIT_FAILURE;
      }
            
      // freeing up the memory
      free(h_lhs);
      free(h_rhs);
      free(h_res);
      free_double_memory_device(d_lhs);
      free_double_memory_device(d_rhs);
      free_double_memory_device(d_res);
      free(c_res);
    }
  }
  printf("ADD       : SUCCESS\n");

  // MULADD
  for (int rows = 1; rows < max_rows; rows *= 2) {
    for (int cols = 1; cols < max_cols; cols *= 2) {
      // calculating size
      size = rows * cols;
      double factor = 1.;
      
      // allocating arrays host
      double* h_lhs = (double*) malloc(size*sizeof(double));
      double* h_rhs = (double*) malloc(size*sizeof(double));
      double* h_res = (double*) malloc(size*sizeof(double));
      
      // allocating arrays device
      double* d_lhs;
      double* d_rhs;
      double* d_res;
      CHECK(cudaMalloc((void**)&d_lhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_rhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_res, size*sizeof(double)));
      
      // allocate comparison array
      double* c_res = (double*) malloc(size*sizeof(double));
      
      // filling in lhs and rhs and result
      create_random_matrix(h_lhs, size, min, max);
      create_random_matrix(h_rhs, size, min, max);
      memset(h_res, 0, size*sizeof(h_res[0]));
      
      // copying arrays to device
      copy_host_to_device_double(h_lhs, d_lhs, size);
      copy_host_to_device_double(h_rhs, d_rhs, size);
      copy_host_to_device_double(h_res, d_res, size);
            
      // calculating host and device result
      mulAdd_cpu  (h_res, h_lhs, h_rhs, factor, size);
      mulAdd_onDev(d_res, d_lhs, d_rhs, factor, size, dummy_threads_block);
      
      // copying result back to host
      copy_device_to_host_double(d_res, c_res, size);
      
      // check for error
      compare_matrices_error_printless(result, h_res, c_res, size);
      if (result[1] > DBL_EPSILON) {
	printf("____________________________________________________\n");
	printf("For rows : %d | cols : %d MULADD ERROR\n", rows, cols);
	printf("min_err      : %.17f\n", result[0]);
	printf("max_err      : %.17f\n", result[1]);
	printf("mean_abs_err : %.17f\n", result[2]);
	printf("mean_rel_err : %.17f\n", result[3]);
	// return EXIT_FAILURE;
      }
            
      // freeing up the memory
      free(h_lhs);
      free(h_rhs);
      free(h_res);
      free_double_memory_device(d_lhs);
      free_double_memory_device(d_rhs);
      free_double_memory_device(d_res);
      free(c_res);
    }
  }
  printf("MULADD : SUCCESS\n");
  
  // ADD_ALONG_COL_DIRECT 
  for (int rows = 1; rows < max_rows; rows *= 2) {
    for (int cols = 1; cols < max_cols; cols *= 2) {
      // calculating size
      size = rows * cols;
      
      // allocating arrays host
      double* h_mat = (double*) malloc(size*sizeof(double));
      double* h_vec = (double*) malloc(cols*sizeof(double));
      
      // allocating arrays device
      double* d_mat;
      double* d_vec;
      CHECK(cudaMalloc((void**)&d_mat, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_vec, cols*sizeof(double)));
      
      // allocate comparison array
      double* c_mat = (double*) malloc(size*sizeof(double));
      
      // filling in lhs and rhs and result
      create_random_matrix(h_mat, size, min, max);
      create_random_matrix(h_vec, cols, min, max);
            
      // copying arrays to device
      copy_host_to_device_double(h_mat, d_mat, size);
      copy_host_to_device_double(h_vec, d_vec, cols);
      
      // calculating host and device result
      add_along_col_direct_cpu  (h_mat, h_vec, rows, cols);
      add_along_col_direct_onDev(d_mat, d_vec, rows, cols);
      
      // copying result back to host
      copy_device_to_host_double(d_mat, c_mat, size);
            
      // check for error
      if (!double_equal(h_mat, c_mat, size, sqrt(2)*DBL_EPSILON)) {
	printf("For rows : %d | cols : %d ADD ALONG COL and 0 not compatible\n", rows, cols);
	return EXIT_FAILURE;
      }
            
      // freeing up the memory
      free(h_mat);
      free(h_vec);
      free_double_memory_device(d_mat);
      free_double_memory_device(d_vec);
      free(c_mat);
    }
  }
  printf("ADD ALONG COL : SUCCESS\n");

  // MULADD_DIRECT
  for (int rows = 1; rows < max_rows; rows *= 2) {
    for (int cols = 1; cols < max_cols; cols *= 2) {
      // calculating size
      size = rows * cols;
      double factor = 1.;
      
      // allocating arrays host
      double* h_res = (double*) malloc(size*sizeof(double));
      double* h_rhs = (double*) malloc(size*sizeof(double));
      
      // allocating arrays device
      double* d_res;
      double* d_rhs;
      CHECK(cudaMalloc((void**)&d_rhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&d_res, size*sizeof(double)));
      
      // allocate comparison array
      double* c_res = (double*) malloc(size*sizeof(double));
      
      // filling in lhs and rhs and result
      create_random_matrix(h_res, size, min, max);
      create_random_matrix(h_rhs, size, min, max);
            
      // copying arrays to device
      copy_host_to_device_double(h_res, d_res, size);
      copy_host_to_device_double(h_rhs, d_rhs, size);
                  
      // calculating host and device result
      mulAdd_direct_cpu  (h_res, h_rhs, factor, size);
      mulAdd_direct_onDev(d_res, d_rhs, factor, size, dummy_threads_block);
      
      // copying result back to host
      copy_device_to_host_double(d_res, c_res, size);
      
      // check for error
      if (!double_equal(h_res, c_res, size, 2*sqrt(2)*DBL_EPSILON)) {
	printf("max_abs_diff : %f", max_abs_diff(h_res, c_res, size));
	printf("For rows : %d | cols : %d MULADD_DIRECT and 0 not compatible\n", rows, cols);
	return EXIT_FAILURE;
      }
      
      // freeing up the memory
      free(h_res);
      free(h_rhs);
      free_double_memory_device(d_res);
      free_double_memory_device(d_rhs);
      free(c_res);
    }
  }
  printf("MULADD_DIRECT : SUCCESS\n");
  return 0;
}
