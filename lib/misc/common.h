/*
  COMMON FUNCTIONS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 21.08.2020
  TO-DO   :
  CAUTION :
*/
#ifndef _COMMON_H_
#define _COMMON_H_

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
  //_______________________________________________________________________________________________
  // check cuda errors
#define CHECK(call)							\
  {									\
    const cudaError_t error = call;					\
    if (error != cudaSuccess)						\
      {									\
	fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);		\
	fprintf(stderr, "code: %d, reason: %s\n", error,		\
		cudaGetErrorString(error));				\
	exit(1);							\
      }									\
  }

  //_______________________________________________________________________________________________
  // misc functions

  double seconds(void);
  void   copy_host_to_device_double(double* ptr_hos, double* ptr_dev, int size);
  void   copy_device_to_host_double(double* ptr_dev, double* ptr_hos, int size);
  void   free_double_memory_device(double* ptr_dev);
  void   print_out_matrix(double* mat, int rows, int cols);
  int    double_equal(const double *A,const double *B,int size,double threshold);
  double max_abs_diff(const double *A,const double *B,int size);

#ifdef __cplusplus
}
#endif

#endif // _COMMON_H_
