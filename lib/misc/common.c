/*
  COMMON FUNCTIONS C FILE

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 21.08.2020
  TO-DO   :
  CAUTION :
*/

#include "common.h"
#include <math.h>

//_________________________________________________________________________________________________
// fucntion to get a time point in seconds
//EXTERN double seconds();
double seconds(void)
{
  struct timeval tp;
  struct timezone tzp;
  gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


//_________________________________________________________________________________________________
// function to allocate memory on device and get pointer fro c++ functions
//EXTERN double* alloc_double_memory_device(int size);
void alloc_double_memory_device(double* ptr_dev,
				int     size)
{
  CHECK(cudaMalloc((void**)&ptr_dev, size*sizeof(double)));
}

void alloc_double_memory_device_ptr(int size,double *ptr_dev)
{
  CHECK(cudaMalloc((void**)&ptr_dev, size*sizeof(double)));
}

//_________________________________________________________________________________________________
// copy host to device memory
//EXTERN void copy_host_to_device_double(double* ptr_hos, double* ptr_dev, int size);
void copy_host_to_device_double(double* ptr_hos,
				double* ptr_dev,
				int     size)
{
  CHECK(cudaMemcpy(ptr_dev, ptr_hos, size*sizeof(double), cudaMemcpyHostToDevice));
}

//_________________________________________________________________________________________________
// copy device to host memory
//EXTERN void copy_device_to_host_double(double* ptr_dev, double* ptr_hos, int size);
void copy_device_to_host_double(double* ptr_dev,
				double* ptr_hos,
				int     size)
{
  CHECK(cudaMemcpy(ptr_hos, ptr_dev, size*sizeof(double), cudaMemcpyDeviceToHost));
}

//_________________________________________________________________________________________________
// free memory from the device
//EXTERN void free_double_memory_device(double* ptr_dev);
void free_double_memory_device(double* ptr_dev)
{
  CHECK(cudaFree(ptr_dev));
}

//________________________________________________________________________________________________
// print a matrix to the console
// maybe do some padding or other stuff here
//EXTERN void print_out_matrix(double* mat, int rows, int cols);
void print_out_matrix(double* mat,
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

//________________________________________________________________________________________________
// check if two double ptr point two space with equal values up to threshold
// maybe do some padding or other stuff here
//EXTERN void print_out_matrix(double* mat, int rows, int cols);
int double_equal(const double *A, const double *B, int size, double threshold){
  int same_result = 1;
  for(int j = 0; j < size; j++){
    if(A[j]) {
      same_result *= fabs((A[j]-B[j])/A[j]) < threshold;
    } else {
      same_result *= fabs(B[j]) < threshold;
    }
  }
  return same_result;
}

double max_abs_diff(const double *A,const double *B,int size){

  double max_d=0;
  double diff;
  for(int i =0;i<size;i++){
     diff=fabs(A[i]-B[i]);
     max_d=(diff>max_d)? diff:max_d;
  }
  return max_d;
}
