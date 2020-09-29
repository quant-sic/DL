// standard c headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cblas.h>
#include <float.h>

// own c headers
#include "../common.h"
#include "../global.h"
#include "../matrix_operations/matrix_operator.h"
#include "../matrix_operations/kernel_utils.h"
#include "../matrix_operations/test_matrix_operator.h"
#include "../matrix_operations/matrix_operator_gpu.h"
#include "../layers/activations.h"
#include "../cost/costfunctions.h"

// define thresholds
#define CEW_THRESHOLD (sqrt(2)*DBL_EPSILON)
#define SUM_FUNC_THRESHOLD(size) (sqrt(2*size)*DBL_EPSILON)

int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Testing Pointwise combination at ");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("Performs the following checks:\n\n - __device__ function yields same result \n - (aX1+bX2)(cY1+dY2) holds for add, hadamrd and scale\n - Get_max and sum_func same result on host and Device\n\n_________________________________________________\n");


  srand(seconds());   // Initialization, should only be called once.

  // time pointwise_combine problem
  double *res1,*res3,*res4,*res2,*res5,*res6,*lhs1,*rhs1,*lhs2,*rhs2,*lhs1s,*rhs1s,*lhs2s,*rhs2s;
  double *dev_lhs1,*dev_rhs1;

  // 1.Difference between using __device__ function and normal

  for(int size=1;size<=1<<12;size*=2){

    res1=(double *)malloc(size*sizeof(double));
    res2=(double *)malloc(size*sizeof(double));
    res3=(double *)malloc(size*sizeof(double));
    res4=(double *)malloc(size*sizeof(double));
    res5=(double *)malloc(size*sizeof(double));
    res6=(double *)malloc(size*sizeof(double));
    lhs2=(double *)malloc(size*sizeof(double));
    rhs2=(double *)malloc(size*sizeof(double));
    lhs1=(double *)malloc(size*sizeof(double));
    rhs1=(double *)malloc(size*sizeof(double));
    lhs2s=(double *)malloc(size*sizeof(double));
    rhs2s=(double *)malloc(size*sizeof(double));
    lhs1s=(double *)malloc(size*sizeof(double));
    rhs1s=(double *)malloc(size*sizeof(double));

    // create random matrices
    create_random_matrix(lhs1,size,0,5);
    create_random_matrix(rhs1,size,0,5);
    create_random_matrix(lhs2,size,0,5);
    create_random_matrix(rhs2,size,0,5);

    // get random scalars
    double alpha,beta,gamma,delta;
    alpha=(5.0*(double)rand()/(double)RAND_MAX);
    beta=(5.0*(double)rand()/(double)RAND_MAX);
    gamma=(5.0*(double)rand()/(double)RAND_MAX);
    delta=(5.0*(double)rand()/(double)RAND_MAX);

    // first scale then add then multiply
    matrix_scalar_cpu(lhs1s,lhs1,alpha,size);
    matrix_scalar_cpu(lhs2s,lhs2,beta,size);
    matrix_scalar_cpu(rhs1s,rhs1,gamma,size);
    matrix_scalar_cpu(rhs2s,rhs2,delta,size);

    matrix_add_cpu(res1,lhs1s,lhs2s,size);
    matrix_add_cpu(res2,rhs1s,rhs2s,size);

    matrix_hadamard_cpu(res5,res1,res2,size);

    // first multiply then scale then add
    matrix_hadamard_cpu(res1,lhs1,rhs1,size);
    matrix_hadamard_cpu(res2,lhs1,rhs2,size);
    matrix_hadamard_cpu(res3,lhs2,rhs1,size);
    matrix_hadamard_cpu(res4,lhs2,rhs2,size);

    matrix_scalar_cpu(res1,res1,alpha*gamma,size);
    matrix_scalar_cpu(res2,res2,alpha*delta,size);
    matrix_scalar_cpu(res3,res3,beta*gamma,size);
    matrix_scalar_cpu(res4,res4,beta*delta,size);

    matrix_add_cpu(res6,res1,res2,size);
    matrix_add_cpu(res6,res6,res3,size);
    matrix_add_cpu(res6,res6,res4,size);

    if(!double_equal(res5,res6,size,4*CEW_THRESHOLD)){
        printf("Distributivgesetz does not hold between scale, hadamrd and add at size: %d\n", size);
    }

    // check pointwise reductions to scalar
    CHECK(cudaMalloc((void**)&dev_rhs1, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_lhs1, size*sizeof(double)));

    copy_host_to_device_double(rhs1,dev_rhs1,size);
    copy_host_to_device_double(lhs1,dev_lhs1,size);

    // sum func reduction
    double sf_onDev=sum_func_array_onDev(dev_lhs1,dev_rhs1,ADD,size);
    double sf_host=sum_func_array(lhs1,rhs1,&add_func,size);

    if(!double_equal(&sf_onDev,&sf_host,1,SUM_FUNC_THRESHOLD(size))){
        printf("sum_func not same resul host and device: %d\n", size);
    }

    // get_max reduction
    double max_onDev=get_max_onDev(dev_lhs1,size);
    double max_host=get_max(lhs1,size);

    if(!double_equal(&max_onDev,&max_host,1,DBL_EPSILON)){
        printf("get_max not same resul host and device: %d\n", size);
    }
  }

  printf("Checks done\n");
}
