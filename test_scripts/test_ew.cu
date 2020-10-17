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
#include "common.h"
#include "global.h"
#include "test_matrix_operator.h"
#include "pw_comp.h"
#include "common_utils.h"
#include "reduce.h"

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

    apply_pointwise_cpu<double>(lhs1,lhs1s,size,scale_functor<double>(alpha));
    apply_pointwise_cpu<double>(lhs2,lhs2s,size,scale_functor<double>(beta));
    apply_pointwise_cpu<double>(rhs1,rhs1s,size,scale_functor<double>(gamma));
    apply_pointwise_cpu<double>(rhs2,rhs2s,size,scale_functor<double>(delta));

    combine_pointwise_cpu<double>(lhs1s,lhs2s,res1,size,add_functor<double>());
    combine_pointwise_cpu<double>(rhs1s,rhs2s,res2,size,add_functor<double>());

    combine_pointwise_cpu<double>(res1,res2,res5,size,mul_functor<double>());

    // first multiply then scale then add
    combine_pointwise_cpu<double>(lhs1,rhs1,res1,size,mul_functor<double>());
    combine_pointwise_cpu<double>(lhs1,rhs2,res2,size,mul_functor<double>()),
    combine_pointwise_cpu<double>(lhs2,rhs1,res3,size,mul_functor<double>());
    combine_pointwise_cpu<double>(lhs2,rhs2,res4,size,mul_functor<double>());

    apply_pointwise_cpu<double>(res1,res1,size,scale_functor<double>(alpha*gamma));
    apply_pointwise_cpu<double>(res2,res2,size,scale_functor<double>(alpha*delta));
    apply_pointwise_cpu<double>(res3,res3,size,scale_functor<double>(gamma*beta));
    apply_pointwise_cpu<double>(res4,res4,size,scale_functor<double>(delta*beta));

    combine_pointwise_cpu<double>(res1,res2,res6,size,add_functor<double>());
    combine_pointwise_cpu<double>(res6,res3,res6,size,add_functor<double>());
    combine_pointwise_cpu<double>(res4,res6,res6,size,add_functor<double>());


    if(!double_equal(res5,res6,size,4*CEW_THRESHOLD)){
        printf("Distributivgesetz does not hold between scale, hadamrd and add at size: %d\n", size);
    }

    // check pointwise reductions to scalar
    CHECK(cudaMalloc((void**)&dev_rhs1, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_lhs1, size*sizeof(double)));

    copy_host_to_device_double(rhs1,dev_rhs1,size);
    copy_host_to_device_double(lhs1,dev_lhs1,size);

    // sum func reduction
    double sf_onDev=sum_func_array_onDev<double>(dev_lhs1,dev_rhs1,size,add_functor<double>());
    double sf_host=sum_func_array<double>(lhs1,rhs1,size,add_functor<double>());

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
