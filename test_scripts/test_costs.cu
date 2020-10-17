/*
  TEST THE COSTS SCRIPT

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 27.08.2020
  TO-DO   :
  CAUTION :
*/

// c++ headers
#include <iostream>

// own headers
#include "test_matrix_operator.h"
#include "common.h"
#include <float.h>
#include <math.h>

#include "mse_cost.h"
#include "cce_cost.h"
#include "cce_soft_cost.h"

#include "common_utils.h"

// define thresholds
#define COST_THRESHOLD(size) (sqrt(size)*DBL_EPSILON)
#define D_COST_THRESHOLD(size) (DBL_EPSILON)
#define SOFTMAX_BB_COMP_THRESHOLD(neurons_out) (sqrt(2*neurons_out)*sqrt(3)*(sqrt(2+2*neurons_out))*DBL_EPSILON)


int main(int argc, char **argv)
{
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("\n\nPerfomse the following checks:\n\n - mse, CAT_CROSS_ENT on HOST and Device\n - HOST and DEVICE same result\n - All Cost values positive\n - All Cost have local minimum around Target\n\n_________________________________________________\n");

  double *act_in,*act_out_soft,*act_out,*target,*delta_mse,*delta_cce,*tmp,*tmp2;
  double *dev_act_in,*dev_act_out_soft,*dev_act_out,*dev_delta_mse,*dev_delta_cce,*dev_target,*dev_tmp,*dev_tmp2;

  int neurons_out=100;
  int batchsize=100;
  int size=neurons_out*batchsize;

  act_in=(double *)malloc(size*sizeof(double));
  act_out_soft=(double *)malloc(size*sizeof(double));
  act_out=(double *)malloc(size*sizeof(double));
  target=(double *)malloc(size*sizeof(double));
  tmp=(double *)malloc(size*sizeof(double));
  tmp2=(double *)malloc(size*sizeof(double));
  delta_mse=(double *)malloc(size*sizeof(double));
  delta_cce=(double *)malloc(size*sizeof(double));

  CHECK(cudaMalloc((void**)&dev_tmp2, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_act_out_soft, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_act_in, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_act_out, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_delta_mse, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_delta_cce, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_tmp, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_target, size*sizeof(double)));

  // check if all costs are positive
  int all_positive=1;
  for(int i=0;i<10;i++){

    create_random_matrix(act_out,size,0,1);
    create_random_matrix(target,size,0,1);

    double mse_value=mse<double>(act_out,target,size);

    double cce_value=cce<double>(act_out,target,size);

    double cce_soft_value=cce_soft_cpu<double>(act_out,target,size,batchsize);

    all_positive*=(mse_value>=0 &&cce_value>=0  &&cce_soft_value>=0 );
  }
  printf("All Cost Values positive %d\n",all_positive);



  // check if all costs have a minimum at target
  create_random_matrix(target,size,0.3,0.6);

  double mse_min_value=mse<double>(target,target,size);

  double cce_min_value=cce<double>(target,target,size);

  double cce_soft_min_value=cce_soft_cpu<double>(target,target,size,batchsize);

  int all_min_at_target=1;

  for(int i=0;i<10;i++){

    // create delta
    double *delta=(double *)malloc(size*sizeof(double));
    create_random_matrix(delta,size,0.01,.1);

    // add onto target
    combine_pointwise_cpu<double>(act_out, target, delta,  size,add_functor<double>());


    double mse_value=mse<double>(act_out,target,size);

    double cce_value=cce<double>(act_out,target,size);

    double cce_soft_value=cce_soft_cpu<double>(act_out,target,size,batchsize);

    all_min_at_target*=(mse_value>mse_min_value &&cce_value>cce_min_value  &&cce_soft_value>=cce_soft_min_value);
  }
  printf("All Cost Values Minimum at Target %d\n",all_positive);


  create_random_matrix(act_in,size,0,1);
  create_random_matrix(act_out,size,0,1);
  create_random_matrix(target,size,0,1);

  copy_host_to_device_double(act_in, dev_act_in, size);
  copy_host_to_device_double(act_out, dev_act_out, size);
  copy_host_to_device_double(target, dev_target, size);


// check if costs yield the same result on host and device
  double mse_value=mse<double>(act_out,target,size);
  double mse_value_gpu=mse_onDev<double>(dev_act_out,dev_target,size);
  printf("MSE same result on Host and Device %d\n",double_equal(&mse_value,&mse_value_gpu,1,COST_THRESHOLD(size)));

  double cce_value=cce<double>(act_out,target,size);
  double cce_value_gpu=cce_onDev<double>(dev_act_out,dev_target,size);
  printf("CCE same result on Host and Device %d\n",double_equal(&cce_value,&cce_value_gpu,1,COST_THRESHOLD(size)));

// check if backpropagation yields the same result on host and device
  d_mse<double>(act_out, target, delta_mse, size);
  d_mse_onDev<double>(dev_act_out,dev_target, dev_delta_mse, size);
  copy_device_to_host_double(dev_delta_mse, tmp, size);
  printf("D_MSE same result on Host and Device %d\n",double_equal(tmp,delta_mse,size,D_COST_THRESHOLD(size)));


  d_cce<double>(act_out, target, delta_cce, size);
  d_cce_onDev<double>(dev_act_out,dev_target, dev_delta_cce, size);
  copy_device_to_host_double(dev_delta_cce, tmp, size);
  printf("D_CCE same result on Host and Device %d\n",double_equal(tmp,delta_cce,size,D_COST_THRESHOLD(size)));


// ___________________________________________________________________________________
// combined. Check if cce_softmax give the same result

// normalise target
  double sum_target=0;
  for(int k=0;k<batchsize;k++){
      sum_target=0;
      for(int i =0;i<neurons_out;i++) sum_target+=target[k*neurons_out+i];
      for(int i =0;i<neurons_out;i++) target[k*neurons_out+i]/=sum_target;
  }

  softmax_activation_cpu<double>(act_in,act_out_soft,batchsize,neurons_out);
  d_cce<double>(act_out_soft, target, delta_cce, size);
  softmax_activation_backprop_cpu<double>(delta_cce,act_in,tmp2,neurons_out,batchsize);
  d_cce_soft_cpu<double>(act_in,target,tmp,size,batchsize);

  printf("BackProp cce_softmax and cce + softmax on Host max absolute difference %e\n",max_abs_diff(tmp,tmp2,size) );
  printf("BackProp cce_softmax and cce + softmax same result on HOST %d\n",double_equal(tmp,tmp2,size,SOFTMAX_BB_COMP_THRESHOLD(neurons_out)));

  copy_host_to_device_double(target,dev_target,size);
  softmax_activation_onDev<double>(dev_act_in,dev_act_out_soft,batchsize,neurons_out);
  d_cce_onDev<double>(dev_act_out_soft, dev_target, dev_delta_cce, size);
  softmax_activation_backprop_onDev(dev_delta_cce,dev_act_in,dev_tmp2,neurons_out,batchsize);
  d_cce_soft_onDev<double>(dev_act_in,dev_target,dev_tmp,size,batchsize);

  copy_device_to_host_double(dev_tmp, tmp, size);
  copy_device_to_host_double(dev_tmp2, tmp2, size);

  printf("BackProp cce_softmax and cce + softmax on Device max absolute difference %e\n",max_abs_diff(tmp,tmp2,size) );
  printf("BackProp cce_softmax and cce + softmax same result on Device %d\n",double_equal(tmp,tmp2,size,SOFTMAX_BB_COMP_THRESHOLD(neurons_out)));


  free(act_out);
  free(target);
  free(tmp);
  CHECK(cudaFree(dev_tmp));
  CHECK(cudaFree(dev_act_out));
  free(delta_cce);
  CHECK(cudaFree(dev_delta_cce));
  free(delta_mse);
  CHECK(cudaFree(dev_target));
  CHECK(cudaFree(dev_delta_mse));

}
