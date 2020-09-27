#include <iostream>
#include <math.h>
#include "costfunctions.h"
#include "../global.h"
#include "../common.h"
#include "../layers/activations.h"
#include "../matrix_operations/kernel_utils.h"

// helper functions for host
void combine_pointwise(const double* in,const double* target,double* delta,int size,comb_pointwise func){
    for(int i=0;i<size;i++) delta[i]=func(in[i],target[i]);
}

double sum_func_array(const double *in, const double* target, comb_pointwise func, int size){
  double res = 0.;
  // add terms onto res
  for (int k = 0; k < size; k++) {
    res += func(in[k], target[k]);
  }
  return res;
}

// helper functions for Device
void combine_pointwise_onDev(const double* dev_in,const double* dev_target,double* dev_delta,int size,int func_int){

    comb_pointwise_1d_kernel<<<pointwise_grid(size),get_pointwise_block()>>>(dev_delta,dev_in, dev_target,size, func_int);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}

// on dev reduction setup
double sum_func_array_onDev(const double *dev_in,const double* dev_target,int func_int,int size){

  double *dev_res;
  double res[1];
  CHECK(cudaMalloc((void**)&dev_res, sizeof(double)));
  add_reduce_rows_func_kernel<<<row_red_grid(size,1),get_row_red_1d_block()>>>(dev_in, dev_target,dev_res,size, func_int);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(&res[0],dev_res , sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dev_res));
  return res[0];
}


// ----------------------------------------------------------------------------------
// Categorical Crossentropy and derivative ond host and device
double categorical_crossentropy(const double *in,double *target,int size,int batchsize){
  double value = sum_func_array(in, target, &summand_cat_cross_ent, size)/((double)size);
  return value;
}

void d_categorical_crossentropy(const double *in,double *target,double *delta,int size){
  combine_pointwise(in,target,delta,size,(comb_pointwise)&d_cat_cross_ent) ;
}

double categorical_crossentropy_onDev(const double *dev_in, double *dev_target, int size,int batchsize){
    return sum_func_array_onDev(dev_in,dev_target,SUMM_CAT,size)/size;
}

void d_categorical_crossentropy_onDev(const double *dev_in, double *dev_target, double *dev_delta, int size){
    combine_pointwise_onDev(dev_in,dev_target,dev_delta,size,D_CAT_CROSS_ENT);
}



// ----------------------------------------------------------------------------------
// RMS and derivative on host and device
double rms(const double *in,double *target,int size,int batchsize){
  return sum_func_array(in,target,&summand_rms,size)/size;
}

void d_rms(const double *in, double *target, double *delta, int size){
  combine_pointwise(in,target,delta,size,(comb_pointwise)&d_rms_func);
}

// on device
double rms_onDev(const double *dev_in, double *dev_target, int size,int batchsize){
  return sum_func_array_onDev(dev_in,dev_target,SUMM_RMS,size)/size;
}

void d_rms_onDev(const double *dev_in, double *dev_target, double *dev_delta, int size){
  combine_pointwise_onDev(dev_in,dev_target,dev_delta,size,D_RMS);
}


// !!!!!!!!!!!!!!CAUTION Target mus be normalised distribution
// _______________________________________________
// CCE, SOFTMAX Combination on host and device
double cce_softmax_cpu(const double *in,double *target,int size,int batchsize){
  double *softmax=(double *)malloc(size*sizeof(double));
  softmax_activation_cpu(in,softmax,batchsize,size/batchsize);
  double cce=categorical_crossentropy(softmax,target,size,batchsize);
  free(softmax);
  return cce;
}

void d_cce_softmax_cpu(const double *in, double *target, double *delta, int size, int batchsize){
  double *softmax=(double *)malloc(size*sizeof(double));
  softmax_activation_cpu(in,softmax,batchsize,size/batchsize);
  combine_pointwise(softmax,target,delta,size,&d_cce_softmax);
  free(softmax);
}

double cce_softmax_onDev(const double *dev_in,double *dev_target,int size,int batchsize){
  double *dev_softmax;
  CHECK(cudaMalloc((void**)&dev_softmax, size*sizeof(double)));
  softmax_activation_onDev(dev_in,dev_softmax,batchsize,size/batchsize);
  double cce=categorical_crossentropy_onDev(dev_softmax,dev_target,size,batchsize);
  cudaFree(dev_softmax);
  return cce;
}

void d_cce_softmax_onDev(const double *dev_in, double *dev_target, double *dev_delta, int size, int batchsize){
  double *dev_softmax;
  CHECK(cudaMalloc((void**)&dev_softmax, size*sizeof(double)));
  softmax_activation_onDev(dev_in,dev_softmax,batchsize,size/batchsize);
  combine_pointwise_onDev(dev_softmax,dev_target,dev_delta,size,D_CCE_SOFTMAX);
  cudaFree(dev_softmax);
}
