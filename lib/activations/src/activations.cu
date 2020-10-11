
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#include "activations.h"
#include "common.h"
#include "matrix_operator.h"
#include "kernel_utils.h"
#include "global.h"

// host helper functions
// determine max in data
double get_max(const double *data, int length){
    double max = 0;
    for (int i = 0; i < length; i++) max=(data[i]>max ? data[i]: max);
    return max;
}

// // apply pointwise function on host
void apply_pointwise(const double* in,double* out,int size,pointwise func){
    for(int i=0;i<size;i++) out[i]=func(in[i]);
}


// // device helper functions
void apply_pointwise_onDev(const double *dev_in, double *dev_out, int size,int func_int){

  dim3 grid=pointwise_grid(size);
  apply_pointwise_kernel<<<grid,get_pointwise_block()>>>(dev_in, dev_out, size,func_int);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------------
// relu activation and derivative and backprop on host and device (gpu here means onDev)
void relu_activation_gpu(const double *dev_in, double *dev_out, int size){
  apply_pointwise_onDev(dev_in,dev_out,size,RELU_FUNC);
}

void d_relu_activation_gpu(const double *dev_in, double *dev_delta, int size){
  apply_pointwise_onDev(dev_in,dev_delta,size,D_RELU_FUNC);
}

void relu_activation_backprop_gpu(const double *dev_da,double *dev_z,double *dev_dz,int size){
  dim3 grid=pointwise_grid(size);
  hadamard_func_kernel<<<grid,get_pointwise_block()>>>(dev_da, dev_z, dev_dz, size,D_RELU_FUNC);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

void relu_activation_cpu(const double *in, double *out, int size){
  apply_pointwise(in,out,size,&relu);
}

void d_relu_activation_cpu(const double *in, double *delta, int size){
  apply_pointwise(in,delta,size,&d_relu);
}

void relu_activation_backprop_cpu(const double *da,double *z,double *dz,int size){
  double *d_relu=(double *)malloc(size*sizeof(double));
  d_relu_activation_cpu(z,d_relu,size);
  matrix_hadamard_cpu (dz, da, d_relu, size);
  free(d_relu);
}


// ----------------------------------------------------------------------------------
// sigmoid activation and derivative and backprop on host and device (gpu here means onDev)
void sigmoid_activation_gpu(const double *dev_in, double *dev_out, int size){
    apply_pointwise_onDev(dev_in,dev_out,size,SIGMOID_FUNC);
}

void d_sigmoid_activation_gpu(const double *dev_in, double *dev_delta, int size){
    apply_pointwise_onDev(dev_in,dev_delta,size,D_SIGMOID_FUNC);
}

void sigmoid_activation_backprop_gpu(const double *dev_da,double *dev_z,double *dev_dz,int size){
    dim3 grid=pointwise_grid(size);
    hadamard_func_kernel<<<grid,get_pointwise_block()>>>(dev_da, dev_z, dev_dz, size,D_SIGMOID_FUNC);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}

void sigmoid_activation_cpu(const double *in,double *out,int size){
  apply_pointwise(in,out,size,&sigmoid);
}

void d_sigmoid_activation_cpu(const double *in,double *delta,int size){
  apply_pointwise(in,delta,size,&d_sigmoid);
}

void sigmoid_activation_backprop_cpu(const double *da,double *z,double *dz,int size){
  double *d_sigmoid=(double *)malloc(size*sizeof(double));
  d_sigmoid_activation_cpu(z,d_sigmoid,size);
  matrix_hadamard_cpu (dz, da, d_sigmoid, size);
  free(d_sigmoid);
}







// ----------------------------------------------------------------------------------
// softmax activation and derivative and backprop on host and device
void softmax_activation_cpu(const double *in,double *out,int batchsize,int neurons_out){
  // determine max input
  double max = get_max(in, batchsize*neurons_out);

  // calculate exponentials and normalisation
  for (int i = 0; i < batchsize*neurons_out; i++) {
    out[i] = exp(in[i]-max);
  }

  double *sum=(double *)malloc(batchsize*sizeof(double));
  add_reduce_dim_cpu(out,sum, batchsize,neurons_out, 1, batchsize);

  // normalise ie scale along axis
  for (int i = 0; i < batchsize; i++){
    for (int j = 0; j < neurons_out; j++){
        out[i*neurons_out+j]/=sum[i];
    }
  }
  free(sum);
}

// d_softmax is a Tensor stufe 3
void d_softmax_activation_cpu(const double *in, double *delta, int batchsize,int neurons_out){
  double *softmax=(double *)malloc(batchsize*neurons_out*sizeof(double));
  softmax_activation_cpu(in,softmax,batchsize,neurons_out);


  for (int k=0;k<batchsize;k++){
    for (int i = 0; i < neurons_out; i++){
      for (int j = 0; j < neurons_out; j++){
        delta[k*neurons_out*neurons_out+i*neurons_out+j] = (i==j)*softmax[k*neurons_out+i]-softmax[k*neurons_out+i]*softmax[k*neurons_out+j];
      }
    }
  }
  free(softmax);
}

void softmax_activation_backprop_cpu(const double *da,double *z,double *dz,int neurons_out,int batchsize){

  // get softmax derivative
  double *d_softmax=(double *)malloc(batchsize*neurons_out*neurons_out*sizeof(double));
  d_softmax_activation_cpu(z,d_softmax,batchsize,neurons_out);

  //multiply with da
  for(int k=0;k<batchsize;k++){
      matMul(&da[k*neurons_out],&d_softmax[k*neurons_out*neurons_out],1,neurons_out,neurons_out,&dz[k*neurons_out]);
  }
  free(d_softmax);
}

// softmax auf gpu
// on dev reduction setup
double get_max_onDev(const double *dev_in,int size){

  double *dev_res;
  double res[1];
  CHECK(cudaMalloc((void**)&dev_res, sizeof(double)));
  // printf("grid %d %d %d block %d %d %d\n",row_red_grid(size,1).x,row_red_grid(size,1).y,row_red_grid(size,1).z,get_row_red_1d_block().x,get_row_red_1d_block().y,get_row_red_1d_block().z );
  get_max_kernel<<<row_red_grid(size,1),get_row_red_1d_block()>>>(dev_in,size, dev_res);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(&res[0],dev_res , sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dev_res));
  return res[0];
}


void combine_pointwise_onDev_actsoft(const double* dev_in,const double dev_max,double* dev_out,int size){

    dim3 grid=pointwise_grid(size);
    combine_pointwise_kernel_actsoft<<<grid,get_pointwise_block()>>>(dev_in, dev_max,dev_out,size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}


void softmax_activation_onDev(const double *dev_in,double *dev_out,int batchsize,int neurons_out){

  int size =batchsize*neurons_out;

  if (size < SOFTMAX_THRESHOLD){
    double *in,*out;
    out=(double *)malloc(size*sizeof(double));
    in=(double *)malloc(size*sizeof(double));
    CHECK(cudaMemcpy(in, dev_in, size*sizeof(double), cudaMemcpyDeviceToHost));
    softmax_activation_cpu(in,out,batchsize,neurons_out);
    CHECK(cudaMemcpy(dev_out, out, size*sizeof(double), cudaMemcpyHostToDevice));
    free(in);
    free(out);

  }else{

    // determine max input
    double dev_max = get_max_onDev(dev_in,size);

    // calculate exponentials and normalisation
    combine_pointwise_onDev_actsoft(dev_in,dev_max,dev_out,size);
    double *dev_sum;
    CHECK(cudaMalloc((void**)&dev_sum, batchsize*sizeof(double)));
    add_reduce_dim_onDev(dev_out,dev_sum, batchsize,neurons_out, 1,batchsize);
    div_along_axis_onDev(dev_out,dev_sum,dev_out,batchsize,neurons_out, 1, batchsize);

    CHECK(cudaFree(dev_sum));
  }

}


void d_softmax_activation_onDev(const double *dev_in, double *dev_delta, int batchsize,int neurons_out){

  int size =batchsize*neurons_out;

  if (size < D_SOFTMAX_THRESHOLD){
    double *in,*delta;
    in=(double *)malloc(size*sizeof(double));
    delta=(double *)malloc(size*neurons_out*sizeof(double));
    CHECK(cudaMemcpy(in, dev_in, size*sizeof(double), cudaMemcpyDeviceToHost));
    d_softmax_activation_cpu(in, delta, batchsize,neurons_out);
    CHECK(cudaMemcpy(dev_delta, delta, size*neurons_out*sizeof(double), cudaMemcpyHostToDevice));
    free(delta);
    free(in);

  }else{

    double *dev_softmax;
    CHECK(cudaMalloc((void**)&dev_softmax, batchsize*neurons_out*sizeof(double)));
    softmax_activation_onDev(dev_in,dev_softmax,batchsize,neurons_out);

    // d_softmax_activation_kernel<<<pointwise3d_grid(batchsize,neurons_out,neurons_out),get_pointwise3d_block()>>>(dev_softmax,dev_delta, batchsize,neurons_out);
    d_softmax_activation_unrolled_kernel<<<pointwise_grid(size),get_pointwise_block()>>>(dev_softmax,dev_delta, batchsize,neurons_out);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaFree(dev_softmax));
  }
}


// softmax backprop on device
void softmax_activation_backprop_onDev(const double *dev_da,const double *dev_z,double *dev_dz,int neurons_out,int batchsize){

    int size =batchsize*neurons_out;

    if (size < SOFTMAX_BB_THRESHOLD){
      double *da,*z,*dz;
      da=(double *)malloc(size*sizeof(double));
      z=(double *)malloc(size*sizeof(double));
      dz=(double *)malloc(size*sizeof(double));
      CHECK(cudaMemcpy(da, dev_da, size*sizeof(double), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(z, dev_z, size*sizeof(double), cudaMemcpyDeviceToHost));

      softmax_activation_backprop_cpu(da,z,dz,neurons_out,batchsize);
      CHECK(cudaMemcpy(dev_dz, dz, size*sizeof(double), cudaMemcpyHostToDevice));

      free(da);
      free(z);
      free(dz);


    }else{
      // get softmax derivative
      double *dev_d_softmax;
      CHECK(cudaMalloc((void**)&dev_d_softmax, batchsize*neurons_out*neurons_out*sizeof(double)));
      d_softmax_activation_onDev(dev_z,dev_d_softmax,batchsize,neurons_out);

      //multiply with da

      // double *dev_da_T;
      // CHECK(cudaMalloc((void**)&dev_da_T,batchsize*neurons_out*sizeof(double)));
      // mat_transpose_kernel<<<pointwise2d_noOpTh_grid(neurons_out,batchsize), get_pointwise2d_block()>>>(dev_da, dev_da_T, batchsize, neurons_out);
      // CHECK(cudaDeviceSynchronize());
      // softmax_backprop_T_kernel<<<matrix_mul_grid(neurons_out,batchsize),get_matrix_mul_block()>>>(dev_da_T,dev_d_softmax,batchsize, neurons_out,dev_dz);

      softmax_backprop_kernel<<<matrix_mul_grid(neurons_out,batchsize),get_matrix_mul_block()>>>(dev_da,dev_d_softmax,batchsize, neurons_out,dev_dz);

      CHECK(cudaDeviceSynchronize());
      CHECK(cudaGetLastError());

      CHECK(cudaFree(dev_d_softmax));
  }
}
