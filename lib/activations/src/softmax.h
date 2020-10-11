/*
  SOFTMAX LAYER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 26.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "layers.h"
#include "global.h"

#include "pw_comp.h"
#include "pw2d_comp.h"
#include "reduce.h"
#include "common_utils.h"

// define algorithm thresholds
#define SOFTMAX_THRESHOLD (1<<3)
#define SOFTMAX_BB_THRESHOLD (1<<3)
#define D_SOFTMAX_THRESHOLD (1<<3)


//_________________________________________________________________________________________________
// class for the softmax layer
class softmax : public layer
{
private :
  matrix a;
  matrix z;
  matrix dz;

public :
  // constructor / destructor
  softmax(std::string name) { this->_name = name; 	this->_type = SOFTMAX;};
  ~softmax(void) {};

  // back and forward propagation
  matrix& prop_forward (matrix& z, bool flag_host = true);
  matrix& prop_backward(matrix& da, double learning_rate = 0.01, bool flag_host = true);

  // operator overloading
  void print_out(std::ostream& out) const { out << "softmax"; };
};


// ----------------------------------------------------------------------------------
// softmax activation and derivative and backprop on host and device
template<typename real_type>
void softmax_activation_cpu(const real_type *in,real_type *out,int batchsize,int neurons_out){
  // determine max input
  real_type max = get_max<real_type>(in, batchsize*neurons_out);

  // calculate exponentials and normalisation
  apply_pointwise<real_type>(in,out,batchsize*neurons_out,exp_sub_val_functor<real_type>(max));


  real_type *sum=(real_type *)malloc(batchsize*sizeof(real_type));
  add_reduce_dim_cpu<real_type>(out,sum, batchsize,neurons_out, 1, batchsize);

  // normalise ie scale along axis
  func_along_axis_cpu(div_functor<real_type>(), out, sum, out,batchsize,neurons_out,1,batchsize);
  
  free(sum);
}

// // d_softmax is a Tensor stufe 3
template<typename real_type>
void d_softmax_activation_cpu(const real_type *in, real_type *delta, int batchsize,int neurons_out){
  real_type *softmax=(real_type *)malloc(batchsize*neurons_out*sizeof(real_type));
  softmax_activation_cpu<real_type>(in,softmax,batchsize,neurons_out);


  for (int k=0;k<batchsize;k++){
    for (int i = 0; i < neurons_out; i++){
      for (int j = 0; j < neurons_out; j++){
        delta[k*neurons_out*neurons_out+i*neurons_out+j] = (i==j)*softmax[k*neurons_out+i]-softmax[k*neurons_out+i]*softmax[k*neurons_out+j];
      }
    }
  }
  free(softmax);
}

template<typename real_type>
void softmax_activation_backprop_cpu(const real_type *da,real_type *z,real_type *dz,int neurons_out,int batchsize){

  // get softmax derivative
  real_type *d_softmax=(real_type *)malloc(batchsize*neurons_out*neurons_out*sizeof(real_type));
  d_softmax_activation_cpu(z,d_softmax,batchsize,neurons_out);

  //multiply with da
  for(int k=0;k<batchsize;k++){
      matMul(&da[k*neurons_out],&d_softmax[k*neurons_out*neurons_out],1,neurons_out,neurons_out,&dz[k*neurons_out]);
  }
  free(d_softmax);
}

// // softmax auf gpu
template<typename real_type>
void softmax_activation_onDev(const real_type *dev_in,real_type *dev_out,int batchsize,int neurons_out){

  int size = batchsize*neurons_out;

  if (size < SOFTMAX_THRESHOLD){
    real_type *in,*out;
    out=(real_type *)malloc(size*sizeof(real_type));
    in=(real_type *)malloc(size*sizeof(real_type));
    CHECK(cudaMemcpy(in, dev_in, size*sizeof(real_type), cudaMemcpyDeviceToHost));
    softmax_activation_cpu<real_type>(in,out,batchsize,neurons_out);
    CHECK(cudaMemcpy(dev_out, out, size*sizeof(real_type), cudaMemcpyHostToDevice));
    free(in);
    free(out);

  }else{

    // determine max input
    real_type dev_max = get_max_onDev<real_type>(dev_in,size);

    // calculate exponentials and normalisation
    apply_pointwise_onDev<real_type>(dev_in,dev_out,size,exp_sub_val_functor<real_type>(dev_max));

    real_type *dev_sum;
    CHECK(cudaMalloc((void**)&dev_sum, batchsize*sizeof(real_type)));
    add_reduce_dim_onDev<real_type>(dev_out,dev_sum, batchsize,neurons_out, 1,batchsize);
    func_along_axis_onDev<real_type>(div_functor<real_type>(),dev_out,dev_sum,dev_out,batchsize,neurons_out, 1, batchsize);
    CHECK(cudaFree(dev_sum));
  }

}


template<typename real_type>
__global__ void d_softmax_activation_unrolled_kernel(const real_type * __restrict__ dev_softmax, real_type *dev_delta, int batchsize,int neurons_out){

  // assign softmax to 3 tensor d_softmax
  int j,k_i;
  for (unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x; (idx < batchsize*neurons_out*neurons_out); idx += blockDim.x*gridDim.x){
    // define index helpers
    k_i=idx/neurons_out;
    j=idx%neurons_out;

    // assign 
    dev_delta[idx]=((k_i%neurons_out)==j)*dev_softmax[k_i] -dev_softmax[k_i]*dev_softmax[(k_i/neurons_out)*neurons_out+j];
  }
}


template<typename real_type>
void d_softmax_activation_onDev(const real_type *dev_in, real_type *dev_delta, int batchsize,int neurons_out){

  int size =batchsize*neurons_out;

  if (size < D_SOFTMAX_THRESHOLD){
    real_type *in,*delta;
    in=(real_type *)malloc(size*sizeof(real_type));
    delta=(real_type *)malloc(size*neurons_out*sizeof(real_type));
    CHECK(cudaMemcpy(in, dev_in, size*sizeof(real_type), cudaMemcpyDeviceToHost));
    d_softmax_activation_cpu<real_type>(in, delta, batchsize,neurons_out);
    CHECK(cudaMemcpy(dev_delta, delta, size*neurons_out*sizeof(real_type), cudaMemcpyHostToDevice));
    free(delta);
    free(in);

  }else{

    real_type *dev_softmax;
    CHECK(cudaMalloc((void**)&dev_softmax, batchsize*neurons_out*sizeof(real_type)));
    softmax_activation_onDev<real_type>(dev_in,dev_softmax,batchsize,neurons_out);

    d_softmax_activation_unrolled_kernel<real_type><<<pointwise_grid(size),get_pointwise_block()>>>(dev_softmax,dev_delta, batchsize,neurons_out);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaFree(dev_softmax));
  }
}

// acts out multiplications on submatrices of a 3 tensor dev_d_softmax with matrix dev_da
template<typename real_type>
__global__ void softmax_backprop_kernel(const real_type * __restrict__ dev_da,const real_type * __restrict__ dev_d_softmax,int batchsize, int neurons_out,real_type *dev_dz){

  //get sample and output neuron
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int k = blockIdx.y*blockDim.y+threadIdx.y;

  const real_type *dev_d_softmax_b=&dev_d_softmax[k*neurons_out*neurons_out];

  if(k<batchsize && j<neurons_out){

    // sum
    real_type sum=0;
    for(int ki=0; ki < neurons_out;ki++){
      sum+=dev_da[k*neurons_out+ki]*dev_d_softmax_b[ki*neurons_out+j];
    }
    // store result of matrix multiplication
    dev_dz[k*neurons_out+j]=sum;
  }
}

// // softmax backprop on device
template<typename real_type>
void softmax_activation_backprop_onDev(const real_type *dev_da,const real_type *dev_z,real_type *dev_dz,int neurons_out,int batchsize){

    int size =batchsize*neurons_out;

    if (size < SOFTMAX_BB_THRESHOLD){
      real_type *da,*z,*dz;
      da=(real_type *)malloc(size*sizeof(real_type));
      z=(real_type *)malloc(size*sizeof(real_type));
      dz=(real_type *)malloc(size*sizeof(real_type));
      CHECK(cudaMemcpy(da, dev_da, size*sizeof(real_type), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(z, dev_z, size*sizeof(real_type), cudaMemcpyDeviceToHost));

      softmax_activation_backprop_cpu<real_type>(da,z,dz,neurons_out,batchsize);
      CHECK(cudaMemcpy(dev_dz, dz, size*sizeof(real_type), cudaMemcpyHostToDevice));

      free(da);
      free(z);
      free(dz);


    }else{
      // get softmax derivative
      real_type *dev_d_softmax;
      CHECK(cudaMalloc((void**)&dev_d_softmax, batchsize*neurons_out*neurons_out*sizeof(real_type)));
      d_softmax_activation_onDev<real_type>(dev_z,dev_d_softmax,batchsize,neurons_out);

      //multiply with da
      softmax_backprop_kernel<real_type><<<matrix_mul_grid(neurons_out,batchsize),get_matrix_mul_block()>>>(dev_da,dev_d_softmax,batchsize, neurons_out,dev_dz);

      CHECK(cudaDeviceSynchronize());
      CHECK(cudaGetLastError());

      CHECK(cudaFree(dev_d_softmax));
  }
}



























#endif // _SOFTMAX_H_
