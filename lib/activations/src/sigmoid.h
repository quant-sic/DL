/*
  CLASS FILE FOR THE SIGMOID

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include "global.h"
#include "layers.h"
#include "common_utils.h"
#include "device_utils.h"
#include "host_utils.h"

#include <math.h>

//_________________________________________________________________________________________________
// class for the sigmoid layer
class sigmoid : public layer
{
private :
  matrix a;
  matrix z;
  matrix dz;

public :
  // constructor / destructor
  sigmoid (std::string name) { this->_name = name; 	this->_type = SIGMOID;};
  ~sigmoid(void) {};

  // back and forward propagation
  matrix& prop_forward (matrix& z, bool flag_host = true);
  matrix& prop_backward(matrix& da, double learning_rate = 0.01, bool flag_host = true);

  // operator overloading
  void print_out(std::ostream& out) const { out << "sigmoid"; };
};



// _________________________________________________________________________________________________________________________
// activation definitions

template <class real_type>
__device__ __host__ real_type sigmoid_func(real_type x){
  return 1/(1+exp(-x));
}

template <class real_type>
__device__ __host__ real_type d_sigmoid_func(real_type x){
  return exp(-x)/((1+exp(-x))*(1+exp(-x)));
}

// create static pointers to relu and d_relu device functions
template <class real_type>
__device__ pointwise_func<real_type> p_sigmoid_func = sigmoid_func<real_type>;

template <class real_type>
__device__ pointwise_func<real_type> p_d_sigmoid_func = d_sigmoid_func<real_type>;


template <class real_type>
void sigmoid_activation_onDev(const real_type *dev_in, real_type *dev_out, int size){
  
  pointwise_func<real_type> dev_sigmoid_func;
  cudaMemcpyFromSymbol(&dev_sigmoid_func, p_sigmoid_func<real_type>, sizeof(pointwise_func<real_type>));
  apply_pointwise_onDev<real_type>(dev_in, dev_out, size, dev_sigmoid_func);
}

template <class real_type>
void d_sigmoid_activation_onDev(const real_type *dev_in, real_type *dev_delta, int size){
  pointwise_func<real_type> dev_d_sigmoid_func;
  cudaMemcpyFromSymbol(&dev_d_sigmoid_func, p_d_sigmoid_func<real_type>, sizeof(pointwise_func<real_type>));
  apply_pointwise_onDev<real_type>(dev_in, dev_delta, size, dev_d_sigmoid_func);
}

template <class real_type>
void sigmoid_activation_backprop_onDev(const real_type *dev_da,real_type *dev_z,real_type *dev_dz,int size){
  pointwise_func<real_type> dev_d_sigmoid_func;
  cudaMemcpyFromSymbol(&dev_d_sigmoid_func, p_d_sigmoid_func<real_type>, sizeof(pointwise_func<real_type>));
  hadamard_func_rhs_kernel<real_type><<<pointwise_grid(size),get_pointwise_block()>>>(dev_da, dev_z, dev_dz, size,dev_d_sigmoid_func);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

template <class real_type>
void sigmoid_activation_cpu(const real_type *in,real_type *out,int size){
  apply_pointwise<real_type>(in, out, size, &sigmoid_func);
}

template <class real_type>
void d_sigmoid_activation_cpu(const real_type *in,real_type *delta,int size){
  apply_pointwise<real_type>(in, delta, size, &d_sigmoid_func);
}

template <class real_type>
void sigmoid_activation_backprop_cpu(const real_type *da,real_type *z,real_type *dz,int size){
  hadamard_func_rhs<real_type>(da, z, dz, size, &d_sigmoid_func);
}


#endif // _SIGMOID_H_
