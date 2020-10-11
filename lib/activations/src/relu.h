/*
  CLASS FOR THE RELU ACTIVATION LAYER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _RELU_H_
#define _RELU_H_

#include "layers.h"

//_________________________________________________________________________________________________
// class for the relu layer
class relu : public layer
{
private :
  matrix a;
  matrix z;
  matrix dz;

public :
  // contructor destructor
  relu (std::string name) { this->_name = name; 	this->_type = RELU;};
  ~relu(void) {};
  
  // back and forward propagation
  matrix& prop_forward (matrix& z, bool flag_host = true);
  matrix& prop_backward(matrix& da, double learning_rate = 0.01, bool flag_host = true);

  // operatro overloading
  void print_out(std::ostream& out) const { out << "relu"; };
};





// _________________________________________________________________________________________________________________________
// activation defiinitions
template <class real_type>
__device__ __host__ real_type relu_func(real_type x)
{
  return (x > 0 ? x : 0);
}

template <class real_type>
__device__ __host__ real_type d_relu_func(real_type x)
{
  return (x > 0 ? 1 : 0);
}

// create static pointers to relu and d_relu device functions
template <class real_type>
__device__ pointwise<real_type> p_relu_func = relu_func<real_type>;

template <class real_type>
__device__ pointwise<real_type> p_d_relu_func = d_relu_func<real_type>;

// relu activation on device
template <class real_type>
void relu_activation_onDev(const real_type *dev_in, real_type *dev_out, int size)
{

  pointwise<real_type> dev_relu_func;
  cudaMemcpyFromSymbol(&dev_relu_func, p_relu_func<real_type>, sizeof(pointwise<real_type>));
  apply_pointwise_onDev<real_type>(dev_in, dev_out, size, dev_relu_func);
}

// d_relu activation on device
template <class real_type>
void d_relu_activation_onDev(const real_type *dev_in, real_type *dev_delta, int size)
{

  pointwise<real_type> dev_d_relu_func;
  cudaMemcpyFromSymbol(&dev_d_relu_func, p_relu_func<real_type>, sizeof(pointwise<real_type>));
  apply_pointwise_onDev<real_type>(dev_in, dev_delta, size, dev_d_relu_func);
}

// relu activation backprop on device
template <class real_type>
void relu_activation_backprop_onDev(const real_type *dev_da, real_type *dev_z, real_type *dev_dz, int size)
{
  pointwise<real_type> dev_d_relu_func;
  cudaMemcpyFromSymbol(&dev_d_relu_func, p_relu_func<real_type>, sizeof(pointwise<real_type>));
  hadamard_func_rhs_kernel<real_type><<<pointwise_grid(size), get_pointwise_block()>>>(dev_da, dev_z, dev_dz, size, dev_d_relu_func);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// relu activation on host
template <class real_type>
void relu_activation_cpu(const real_type *in, real_type *out, int size)
{
  apply_pointwise<real_type>(in, out, size, &relu_func);
}

// d_relu activation on host
template <class real_type>
void d_relu_activation_cpu(const real_type *in, real_type *delta, int size)
{
  apply_pointwise<real_type>(in, delta, size, &d_relu_func);
}

// relu activation backprop on host
template <class real_type>
void relu_activation_backprop_cpu(const real_type *da, real_type *z, real_type *dz, int size)
{
  hadamard_func_rhs<real_type>(da, z, dz, size, &d_relu_func);
}

#endif // _RELU_H_
