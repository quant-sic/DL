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
#include "global.h"
#include "pw_comp.h"


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
template<class real_type>
struct relu_functor {
  __device__ __host__ real_type operator()(const real_type x) { return  (x > 0 ? x : 0); }
};

template<class real_type>
struct d_relu_functor {
  __device__ __host__ real_type operator()(const real_type x) { return  (x > 0 ? 1 : 0); }
};


template<class real_type>
void relu_activation_onDev(const real_type *dev_in, real_type *dev_out, int size)
{
  apply_pointwise_onDev<real_type>(dev_in, dev_out, size,relu_functor<real_type>());
}

// d_relu activation on device
template <class real_type>
void d_relu_activation_onDev(const real_type *dev_in, real_type *dev_delta, int size)
{
  apply_pointwise_onDev<real_type>(dev_in, dev_delta, size, d_relu_functor<real_type>());
}

// relu activation backprop on device
template <class real_type>
void relu_activation_backprop_onDev(const real_type *dev_da, real_type *dev_z, real_type *dev_dz, int size)
{
  hadamard_func_rhs_kernel<real_type><<<pointwise_grid(size), get_pointwise_block()>>>(dev_da, dev_z, dev_dz, size, d_relu_functor<real_type>());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// relu activation on host
template <class real_type>
void relu_activation_cpu(const real_type *in, real_type *out, int size)
{
  apply_pointwise_cpu<real_type>(in, out, size, relu_functor<real_type>());
}

// d_relu activation on host
template <class real_type>
void d_relu_activation_cpu(const real_type *in, real_type *delta, int size)
{
  apply_pointwise_cpu<real_type>(in, delta, size, d_relu_functor<real_type>());
}

// relu activation backprop on host
template <class real_type>
void relu_activation_backprop_cpu(const real_type *da, real_type *z, real_type *dz, int size)
{
  hadamard_func_rhs<real_type>(da, z, dz, size, d_relu_functor<real_type>());
}

#endif // _RELU_H_
