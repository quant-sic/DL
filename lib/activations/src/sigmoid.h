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

#include "pw_comp.h"

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
template<class real_type>
struct sigmoid_functor {
  __device__ __host__ real_type operator()(const real_type x) { return 1/(1+exp(-x)); }
};

template<class real_type>
struct d_sigmoid_functor {
  __device__ __host__ real_type operator()(const real_type x) { return exp(-x)/((1+exp(-x))*(1+exp(-x))); }
};

template <class real_type>
void sigmoid_activation_onDev(const real_type *dev_in, real_type *dev_out, int size){
  apply_pointwise_onDev<real_type>(dev_in, dev_out, size, sigmoid_functor<real_type>());
}

template <class real_type>
void d_sigmoid_activation_onDev(const real_type *dev_in, real_type *dev_delta, int size){
  apply_pointwise_onDev<real_type>(dev_in, dev_delta, size, d_sigmoid_functor<real_type>());
}

template <class real_type>
void sigmoid_activation_backprop_onDev(const real_type *dev_da,real_type *dev_z,real_type *dev_dz,int size){

  hadamard_func_rhs_kernel<real_type><<<pointwise_grid(size),get_pointwise_block()>>>(dev_da, dev_z, dev_dz, size,d_sigmoid_functor<real_type>());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

template <class real_type>
void sigmoid_activation_cpu(const real_type *in,real_type *out,int size){
  apply_pointwise_cpu<real_type>(in, out, size, sigmoid_functor<real_type>());
}

template <class real_type>
void d_sigmoid_activation_cpu(const real_type *in,real_type *delta,int size){
  apply_pointwise_cpu<real_type>(in, delta, size, d_sigmoid_functor<real_type>());
}

template <class real_type>
void sigmoid_activation_backprop_cpu(const real_type *da,real_type *z,real_type *dz,int size){
  hadamard_func_rhs<real_type>(da, z, dz, size, d_sigmoid_functor<real_type>());
}


#endif // _SIGMOID_H_
