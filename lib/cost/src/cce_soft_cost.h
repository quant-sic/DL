#ifndef _CCE_SOFT_COST_H_
#define _CCE_SOFT_COST_H_

#include "softmax.h"
#include "costs.h"
#include "cce_cost.h"

//_______________________________________________________________________________________________
// categorical cross entropy cost
class cce_soft_cost : public costs
{
public :
  // constructor
  cce_soft_cost (void) {this->_type = CCE_SOFT;};
  ~cce_soft_cost(void) {};

  // functions to calculate
  double cost (matrix predict, matrix target, bool flag_host);
  matrix dcost(matrix predict, matrix target, matrix dy, bool flag_host);

  // operator overloading
  //friend std::ostream& operator <<(std::ostream& out, const cce_soft_cost& cce_soft);
  void print_out(std::ostream& out) const { out << "cce_soft"; };
};

// _______________________________________________________________________________________________
// cce_soft_cost implementation

template<typename _real_type>
_real_type cce_soft_cpu(const _real_type *in,_real_type *target,int size,int batchsize){
  _real_type *softmax=(_real_type *)malloc(size*sizeof(_real_type));
  softmax_activation_cpu<_real_type>(in,softmax,batchsize,size/batchsize);
  _real_type cce_value=cce<_real_type>(softmax,target,size);
  free(softmax);
  return cce_value;
}

template<typename _real_type>
void d_cce_soft_cpu(const _real_type *in, _real_type *target, _real_type *delta, int size,int batchsize){
  _real_type *softmax=(_real_type *)malloc(size*sizeof(_real_type));
  softmax_activation_cpu<_real_type>(in,softmax,batchsize,size/batchsize);
  combine_pointwise_cpu<_real_type>(softmax,target,delta,size,sub_functor<double>());
  free(softmax);
}

template<typename _real_type>
_real_type cce_soft_onDev(const _real_type *dev_in,_real_type *dev_target,int size,int batchsize){
  _real_type *dev_softmax;
  CHECK(cudaMalloc((void**)&dev_softmax, size*sizeof(_real_type)));
  softmax_activation_onDev<double>(dev_in,dev_softmax,batchsize,size/batchsize);
  _real_type cce_value=cce_onDev<double>(dev_softmax,dev_target,size);
  cudaFree(dev_softmax);
  return cce_value;
}

template<typename _real_type>
void d_cce_soft_onDev(const _real_type *dev_in, _real_type *dev_target, _real_type *dev_delta, int size,int batchsize){
  _real_type *dev_softmax;
  CHECK(cudaMalloc((void**)&dev_softmax, size*sizeof(_real_type)));
  softmax_activation_onDev<double>(dev_in,dev_softmax,batchsize,size/batchsize);
  combine_pointwise_onDev<double>(dev_softmax,dev_target,dev_delta,size,sub_functor<double>());
  cudaFree(dev_softmax);
}



#endif // _CCE_SOFT_COST_H_


