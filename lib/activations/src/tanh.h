#ifndef _TANH_H_
#define _TANH_H_


#include "global.h"
#include "pw_comp.h"

// standard libraries 
#include <math.h>


template<class real_type>
struct tanh_functor {
  __device__ __host__ real_type operator()(const real_type x) { return  tanh(x); }
};

template<class real_type>
void tanh_activation_cpu(const real_type *in, real_type *out, int size){
    apply_pointwise_cpu(in,out,size,tanh_functor<real_type>());
}


template<class real_type>
void tanh_activation_onDev(const real_type *dev_in, real_type *dev_out, int size){
    apply_pointwise_onDev<real_type>(dev_in,dev_out,size,tanh_functor<real_type>());
}


#endif // _TANH_H_
