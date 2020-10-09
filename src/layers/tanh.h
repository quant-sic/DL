#ifndef _TANH_H_
#define _TANH_H_


#include "../global.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/host_utils.h"

// standard libraries 
#include <math.h>


template<class real_type> 
__device__ pointwise<real_type> p_tanh_func = tanh;


template<class real_type>
void tanh_activation_cpu(const real_type *in, real_type *out, int size){
    apply_pointwise(in,out,size,&tanh);
}


template<class real_type>
void tanh_activation_onDev(const real_type *dev_in, real_type *dev_out, int size){
    
    pointwise<real_type> dev_tanh_func;
    cudaMemcpyFromSymbol(&dev_tanh_func, p_tanh_func<real_type>, sizeof(pointwise<real_type>));

    apply_pointwise_onDev<real_type>(dev_in,dev_out,size,dev_tanh_func);
}


#endif // _TANH_H_
