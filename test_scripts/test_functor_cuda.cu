#include "global.h"
#include "common.h"
#include "test_matrix_operator.h"

template<class real_type>
__device__ __host__ real_type relu_func(real_type x)
{
  return (x > 0 ? x : 0);
}

template<class real_type>
__device__ __host__ real_type d_relu_func(real_type x)
{
  return (x > 0 ? 1 : 0);
}


template<class real_type>
struct relu_functor {
  __device__ __host__ real_type operator()(const real_type x) { return  (x > 0 ? x : 0); }
};


template<class real_type,typename Functor>
__global__ void apply_pointwise_kernel(const real_type *dev_in, real_type *dev_out, int size, Functor dev_functor)
{
  // apply fuction pointwise
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x * gridDim.x)
  {
    dev_out[idx] = dev_functor(dev_in[idx]);
  }
}

template<class real_type,typename Functor>
void apply_pointwise_onDev(const real_type *dev_in, real_type *dev_out, int size,Functor dev_functor)
{

  apply_pointwise_kernel<real_type><<<pointwise_grid(size), get_pointwise_block()>>>(dev_in, dev_out, size, dev_functor);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

template<class real_type>
void relu_activation_onDev(const real_type *dev_in, real_type *dev_out, int size)
{
  apply_pointwise_onDev<real_type>(dev_in, dev_out, size,relu_functor<real_type>());
}


// template<class real_type,template <typename> class function>
// __global__ void apply_pointwise_kernel(const real_type *dev_in, real_type *dev_out, int size)
// {
//   // apply fuction pointwise
//   for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x * gridDim.x)
//   {
//     dev_out[idx] = function<real_type>(dev_in[idx]);
//   }
// }

// template<class real_type,template <typename> class function>
// void apply_pointwise_onDev(const real_type *dev_in, real_type *dev_out, int size)
// {

//   apply_pointwise_kernel<real_type,function><<<pointwise_grid(size), get_pointwise_block()>>>(dev_in, dev_out, size);
//   CHECK(cudaDeviceSynchronize());
//   CHECK(cudaGetLastError());
// }

// template<class real_type>
// void relu_activation_onDev(const real_type *dev_in, real_type *dev_out, int size)
// {
//   apply_pointwise_onDev<real_type,relu_func<real_type>>(dev_in, dev_out, size);
// }



int main(){

  double *act_in,*act_out_relu;
  double *dev_act_in,*dev_act_out_relu;


// setup
  int size=100;

  act_in=(double *)malloc(size*sizeof(double));
  act_out_relu=(double *)malloc(size*sizeof(double));



  create_random_matrix(act_in,size,-1,1);

  CHECK(cudaMalloc((void**)&dev_act_in, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_act_out_relu, size*sizeof(double)));


  copy_host_to_device_double(act_in, dev_act_in, size);


  relu_activation_onDev<double>(dev_act_in,dev_act_out_relu, size);

  copy_device_to_host_double(dev_act_out_relu, act_out_relu, size);

  print_out_matrix(act_in,10,10);
  print_out_matrix(act_out_relu,10,10);
}