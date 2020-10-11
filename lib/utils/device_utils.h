#ifndef _DEVICE_UTILS_H_
#define _DEVICE_UTILS_H_

// applies a pointwise function and stores the result in another array
template <class real_type>
__global__ void apply_pointwise_kernel(const real_type *dev_in, real_type *dev_out, int size, pointwise<real_type> func)
{

  // apply fuction pointwise
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x * gridDim.x)
  {
    dev_out[idx] = func(dev_in[idx]);
  }
}

template <class real_type>
void apply_pointwise_onDev(const real_type *dev_in, real_type *dev_out, int size, pointwise<real_type> dev_func)
{

  apply_pointwise_kernel<real_type><<<pointwise_grid(size), get_pointwise_block()>>>(dev_in, dev_out, size, dev_func);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// Hadamard kernel with an extra function applied to the rhs
// essentially combine pointwise
template <class real_type>
__global__ void hadamard_func_rhs_kernel(const real_type *dev_lhs, real_type *dev_rhs, real_type *dev_res, int size, pointwise<real_type> dev_func)
{

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x * gridDim.x)
    dev_res[idx] = dev_lhs[idx] * dev_func(dev_rhs[idx]);
}

#endif // _DEVICE_UTILS_H_
