#ifndef _PW_COMP_H_
#define _PW_COMP_H_

// applies a pointwise function and stores the result in another array
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

// Hadamard kernel with an extra function applied to the rhs
// essentially combine pointwise
template <class real_type,typename Functor>
__global__ void hadamard_func_rhs_kernel(const real_type *dev_lhs, real_type *dev_rhs, real_type *dev_res, int size, Functor dev_functor)
{

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x * gridDim.x)
    dev_res[idx] = dev_lhs[idx] * dev_functor(dev_rhs[idx]);
}

template <class real_type,typename Functor>
void apply_pointwise(const real_type *in, real_type *out, int size, Functor functor)
{
    for (int i = 0; i < size; i++)
        out[i] = functor(in[i]);
}

template <class real_type,typename Functor>
void hadamard_func_rhs(const real_type *lhs, real_type *rhs, real_type *res, int size, Functor functor)
{
    for (int i = 0; i < size; i++)
        res[i] = lhs[i] * functor(rhs[i]);
}

#endif // _PW_COMP_H_