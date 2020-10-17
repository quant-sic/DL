#ifndef _REDUCE_H_
#define _REDUCE_H_


#include "global.h"
#include "common.h"



template<typename real_type>
extern void add_reduce_dim_cpu(const real_type* mat_in,real_type *vec_out, int rows,int cols, int dim_red,int dim_vec);

template<typename real_type>
extern __global__ void add_reduce_rows_kernel(const real_type * __restrict__  dev_mat_in, real_type * __restrict__  dev_vec_out, const int rows, const int cols);

template<typename real_type>
extern __global__ void add_reduce_cols_kernel(const real_type* dev_mat_in,real_type *dev_vec_out, int rows,int cols);

template<typename real_type>
extern void add_reduce_dim_onDev(const real_type* dev_mat_in,real_type *dev_vec_out, int rows,int cols, int dim_red,int size_vec);

template <class real_type>
extern real_type get_max(const real_type *data, int length);

template<typename real_type>
extern real_type get_max_onDev(const real_type *dev_in,int size);

template<typename real_type,typename Functor>
real_type sum_func_array(const real_type *in, const real_type* target, int size,Functor functor){
  real_type res = 0.;
  // add terms onto res
  for (int k = 0; k < size; k++) res += functor(in[k], target[k]);
  return res;
}


// Row reduction kernel
// kernel assumes 1 block assigned per row, use block-striding methodology
// assumes blocksize is power of 2
// Also there is an elementwise combination included
template<typename real_type,typename Functor>
__global__ void add_reduce_rows_func_kernel(const real_type * __restrict__  dev_in, const real_type * __restrict__  dev_target, real_type* dev_res, const int size,Functor dev_functor) {

  // set up shared memory
  __shared__ real_type tmp[BS_R_RED_1D];

  tmp[threadIdx.x] = 0;

  // apply function and sum into shared memory
  for (int i = threadIdx.x; i < size; i += blockDim.x) // block-stride
    tmp[threadIdx.x] += dev_functor(dev_in[i],dev_target[i]);
  __syncthreads();

  // reduce in place in shared memory
  for (int i = blockDim.x>>1; i > 0; i>>=1){
    if (threadIdx.x < i) tmp[threadIdx.x] += tmp[threadIdx.x+i];
    __syncthreads();
  }
  // assign result
  if (!threadIdx.x) *dev_res = tmp[0];
}

// on dev reduction setup
template<typename real_type,typename Functor>
real_type sum_func_array_onDev(const real_type *dev_in,const real_type* dev_target,int size,Functor dev_functor){

  real_type *dev_res;
  real_type res[1];
  CHECK(cudaMalloc((void**)&dev_res, sizeof(real_type)));
  add_reduce_rows_func_kernel<real_type><<<row_red_grid(size,1),get_row_red_1d_block()>>>(dev_in, dev_target,dev_res,size, dev_functor);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(&res[0],dev_res , sizeof(real_type), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dev_res));
  return res[0];
}



#endif // _REDUCE_H