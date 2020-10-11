#include "reduce.h"
#include "global.h"
#include "common.h"

#include <assert.h>

// Row reduction kernel
// kernel assumes 1 block assigned per row, use block-striding methodology
// assumes blocksize is power of 2
template<typename real_type>
__global__ void add_reduce_rows_kernel(const real_type * __restrict__  dev_mat_in, real_type * __restrict__  dev_vec_out, const int rows, const int cols) {

  __shared__ real_type tmp[BS_R_RED_2D];

  tmp[threadIdx.x] = 0;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) // block-stride
    tmp[threadIdx.x] += dev_mat_in[(blockIdx.y * cols) + i];
  __syncthreads();

  for (int i = blockDim.x>>1; i > 0; i>>=1){
    if (threadIdx.x < i) tmp[threadIdx.x] += tmp[threadIdx.x+i];
    __syncthreads();
  }

  if (!threadIdx.x) dev_vec_out[blockIdx.y] = tmp[0];
}
template __global__ void add_reduce_rows_kernel<double>(const double * __restrict__  dev_mat_in, double * __restrict__  dev_vec_out, const int rows, const int cols);
template __global__ void add_reduce_rows_kernel<float>(const float * __restrict__  dev_mat_in, float * __restrict__  dev_vec_out, const int rows, const int cols);



// coll reduction kernel
// kernel assumes one thread assigned per column sum
// launch number of columns threads
template<typename real_type>
__global__ void add_reduce_cols_kernel(const real_type* dev_mat_in,real_type *dev_vec_out, int rows,int cols){

    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if ( idx < cols){
      // add onto intermediate result
      real_type tmp = 0;
      for (int j = 0; j < rows; j++) tmp += dev_mat_in[(j*cols)+idx];

      // save in output vector
      dev_vec_out[idx] = tmp;
    }
}
template __global__ void add_reduce_cols_kernel<double>(const double* dev_mat_in,double *dev_vec_out, int rows,int cols);
template __global__ void add_reduce_cols_kernel<float>(const float* dev_mat_in,float *dev_vec_out, int rows,int cols);



template<typename real_type>
void add_reduce_dim_cpu(const real_type* mat_in,real_type *vec_out, int rows,int cols, int dim_red,int dim_vec){

  assert(dim_red<2 && (dim_red ? rows : cols)==dim_vec);
  memset(vec_out,0,dim_vec*sizeof(real_type));

  if(dim_red==0){
    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
	      vec_out[j]+=mat_in[i*cols+j];
      }
    }
  } else if (dim_red==1) {

    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
	      vec_out[i]+=mat_in[i*cols+j];
      }
    }
  }
}
template void add_reduce_dim_cpu<float>(const float* mat_in,float *vec_out, int rows,int cols, int dim_red,int dim_vec);
template void add_reduce_dim_cpu<double>(const double* mat_in,double *vec_out, int rows,int cols, int dim_red,int dim_vec);



// function onDev
// dim_red is the dimension which is supposed to be reduced -> the other dimension will remain and should be equal to the size of the given vector
template<typename real_type>
void add_reduce_dim_onDev(const real_type* dev_mat_in,real_type *dev_vec_out, int rows,int cols, int dim_red,int size_vec){

    assert(dim_red<2 && (dim_red ? rows : cols)==size_vec);
    CHECK(cudaMemset(dev_vec_out, 0, size_vec*sizeof(real_type)));
    if(dim_red==0){
      add_reduce_cols_kernel<real_type><<<col_red_grid(cols,rows),get_col_red_block()>>>(dev_mat_in,dev_vec_out,rows,cols);
    }else if (dim_red==1){
      add_reduce_rows_kernel<real_type><<<row_red_grid(cols,rows),get_row_red_2d_block()>>>(dev_mat_in, dev_vec_out,rows,cols);
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

}
template void add_reduce_dim_onDev<float>(const float* dev_mat_in,float *dev_vec_out, int rows,int cols, int dim_red,int size_vec);
template void add_reduce_dim_onDev<double>(const double* dev_mat_in,double *dev_vec_out, int rows,int cols, int dim_red,int size_vec);


template <class real_type>
real_type get_max(const real_type *data, int length)
{
    real_type max = 0;
    for (int i = 0; i < length; i++)
        max = (data[i] > max ? data[i] : max);
    return max;
}
template float get_max(const float *data,int length);
template double get_max(const double *data,int length);




// on dev reduction setup
// kernel that gets the maximum value for every row
template<typename real_type>
__global__ void get_max_kernel(const real_type * __restrict__ data, int length,real_type *dev_res){

    __shared__ real_type tmp[BS_R_RED_1D];

    tmp[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < length; i += blockDim.x) // block-stride
      tmp[threadIdx.x] = (data[i]>tmp[threadIdx.x] ? data[i]:tmp[threadIdx.x]);
    __syncthreads();

    for (int i = blockDim.x>>1; i > 0; i>>=1){
      if (threadIdx.x < i) tmp[threadIdx.x] = (tmp[threadIdx.x+i]>tmp[threadIdx.x]?tmp[threadIdx.x+i]:tmp[threadIdx.x]);
      __syncthreads();
    }

    if (!threadIdx.x) *dev_res = tmp[0];
}
template __global__ void get_max_kernel<double>(const double * __restrict__ data, int length,double *dev_res);
template __global__ void get_max_kernel<float>(const float * __restrict__ data, int length,float *dev_res);





template<typename real_type>
real_type get_max_onDev(const real_type *dev_in,int size){

  real_type *dev_res;
  real_type res[1];
  CHECK(cudaMalloc((void**)&dev_res, sizeof(real_type)));
  get_max_kernel<<<row_red_grid(size,1),get_row_red_1d_block()>>>(dev_in,size, dev_res);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(&res[0],dev_res , sizeof(real_type), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(dev_res));
  return res[0];
}
template float get_max_onDev(const float *dev_in,int size);
template double get_max_onDev(const double *dev_in,int size);
