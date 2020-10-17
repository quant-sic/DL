#ifndef _COMMON_UTILS_H_
#define _COMMON_UTILS_H_

#include "global.h"

template<typename real_type>
struct exp_sub_val_functor {
  exp_sub_val_functor(real_type val) : _val(val) {}  // Constructor

  __device__ __host__ real_type operator()(const real_type x) { return  exp( x-_val); }

  private:
    real_type _val;
};

template<typename real_type>
struct scale_functor {
  scale_functor(real_type val) : _val(val) {}  // Constructor
  __device__ __host__ real_type operator()(const real_type x) { return  _val*x; }

  private:
    real_type _val;
};

template<typename real_type>
struct mul_add_functor {
  mul_add_functor(real_type val) : _val(val) {}  // Constructor
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x+ _val*y; }

  private:
    real_type _val;
};

template<typename real_type>
struct add_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x+y; }
};

template<typename real_type>
struct sub_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x-y; }
};

template<typename real_type>
struct mul_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x*y; }
};

template<typename real_type>
struct div_functor {
  __device__ __host__ real_type operator()(const real_type x,const real_type y) { return  x/y; }
};




// matrix_transpose_cpu and onDev
// computes the transposed matrix of a double matrix with arbitrary size on cpu
template<typename real_type>
void mat_transpose_cpu(real_type* out,
			  real_type* inp,
			  int     rows,
			  int     cols)
{
  for (int row = 0; row < rows; row++)
    for (int col = 0; col < cols; col++)
      out[col*rows+row] = inp[row*cols+col];
}


// kernel that transposes a matrix
template<typename real_type>
__global__ void mat_transpose_kernel(const real_type *mat_in, real_type *mat_out, int rows, int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}


// computes the transposed matrix of a double matrix with arbitrary size on device
template<typename real_type>
void mat_transpose_onDev(const real_type* dev_mat_in, real_type* dev_mat_out, int rows, int cols){

	// Invoke kernel
	mat_transpose_kernel<real_type><<<matrix_mul_grid(cols,rows), get_matrix_mul_block()>>>(dev_mat_in,dev_mat_out,rows,cols);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
}





#endif // _COMMON_UTILS_H_