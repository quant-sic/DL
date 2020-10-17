#ifndef _PW2D_COMP_H_
#define _PW2D_COMP_H_

#include <assert.h>

// on cpu
// dim_add is dimension along which the vector should be added. vector should have the size of the other dimension
template<typename real_type,typename Functor>
void func_along_axis_cpu(Functor functor,const real_type* mat_in,const real_type *vec,real_type* mat_out,const int rows,const int cols,const int dim_add,const int size_vec){

  // assert that dimensions match
  assert(dim_add<2 && (dim_add ? rows : cols)==size_vec);

  // case add along dimenion 0
  if(dim_add==0){
    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
        mat_out[i*cols+j]=functor(mat_in[i*cols+j],vec[j]);
      }
    }

  // case add along dimenion 1
  }else if (dim_add==1){
    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
        mat_out[i*cols+j]=functor(mat_in[i*cols+j],vec[i]);
      }
    }
  }
}


// takes a vector and a matrix and combines them along x axis. Dimensions of course have to match -> see onDev function
// x axis are the columns
template<typename real_type,typename Functor>
__global__ void func_along_axis_x_kernel(Functor functor,const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols){

    // traverse 2D elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < cols); idx += blockDim.x*gridDim.x){
        for (int idy = blockIdx.y * blockDim.y + threadIdx.y; (idy < rows); idy += blockDim.y*gridDim.y){
            // apply elementwise cmbination
            dev_mat_out[idy*cols+idx]=functor(dev_mat_in[idy*cols+idx],dev_vec[idy]);
        }
    }
}

// takes a vector and a matrix and combines them along y axis. Dimensions of course have to match -> see onDev function
// y axis are the rows
template<typename real_type,typename Functor>
__global__ void func_along_axis_y_kernel(Functor functor,const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols){

    // traverse 2D elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < cols); idx += blockDim.x*gridDim.x){
        for (int idy = blockIdx.y * blockDim.y + threadIdx.y; (idy < rows); idy += blockDim.y*gridDim.y){
            dev_mat_out[idy*cols+idx]=functor(dev_mat_in[idy*cols+idx],dev_vec[idx]);
        }
    }
}


// onDev
// dim_add is dimension along which the vector should be added. vector should have the size of the other dimension
template<typename real_type,typename Functor>
void func_along_axis_onDev(Functor functor,const real_type* dev_mat_in,const real_type *dev_vec,real_type* dev_mat_out,const int rows,const int cols,const int dim_add,const int size_vec){

    // assert dimensions match
    assert(dim_add<2 && (dim_add ? rows : cols)==size_vec);

    // case add along dimenion 0
    if(dim_add==0){
        func_along_axis_y_kernel<real_type><<<pointwise2d_grid(cols,rows),get_pointwise2d_block()>>>(functor,dev_mat_in,dev_vec,dev_mat_out,rows,cols);

    // case add along dimenion 1
    }else if (dim_add==1){
        func_along_axis_x_kernel<real_type><<<pointwise2d_grid(cols,rows),get_pointwise2d_block()>>>(functor,dev_mat_in,dev_vec,dev_mat_out,rows,cols);
    }

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
}


#endif // _PW2D_COMP_H_