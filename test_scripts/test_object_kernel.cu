
#include "global.h"
#include "common.h"
#include "test_matrix_operator.h"
// standard c++ headers
#include <iostream>
#include <memory>

#include "matrix.h"
#include "common_utils.h"
#include "relu.h"


// applies a pointwise function and stores the result in another array
template<typename Functor>
__global__ void apply_pointwise_kernel(matrix mat_in, matrix mat_out, int size,Functor dev_functor)
{
  // apply fuction pointwise
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x * gridDim.x)
  {
    mat_out.data_device.get()[idx] = dev_functor(mat_in.data_device.get()[idx]);
  }
}

template<typename Functor>
void apply_pointwise_onDev(matrix mat_in, matrix mat_out,int size,Functor dev_functor)
{

  apply_pointwise_kernel<<<pointwise_grid(size), get_pointwise_block()>>>(mat_in, mat_out,size, dev_functor);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}








int main(){

    int l=10;
    int n=10;
    matrix mat1(l,n);
    mat1.alloc();
    create_random_matrix(mat1.data_host.get(),l*n,-1,1);
    mat1.copy_host_to_device();

    matrix mat2(l,n);
    mat2.alloc();

    // output_matrix(mat1,true);
    // std::cout << mat1;


    apply_pointwise_onDev(mat1,mat2,l*n,relu_functor<double>());


}