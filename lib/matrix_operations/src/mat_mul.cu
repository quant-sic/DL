#include "global.h"
#include "common.h"
#include "common_utils.h"
#include <assert.h>

// computes the matrixproduct of double matrices with arbitrary size on host
template<typename real_type>
void mat_mul_cpu(const real_type *A,const real_type *B,int M,int N, int K,real_type *C)
{

  real_type interm_sum;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      interm_sum = 0.;
      for (int kk = 0; kk < K; kk++)
	interm_sum += A[i*K+kk]*B[kk*N+j];
      C[i*N+j] = interm_sum;
    }
  }
}
template void mat_mul_cpu<double>(const double *A,const double *B,int M,int N, int K,double *C);
template void mat_mul_cpu<float>(const float *A,const float *B,int M,int N, int K,float *C);


// naive Matrix Multiplication kernel with coalesced global memory access.
// A needs to be given transposed
template<typename real_type>
__global__ void mat_mul_coa_kernel(const real_type *d_A_T,const real_type *d_B,int M, int N,int K,real_type *d_C){

  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int i = blockIdx.y*blockDim.y+threadIdx.y;

  if(i<M && j<N){

    real_type sum=0;

    for(int ki=0; ki < K;ki++){

      sum+=d_A_T[ki*M+i]*d_B[ki*N+j];

    }

    d_C[i*N+j]=sum;
  }
}
template __global__ void mat_mul_coa_kernel<double>(const double *d_A_T,const double *d_B,int M, int N,int K,double *d_C);
template __global__ void mat_mul_coa_kernel<float>(const float *d_A_T,const float *d_B,int M, int N,int K,float *d_C);


// computes the matrix product of double matrices with arbitrary size on device
// naive implementation with transposed matrix A -> coalesced memory access
template<typename real_type>
void mat_mul_coa_onDev(const real_type *d_A, const real_type *d_B, int M,int N,int K,real_type *d_C)
{

		real_type *d_A_T;
		CHECK(cudaMalloc((void**)&d_A_T,M*K*sizeof(real_type)));

        dim3 block = get_matrix_mul_block();
		dim3 grid = matrix_mul_grid(N,M);
		dim3 grid_A_T ((K+block.x-1)/block.x,(M+block.y-1)/block.y);

		// Invoke kernel
		mat_transpose_kernel<<<grid_A_T, block>>>(d_A, d_A_T, M, K);
		CHECK(cudaDeviceSynchronize());
		mat_mul_coa_kernel<real_type><<<grid, block>>>(d_A_T, d_B,M,N,K,d_C);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
}
template void mat_mul_coa_onDev<double>(const double *d_A, const double *d_B, int M,int N,int K,double *d_C);
template void mat_mul_coa_onDev<float>(const float *d_A, const float *d_B, int M,int N,int K,float *d_C);


// matmul_kernel_sm_tr ohne indexing Funktionen
template <typename real_type>
__global__ void mat_mul_tr_kernel(const real_type *__restrict__ d_A,const real_type * __restrict__ d_B,const int A_TRANSP,const int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,real_type *d_C){

  real_type CValue = 0;

  int row = blockIdx.y*BS_2D + threadIdx.y;
  int col = blockIdx.x*BS_2D + threadIdx.x;

  __shared__ real_type As[BS_2D][BS_2D];
  __shared__ real_type Bs[BS_2D][BS_2D];

  if (A_TRANSP && B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[(kk*BS_2D + threadIdx.x)*cols_A+row];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[col*cols_B+(kk*BS_2D + threadIdx.y)];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

  }else if(A_TRANSP && !B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[(kk*BS_2D + threadIdx.x)*cols_A+row];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[(kk*BS_2D + threadIdx.y)*cols_B+col];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

  }else if(!A_TRANSP && !B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[row*cols_A+(kk*BS_2D + threadIdx.x)];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[(kk*BS_2D + threadIdx.y)*cols_B+col];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

  }else if(!A_TRANSP && B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[row*cols_A+(kk*BS_2D + threadIdx.x)];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[col*cols_B+(kk*BS_2D + threadIdx.y)];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;
  }

}
template __global__ void mat_mul_tr_kernel<double>(const double *__restrict__ d_A,const double * __restrict__ d_B,const int A_TRANSP,const int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,double *d_C);
template __global__ void mat_mul_tr_kernel<float>(const float *__restrict__ d_A,const float * __restrict__ d_B,const int A_TRANSP,const int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,float *d_C);


// computes the matrix product of double matrices with arbitrary size on device
// tiled implementation with static shared memory and transposed matrices using if for 4 kernel versions
template<typename real_type>
void mat_mul_tr_onDev(const real_type *d_A, const real_type *d_B,const int A_TRANSP,const int B_TRANSP,const int rows_op_A,const int cols_op_A,const int rows_op_B,const int cols_op_B,real_type *d_C)
{
    // assert matrizes do match
    assert(cols_op_A==rows_op_B);

    // get matrix dimensions
    int rows_A,cols_A,rows_B,cols_B;
    if(A_TRANSP){
      rows_A=cols_op_A;
      cols_A=rows_op_A;
    }else{
      rows_A=rows_op_A;
      cols_A=cols_op_A;
    }

    if(B_TRANSP){
      rows_B=cols_op_B;
      cols_B=rows_op_B;
    }else{
      rows_B=rows_op_B;
      cols_B=cols_op_B;
    }

    // Invoke kernel
    mat_mul_tr_kernel<real_type><<<matrix_mul_grid(cols_op_B,rows_op_A), get_matrix_mul_block()>>>((const real_type *)d_A, (const real_type *)d_B,A_TRANSP,B_TRANSP,rows_op_A,cols_op_B,cols_op_A,rows_A,cols_A,rows_B,cols_B,d_C);


    // error handling
    if(cudaDeviceSynchronize()||cudaGetLastError()){
      printf("Error in matMul_sm_tr_onDev\n");
      printf("Matrix Dimensions: M %d,N %d,K %d\n",rows_op_A,cols_op_B,cols_op_A);

      CHECK(cudaDeviceSynchronize());
      CHECK(cudaGetLastError());
    }
}
template void mat_mul_tr_onDev<double>(const double *d_A, const double *d_B,const int A_TRANSP,const int B_TRANSP,const int rows_op_A,const int cols_op_A,const int rows_op_B,const int cols_op_B,double *d_C);
template void mat_mul_tr_onDev<float>(const float *d_A, const float *d_B,const int A_TRANSP,const int B_TRANSP,const int rows_op_A,const int cols_op_A,const int rows_op_B,const int cols_op_B,float *d_C);
