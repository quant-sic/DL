#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include <assert.h>
#include "t_contract_d.h"
#include "../common.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define BLOCKSIZE 32

void matMul(double *A,double *B,int M,int N, int K,double *C)
{

  int i,j,kk;
  double interm_sum;


  for (i=0;i<M;i++){

      for (j=0;j<N;j++){

          interm_sum=0;

          for (kk=0;kk<K;kk++){
              interm_sum+=A[i*K+kk]*B[kk*N+j];

          }
          C[i*N+j]=interm_sum;
      }
  }
}

__global__ void mat_transpose_kernel(const double *mat_in, double *mat_out, int rows, int cols)
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

void mat_transpose_gpu(const double* mat_in, double* mat_out, int rows, int cols){

    double *d_mat_in,*d_mat_out;
    CHECK(cudaMalloc((void**)&d_mat_in,rows*cols*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_mat_out,rows*cols*sizeof(double)));
    CHECK(cudaMemcpy(d_mat_in, mat_in, rows*cols*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block (BLOCKSIZE,BLOCKSIZE);
    dim3 grid ((cols+block.x-1)/block.x,(rows+block.y-1)/block.y);
    // printf("M %d, N %d   grid %d %d\n",rows,cols,grid.y,grid.x );

    // Invoke kernel
    mat_transpose_kernel<<<grid, block>>>(d_mat_in,d_mat_out,rows,cols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read C from device memory
    cudaMemcpy(mat_out, d_mat_out, rows*cols*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_mat_in);
    cudaFree(d_mat_out);
}


// Get a matrix element
__device__ double GetElement(const double *A,int stride, int row, int col)
{
    return A[row * stride + col];
}

// Set a matrix element
__device__ void SetElement(double *A, int row, int col,int stride,
                           double value)
{
    A[row * stride + col] = value;
}


__global__ void matMul_kernel1(double *d_A,double *d_B,int M, int N,int K,double *d_C){

  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

  if(i<M && j<N){

    double sum=0;

    for(int ki=0; ki < K;ki++){

      sum+=d_A[i*K+ki]*d_B[ki*N+j];

    }

    d_C[i*N+j]=sum;
  }
}

void matMul_gpu1(const double *A, const double *B, int M,int N,int K,double *C)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block (BLOCKSIZE,BLOCKSIZE);
    dim3 grid ((N+block.x-1)/block.x,(M+block.y-1)/block.y);

    // printf("M %d, N %d   grid %d %d\n",M,N,grid.y,grid.x );
    // Invoke kernel
    matMul_kernel1<<<grid, block>>>(d_A, d_B,M,N,K,d_C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matMul_kernel2(const double *d_A_T,const double *d_B,int M, int N,int K,double *d_C){

  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int i = blockIdx.y*blockDim.y+threadIdx.y;

  if(i<M && j<N){

    double sum=0;

    for(int ki=0; ki < K;ki++){

      sum+=d_A_T[ki*M+i]*d_B[ki*N+j];

    }

    d_C[i*N+j]=sum;
  }
}


void matMul_gpu2(const double *A, const double *B, int M,int N,int K,double *C)
{

    double *d_A,*d_B,*d_A_T,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_A_T,M*K*sizeof(double)));


    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block (BLOCKSIZE,BLOCKSIZE);
    dim3 grid ((N+block.x-1)/block.x,(M+block.y-1)/block.y);
    dim3 grid_A_T ((K+block.x-1)/block.x,(M+block.y-1)/block.y);

    // Invoke kernel
    mat_transpose_kernel<<<grid_A_T, block>>>(d_A, d_A_T, M, K);
    CHECK(cudaDeviceSynchronize());
    matMul_kernel2<<<grid, block>>>(d_A_T, d_B,M,N,K,d_C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



__global__ void matMul_kernel_sm(double *d_A,double *d_B,int M, int N,int K,double *d_C){

  double CValue = 0;

  int row = blockIdx.y*BLOCKSIZE + threadIdx.y;
  int col = blockIdx.x*BLOCKSIZE + threadIdx.x;

  __shared__ double As[BLOCKSIZE][BLOCKSIZE];
  __shared__ double Bs[BLOCKSIZE][BLOCKSIZE];

  for (int kk = 0; kk < (BLOCKSIZE + K - 1)/BLOCKSIZE; kk++) {

       if (kk*BLOCKSIZE + threadIdx.x < K && row < M)
           As[threadIdx.y][threadIdx.x] = d_A[row*K + kk*BLOCKSIZE + threadIdx.x];
       else
           As[threadIdx.y][threadIdx.x] = 0.0;

       if (kk*BLOCKSIZE + threadIdx.y < K && col < N)
           Bs[threadIdx.y][threadIdx.x] = d_B[(kk*BLOCKSIZE + threadIdx.y)*N + col];
       else
           Bs[threadIdx.y][threadIdx.x] = 0.0;

       __syncthreads();

       for (int n = 0; n < BLOCKSIZE; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

       __syncthreads();
  }

  if (row < M && col < N) d_C[(row*N) +col] = CValue;

}


void matMul_gpu_sm(const double *A, const double *B, int M,int N,int K,double *C)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block (BLOCKSIZE,BLOCKSIZE);
    dim3 grid ((N+block.x-1)/block.x,(M+block.y-1)/block.y);

    // Invoke kernel
    matMul_kernel_sm<<<grid, block>>>(d_A, d_B,M,N,K,d_C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}





__global__ void matMul_kernel_sm_coa(double *d_A,double *d_B_T,int M, int N,int K,double *d_C){

  double CValue = 0;

  int row = blockIdx.y*BLOCKSIZE + threadIdx.y;
  int col = blockIdx.x*BLOCKSIZE + threadIdx.x;

  __shared__ double As[BLOCKSIZE][BLOCKSIZE];
  __shared__ double Bs[BLOCKSIZE][BLOCKSIZE];

  for (int kk = 0; kk < (BLOCKSIZE + K - 1)/BLOCKSIZE; kk++) {

       if (kk*BLOCKSIZE + threadIdx.x < K && row < M)
           As[threadIdx.y][threadIdx.x] = d_A[row*K + kk*BLOCKSIZE + threadIdx.x];
       else
           As[threadIdx.y][threadIdx.x] = 0.0;

       if (kk*BLOCKSIZE + threadIdx.y < K && col < N)
           Bs[threadIdx.y][threadIdx.x] = d_B_T[col*K + (kk*BLOCKSIZE + threadIdx.y)];
       else
           Bs[threadIdx.y][threadIdx.x] = 0.0;

       __syncthreads();

       for (int n = 0; n < BLOCKSIZE; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

       __syncthreads();
  }

  if (row < M && col < N) d_C[(row*N) +col] = CValue;

}


void matMul_gpu_sm_coa(const double *A, const double *B, int M,int N,int K,double *C)
{

    double *d_A,*d_B_T,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B_T,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block (BLOCKSIZE,BLOCKSIZE);
    dim3 grid ((N+block.x-1)/block.x,(M+block.y-1)/block.y);
    dim3 grid_B_T ((N+block.x-1)/block.x,(K+block.y-1)/block.y);

    // printf("M %d, N %d, K %d   grid %d %d\n",M,N,K,grid.y,grid.x );

    // Invoke kernel
    mat_transpose_kernel<<<grid_B_T, block>>>(d_B, d_B_T, K, N);
    CHECK(cudaDeviceSynchronize());
    matMul_kernel_sm_coa<<<grid, block>>>(d_A, d_B_T,M,N,K,d_C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_B_T);

}


void matMul_cublas(const double *A, const double *B, int M,int N,int K,double *C)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));


    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
    }

    const double alpha=1.0;
    const double beta=0.0;


    // Invoke kernel
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,&alpha,(const double *)d_B, N,(const double *)d_A, K,&beta,(double *)d_C, N);
    CHECK(cudaDeviceSynchronize());

    cublasDestroy(handle);

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}




__global__ void matMul_kernel_dsm(double *d_A,double *d_B,int M, int N,int K,double *d_C){

  double CValue = 0;

  // should be the same
  assert(blockDim.x==blockDim.y);
  const int block_dim=blockDim.x;

  int row = blockIdx.y*block_dim + threadIdx.y;
  int col = blockIdx.x*block_dim + threadIdx.x;

  extern __shared__ double Sm[];

  double *As=Sm;
  double *Bs=(double *)&Sm[block_dim*block_dim];


  for (int kk = 0; kk < (block_dim + K - 1)/block_dim; kk++) {

       if (kk*block_dim + threadIdx.x < K && row < M)
           As[threadIdx.y*block_dim+threadIdx.x] = d_A[row*K + kk*block_dim + threadIdx.x];
       else
           As[threadIdx.y*block_dim+threadIdx.x] = 0.0;

       if (kk*block_dim + threadIdx.y < K && col < N)
           Bs[threadIdx.y*block_dim+threadIdx.x] = d_B[(kk*block_dim + threadIdx.y)*N + col];
       else
           Bs[threadIdx.y*block_dim+threadIdx.x] = 0.0;

       __syncthreads();

       for (int n = 0; n < block_dim; ++n) CValue += As[threadIdx.y*block_dim+n] * Bs[n*block_dim+threadIdx.x];

       __syncthreads();
  }

  if (row < M && col < N) d_C[(row*N) +col] = CValue;

}


void matMul_gpu_dsm(const double *A, const double *B, int M,int N,int K,double *C,int threads_block)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    int block_dim=(int)sqrt(threads_block);
    threads_block=block_dim*block_dim;

    dim3 block (block_dim,block_dim);
    dim3 grid ((N+block.x-1)/block.x,(M+block.y-1)/block.y);

    // Invoke kernel
    matMul_kernel_dsm<<<grid, block,2*threads_block*sizeof(double)>>>(d_A, d_B,M,N,K,d_C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
