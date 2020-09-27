/*
  SOURCE OPERATOR DEVICE MATRIX OPERATIONS GPU
  BASICALLY ONLY USED FOR Testing
  CONTAINS COPY OVERHAD

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 11.08.2020
  TO-DO   :
  CAUTION :
*/

// c++ standard headers
#include <iostream>

// standard c headers
#include <assert.h>

// own headers
#include "../common.h"
#include "matrix_operator.h"
#include "matrix_operator_gpu.h"
#include "kernel_utils.h"

#include "../global.h"

// cublas headers
#include "cublas_v2.h"
#include <cuda_runtime.h>



void add_reduce_dim_gpu(const double* mat_in,double *vec_out, int rows,int cols, int dim_red,int size_vec){

  double *dev_mat_in,*dev_vec_out;
  CHECK(cudaMalloc((void**)&dev_mat_in, rows*cols*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_vec_out, size_vec*sizeof(double)));

  CHECK(cudaMemcpy(dev_mat_in, mat_in, rows*cols*sizeof(double), cudaMemcpyHostToDevice));

  add_reduce_dim_onDev(dev_mat_in,dev_vec_out, rows,cols, dim_red,size_vec);

  CHECK(cudaMemcpy(vec_out, dev_vec_out, size_vec*sizeof(double), cudaMemcpyDeviceToHost));

  // free cuda storage
  CHECK(cudaFree(dev_mat_in));
  CHECK(cudaFree(dev_vec_out));

}


void add_along_axis_gpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_add, int size_vec){

  double *dev_mat_in,*dev_mat_out,*dev_vec;
  CHECK(cudaMalloc((void**)&dev_mat_in, rows*cols*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_mat_out, rows*cols*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_vec, size_vec*sizeof(double)));

  CHECK(cudaMemcpy(dev_mat_in, mat_in, rows*cols*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_vec, vec, size_vec*sizeof(double), cudaMemcpyHostToDevice));

  add_along_axis_onDev(dev_mat_in,dev_vec,dev_mat_out,rows,cols,dim_add,size_vec);

  CHECK(cudaMemcpy(mat_out, dev_mat_out, rows*cols*sizeof(double), cudaMemcpyDeviceToHost));

  // free cuda storage
  CHECK(cudaFree(dev_mat_in));
  CHECK(cudaFree(dev_mat_out));
  CHECK(cudaFree(dev_vec));
}


void div_along_axis_gpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_div, int size_vec){

  double *dev_mat_in,*dev_mat_out,*dev_vec;
  CHECK(cudaMalloc((void**)&dev_mat_in, rows*cols*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_mat_out, rows*cols*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_vec, size_vec*sizeof(double)));

  CHECK(cudaMemcpy(dev_mat_in, mat_in, rows*cols*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_vec, vec, size_vec*sizeof(double), cudaMemcpyHostToDevice));

  div_along_axis_onDev(dev_mat_in,dev_vec,dev_mat_out,rows,cols,dim_div,size_vec);

  CHECK(cudaMemcpy(mat_out, dev_mat_out, rows*cols*sizeof(double), cudaMemcpyDeviceToHost));

  // free cuda storage
  CHECK(cudaFree(dev_mat_in));
  CHECK(cudaFree(dev_mat_out));
  CHECK(cudaFree(dev_vec));
}

//___________________________________________________________________________________________________
// matrix ard_gpu
void matrix_hadamard_gpu(double* res,
			 const double* lhs,
			 const double* rhs,
			 int     size,
			 int     threads_block)
{
  // alloc cuda storage
  double* d_res;
  double* d_lhs;
  double* d_rhs;
  CHECK(cudaMalloc((void**)&d_res, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&d_lhs, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&d_rhs, size*sizeof(double)));

  // moving matrices to device
  CHECK(cudaMemcpy(d_lhs, lhs, size*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_rhs, rhs, size*sizeof(double), cudaMemcpyHostToDevice));

  // calling ard onDev
  matrix_hadamard_onDev(d_res, d_lhs, d_rhs,    size,    threads_block);


  // moving matrices back from memory
  CHECK(cudaMemcpy(res, d_res, size*sizeof(double), cudaMemcpyDeviceToHost));

  // free cuda storage
  CHECK(cudaFree(d_res));
  CHECK(cudaFree(d_lhs));
  CHECK(cudaFree(d_rhs));
}

//___________________________________________________________________________________________________
// matrix hadamard_gpu
void matrix_add_gpu(double* res,
		    const double* lhs,
		    const double* rhs,
		    int     size,
		    int     threads_block)
{
  // alloc cuda storage
  double* d_res;
  double* d_lhs;
  double* d_rhs;
  CHECK(cudaMalloc((void**)&d_res, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&d_lhs, size*sizeof(double)));
  CHECK(cudaMalloc((void**)&d_rhs, size*sizeof(double)));

  // moving matrices to device
  CHECK(cudaMemcpy(d_lhs, lhs, size*sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_rhs, rhs, size*sizeof(double), cudaMemcpyHostToDevice));

  // calling add onDev
  matrix_add_onDev( d_res ,  d_lhs, d_rhs, size, threads_block);

  // moving matrices back from memory
  CHECK(cudaMemcpy(res, d_res, size*sizeof(double), cudaMemcpyDeviceToHost));

  // free cuda storage
  CHECK(cudaFree(d_res));
  CHECK(cudaFree(d_lhs));
  CHECK(cudaFree(d_rhs));
}


void mulAdd_gpu(double* res, const double* lhs, const double* rhs, const double factor, int size,int threads_block)
{
	// alloc cuda storage
	double *dev_res,*dev_lhs,*dev_rhs;
	CHECK(cudaMalloc((void**)&dev_res, size*sizeof(double)));
	CHECK(cudaMalloc((void**)&dev_lhs, size*sizeof(double)));
	CHECK(cudaMalloc((void**)&dev_rhs, size*sizeof(double)));

	// moving matrices to device
	CHECK(cudaMemcpy(dev_lhs, lhs, size*sizeof(double), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_rhs, rhs, size*sizeof(double), cudaMemcpyHostToDevice));

	mulAdd_onDev(dev_res,dev_lhs,dev_rhs,factor,size,threads_block);

	// moving matrices back from memory
	CHECK(cudaMemcpy(res, dev_res, size*sizeof(double), cudaMemcpyDeviceToHost));

	// free cuda storage
	CHECK(cudaFree(dev_res));
  CHECK(cudaFree(dev_lhs));
  CHECK(cudaFree(dev_rhs));
}

void mat_transpose_gpu(const double* mat_in, double* mat_out, int rows, int cols, int threads_block){

    double *d_mat_in,*d_mat_out;
    CHECK(cudaMalloc((void**)&d_mat_in,rows*cols*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_mat_out,rows*cols*sizeof(double)));
    CHECK(cudaMemcpy(d_mat_in, mat_in, rows*cols*sizeof(double), cudaMemcpyHostToDevice));

    mat_transpose_onDev(d_mat_in, d_mat_out, rows, cols, threads_block);

    // Read C from device memory
    cudaMemcpy(mat_out, d_mat_out, rows*cols*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_mat_in));
    CHECK(cudaFree(d_mat_out));
}


void matMul_gpu1(const double *A, const double *B, int M,int N,int K,double *C, int threads_block)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    matMul_onDev1(d_A,d_B, M,N,K,d_C, threads_block);

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_B));
}


void matMul_gpu2(const double *A, const double *B, int M,int N,int K,double *C,int threads_block)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));


    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    matMul_onDev2(d_A, d_B,  M, N, K,d_C, threads_block);


    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
}


void matMul_gpu_dsm(const double *A, const double *B, int M,int N,int K,double *C,int threads_block)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    matMul_dsm_onDev(d_A,d_B,  M, N, K,d_C,threads_block);

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}

void matMul_gpu_dsm_coa(const double *A, const double *B, int M,int N,int K,double *C,int threads_block)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));

    matMul_dsm_coa_onDev(d_A, d_B,  M, N, K,d_C,threads_block);

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

}

//___________________________________________________________________________________________________
// matMul_cublas
// computes the matrix product of double matrices with arbitrary size on device
// utilisation of cublas
void matMul_cublas(const double *A, const double *B, int M,int N,int K,double *C,int threads_block)
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
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}

void matMul_gpu_sm(const double *A, const double *B, int M,int N,int K,double *C)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,M*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,N*K*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,M*N*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, K*N*sizeof(double), cudaMemcpyHostToDevice));


    matMul_sm_onDev(d_A, d_B, M,N, K,d_C);

    // Read C from device memory
    cudaMemcpy(C, d_C, M*N*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}


void matMul_gpu_sm_tr(const double *A, const double *B,int A_TRANSP,int B_TRANSP,int rows_op_A,int cols_op_A,int rows_op_B,int cols_op_B, double *C)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,rows_op_A*cols_op_A*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,rows_op_B*cols_op_B*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,rows_op_A*cols_op_B*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, rows_op_A*cols_op_A*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, rows_op_B*cols_op_B*sizeof(double), cudaMemcpyHostToDevice));

    matMul_sm_onDev_tr(d_A,d_B,A_TRANSP,B_TRANSP, rows_op_A,cols_op_A,rows_op_B,cols_op_B,d_C);

    // Read C from device memory
    cudaMemcpy(C, d_C, rows_op_A*cols_op_B*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}

void matMul_gpu_sm_tr_ind(const double *A, const double *B,int A_TRANSP,int B_TRANSP,int rows_op_A,int cols_op_A,int rows_op_B,int cols_op_B, double *C)
{

    double *d_A,*d_B,*d_C;
    CHECK(cudaMalloc((void**)&d_A,rows_op_A*cols_op_A*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_B,rows_op_B*cols_op_B*sizeof(double)));
    CHECK(cudaMalloc((void**)&d_C,rows_op_A*cols_op_B*sizeof(double)));

    CHECK(cudaMemcpy(d_A, A, rows_op_A*cols_op_A*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, rows_op_B*cols_op_B*sizeof(double), cudaMemcpyHostToDevice));

    matMul_sm_onDev_tr_ind(d_A,d_B,A_TRANSP,B_TRANSP, rows_op_A,cols_op_A,rows_op_B,cols_op_B,d_C);

    // Read C from device memory
    cudaMemcpy(C, d_C, rows_op_A*cols_op_B*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    // Free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}
