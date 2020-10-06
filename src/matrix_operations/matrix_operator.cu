/*
  SOURCE OPERATOR DEVICE MATRIX OPERATIONS CONTAINS onDEV AND cpu FUNCTIONS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
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
#include "kernel_utils.h"
#include "../global.h"


// cublas headers
#include "cublas_v2.h"
#include <cuda_runtime.h>








//___________________________________________________________________________________________________
// ADD REDUCE

// function on cpu
// dim_red is the dimension which is supposed to be reduced -> the other dimension will remain and should be equal to the size of the given vector
void add_reduce_dim_cpu(const double* mat_in,double *vec_out, int rows,int cols, int dim_red,int dim_vec){

  assert(dim_red<2 && (dim_red ? rows : cols)==dim_vec);
  memset(vec_out,0,dim_vec*sizeof(double));

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


// function onDev
// dim_red is the dimension which is supposed to be reduced -> the other dimension will remain and should be equal to the size of the given vector
void add_reduce_dim_onDev(const double* dev_mat_in,double *dev_vec_out, int rows,int cols, int dim_red,int size_vec){

    assert(dim_red<2 && (dim_red ? rows : cols)==size_vec);
    CHECK(cudaMemset(dev_vec_out, 0, size_vec*sizeof(double)));
    if(dim_red==0){
      dim3 grid=col_red_grid(cols,rows);
      add_reduce_cols_kernel<<<grid,get_col_red_block()>>>(dev_mat_in,dev_vec_out,rows,cols);
    }else if (dim_red==1){
      dim3 grid=row_red_grid(cols,rows);
      add_reduce_rows_kernel<<<grid,get_row_red_2d_block()>>>(dev_mat_in, dev_vec_out,rows,cols);

    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

}




//___________________________________________________________________________________________________
// COMBINE ALONG AXIS

// on cpu
// dim_add is dimension along which the vector should be added. vector should have the size of the other dimension
void add_along_axis_cpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_add, int size_vec){

  // assert that dimensions match
  assert(dim_add<2 && (dim_add ? rows : cols)==size_vec);

  // case add along dimenion 0
  if(dim_add==0){
    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
        mat_out[i*cols+j]=add_func(mat_in[i*cols+j],vec[j]);
      }
    }

  // case add along dimenion 1
  }else if (dim_add==1){
    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
        mat_out[i*cols+j]=add_func(mat_in[i*cols+j],vec[i]);
      }
    }
  }
}


// onDev
// dim_add is dimension along which the vector should be added. vector should have the size of the other dimension
void add_along_axis_onDev(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols, int dim_add, int size_vec){

  // assert dimensions match
  assert(dim_add<2 && (dim_add ? rows : cols)==size_vec);

  // case add along dimenion 0
  if(dim_add==0){
    func_along_axis_y_kernel<<<pointwise2d_grid(cols,rows),get_pointwise2d_block()>>>(dev_mat_in,dev_vec,dev_mat_out,rows,cols,ADD);

  // case add along dimenion 1
  }else if (dim_add==1){
    func_along_axis_x_kernel<<<pointwise2d_grid(cols,rows),get_pointwise2d_block()>>>(dev_mat_in,dev_vec,dev_mat_out,rows,cols,ADD);
  }
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// add along col direct cpu
// adds directly onto the matrix
void add_along_col_direct_cpu(double*       dev_mat,const double* dev_vec,int           rows,int           cols){

  for (int col = 0; col < cols; col++)
    for (int row = 0; row < rows; row++)
      dev_mat[row*cols+col] += dev_vec[col];
}

//_______________________________________________________________________________________________
// add along cols -> linear forward gpu
// adds directly onto the matrix
void add_along_col_direct_onDev(double*       dev_mat,const double* dev_vec,int           rows,int           cols)
{
  dim3 grid=pointwise2d_grid(cols, rows);
  add_along_col_direct_kernel<<<grid,get_pointwise2d_block()>>>(dev_mat, dev_vec, rows, cols);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// scale along axis cpu
// dim_div is dimension along which the vector should be divided. vector should have the size of the other dimension
void div_along_axis_cpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_div, int size_vec){

  assert(dim_div<2 && (dim_div ? rows : cols)==size_vec);

  if(dim_div==0){

    for(int i=0;i<rows;i++){
      for(int j=0;j<cols;j++){
	        mat_out[i*cols+j]=div_func(mat_in[i*cols+j],vec[j]);
      }
    }
  }else if (dim_div==1){

      for(int i=0;i<rows;i++){
      	for(int j=0;j<cols;j++){
      	  mat_out[i*cols+j]=div_func(mat_in[i*cols+j],vec[i]);
      	}
      }
    }
}

// scale along axis onDev
// dim_div is dimension along which the vector should be divided. vector should have the size of the other dimension
void div_along_axis_onDev(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols, int dim_div, int size_vec){

  assert(dim_div<2 && (dim_div ? rows : cols)==size_vec);

    dim3 grid=pointwise2d_grid(cols,rows);

    if(dim_div==0){
      func_along_axis_y_kernel<<<grid,get_pointwise2d_block()>>>(dev_mat_in,dev_vec,dev_mat_out,rows,cols,DIV);
    }else if (dim_div==1){
      func_along_axis_x_kernel<<<grid,get_pointwise2d_block()>>>(dev_mat_in,dev_vec,dev_mat_out,rows,cols,DIV);
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

}






//___________________________________________________________________________________________________
// matrix_scalar_cpu
// multiply matrix by a scalar
void matrix_scalar_cpu(double*       res,const double* inp,double        factor,int           size)
{
  for (int idx = 0; idx < size; idx++)
    res[idx] = inp[idx] * factor;
}




//___________________________________________________________________________________________________
// matrix_transpose_cpu and onDev
// computes the transposed matrix of a double matrix with arbitrary size on cpu
void matrix_transpose_cpu(double* out,
			  double* inp,
			  int     rows,
			  int     cols)
{
  for (int row = 0; row < rows; row++)
    for (int col = 0; col < cols; col++)
      out[col*rows+row] = inp[row*cols+col];
}

// computes the transposed matrix of a double matrix with arbitrary size on device
void mat_transpose_onDev(const double* dev_mat_in, double* dev_mat_out, int rows, int cols, int threads_block){

	int block_dim=(int)sqrt(threads_block);
	threads_block=block_dim*block_dim;
	dim3 block (block_dim,block_dim);
	dim3 grid ((cols+block.x-1)/block.x,(rows+block.y-1)/block.y);

	// Invoke kernel
	mat_transpose_kernel<<<grid, block>>>(dev_mat_in,dev_mat_out,rows,cols);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
}



//___________________________________________________________________________________________________
// Combine pointwise

// Hadamard on Dev
void matrix_hadamard_onDev(double* dev_res,const double* dev_lhs,const double* dev_rhs,int     size,int     threads_block)
{
  // calling hadamard kernel
  comb_pointwise_1d_kernel<<<pointwise_grid(size), get_pointwise_block()>>>(dev_res, dev_lhs, dev_rhs, size,MUL);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// computes the hadamard matrix product for 2 matrices of type double and same size
void matrix_hadamard_cpu(double* res,
			 const double* lhs,
			 const double* rhs,
			 int     size)
{
  // loop over all array elements
  for (int idx = 0; idx < size; idx++)
    res[idx] = mul_func(lhs[idx],rhs[idx]);
}



// Add onDev
void matrix_add_onDev(double* dev_res,
		      const double* dev_lhs,
		      const double* dev_rhs,
		      int     size,
		      int     threads_block)
{
  // calling add kernel
  comb_pointwise_1d_kernel<<<pointwise_grid(size), get_pointwise_block()>>>(dev_res, dev_lhs, dev_rhs, size,ADD);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
}

// computes the addition for two matrices of type double and same size
void matrix_add_cpu(double* res,
		    const double* lhs,
		    const double* rhs,
		    int     size)
{
  for (int idx = 0; idx < size; idx++)
    res[idx] = lhs[idx] + rhs[idx];
}


// computes the Multiply addition for two matrices of type double and double factor and same size on Cpu
void mulAdd_cpu(double* res, const double* lhs, const double* rhs, const double factor, int size)
{
  for (int idx = 0; idx < size; idx++)
    res[idx] = mulAdd(lhs[idx],rhs[idx],factor);
}

// computes the Multiply addition for two matrices of type double and double factor and same size on Device
void mulAdd_onDev(double *dev_res,const double* dev_lhs,const double* dev_rhs,const double factor,int size,int threads_block)
{
  // calling add kernel
  mulAdd_kernel<<<pointwise_grid(size), get_pointwise_block()>>>(dev_res, dev_lhs, dev_rhs,factor,size);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

}

// mull add direct on Dev function -> no in between result (lhs is the result)
void mulAdd_direct_onDev(double*       dev_lhs,
			 const double* dev_rhs,
			 const double  factor,
			 int           size,
			 int           threads_block)
{
  // calling mulAdd_direct kernel
  mulAdd_direct_kernel<<<pointwise_grid(size), get_pointwise_block()>>>(dev_lhs, dev_rhs, factor, size);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

}

// mulAdd direct on cpu
void mulAdd_direct_cpu(double*       lhs,
		       const double* rhs,
		       const double  factor,
		       int           size)
{
  for (int idx = 0; idx < size; idx++)
    lhs[idx] += rhs[idx]*factor;
}




//__________________________________________________________________________________________________
// Matrix Multiplications


// computes the matrixproduct of double matrices with arbitrary size on host
void matMul(const double *A,const double *B,int M,int N, int K,double *C)
{

  double interm_sum;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      interm_sum = 0.;
      for (int kk = 0; kk < K; kk++)
	interm_sum += A[i*K+kk]*B[kk*N+j];
      C[i*N+j] = interm_sum;
    }
  }
}

// computes the matrix product of double matrices with arbitrary size on device
// naive implementation with transposed matrix A -> coalesced memory access
void matMul_onDev2(const double *d_A, const double *d_B, int M,int N,int K,double *d_C, int threads_block)
{

		double *d_A_T;
		CHECK(cudaMalloc((void**)&d_A_T,M*K*sizeof(double)));

		int block_dim=(int)sqrt(threads_block);
		threads_block=block_dim*block_dim;

		dim3 block (block_dim,block_dim);
		dim3 grid ((N+block.x-1)/block.x,(M+block.y-1)/block.y);
		dim3 grid_A_T ((K+block.x-1)/block.x,(M+block.y-1)/block.y);

		// Invoke kernel
		mat_transpose_kernel<<<grid_A_T, block>>>(d_A, d_A_T, M, K);
		CHECK(cudaDeviceSynchronize());
		matMul_kernel2<<<grid, block>>>(d_A_T, d_B,M,N,K,d_C);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
}


// computes the matrix product of double matrices with arbitrary size on device
// tiled implementation with static shared memory and transposed matrices using if for 4 kernel versions
void matMul_sm_onDev_tr(const double *d_A, const double *d_B,const int A_TRANSP,const int B_TRANSP,const int rows_op_A,const int cols_op_A,const int rows_op_B,const int cols_op_B,double *d_C)
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
    matMul_kernel_sm_tr3<<<matrix_mul_grid(cols_op_B,rows_op_A), get_matrix_mul_block()>>>((const double *)d_A, (const double *)d_B,A_TRANSP,B_TRANSP,rows_op_A,cols_op_B,cols_op_A,rows_A,cols_A,rows_B,cols_B,d_C);


    // error handling
    if(cudaDeviceSynchronize()||cudaGetLastError()){
      printf("Error in matMul_sm_tr_onDev\n");
      printf("Matrix Dimensions: M %d,N %d,K %d\n",rows_op_A,cols_op_B,cols_op_A);

      CHECK(cudaDeviceSynchronize());
      CHECK(cudaGetLastError());
    }
}
