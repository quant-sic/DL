#ifndef _MATRIX_OPERATOR_GPU_H_
#define _MATRIX_OPERATOR_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

  // gpu functions
  void matrix_hadamard_gpu(double* res, const double* lhs, const double* rhs, int size, int threads_block);
  void matrix_add_gpu     (double* res, const double* lhs, const double* rhs, int size, int threads_block);
  void mat_transpose_gpu(const double* mat_in, double* mat_out, int rows, int cols, int threads_block);
  void matMul_gpu1(const double *A, const double *B, int M,int N,int K,double *C, int threads_block);
  void matMul_gpu2(const double *A, const double *B, int M,int N,int K,double *C,int threads_block);
  void matMul_gpu_dsm(const double *A, const double *B, int M,int N,int K,double *C,int threads_block);
  void matMul_gpu_dsm_coa(const double *A, const double *B, int M,int N,int K,double *C,int threads_block);
  void matMul_cublas(const double *A, const double *B, int M,int N,int K,double *C,int threads_block);
  void matMul_gpu_sm(const double *A, const double *B, int M,int N,int K,double *C);
  void mulAdd_gpu(double* res, const double* lhs, const double* rhs, const double factor, int size);
  void add_along_axis_gpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_add, int size_vec);
  void add_reduce_dim_gpu(const double* mat_in,double *vec_out, int rows,int cols, int dim_red,int size_vec);
  void matMul_gpu_sm_tr(const double *A, const double *B,int A_TRANSP,int B_TRANSP,int rows_op_A,int cols_op_A,int rows_op_B,int cols_op_B, double *C);
  void matMul_gpu_sm_tr_ind(const double *A, const double *B,int A_TRANSP,int B_TRANSP,int rows_op_A,int cols_op_A,int rows_op_B,int cols_op_B, double *C);
  void div_along_axis_gpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_div, int size_vec);


#ifdef __cplusplus
}
#endif
#endif // _MATRIX_OPERATOR_GPU_H_
