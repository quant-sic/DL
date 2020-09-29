/*
  HEADER OPERATOR DEVICE MATRIX OPERATIONS

  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 11.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _MATRIX_OPERATOR_H_
#define _MATRIX_OPERATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

  enum get_element_func_names{NORMAL=0,TRANSPOSED=1};


  // device without overhead
  void mat_transpose_onDev(const double* dev_mat_in, double* dev_mat_out, int rows, int cols, int threads_block);
  void mulAdd_onDev(double *dev_res,const double* dev_lhs,const double* dev_rhs,const double factor,int size,int threads_block);
  void matrix_hadamard_onDev(double* dev_res,const double* dev_lhs,const double* dev_rhs,int     size,int     threads_block);
  void matrix_add_onDev(double* dev_res,const double* dev_lhs,const double* dev_rhs,int     size,int     threads_block);
  void add_along_axis_onDev(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols, int dim_add, int size_vec);
  void add_reduce_dim_onDev(const double* dev_mat_in,double *dev_vec_out, int rows,int cols, int dim_red,int size_vec);
  void matMul_sm_onDev_tr(const double *d_A, const double *d_B,const int A_TRANSP,const int B_TRANSP,const int rows_op_A,const int cols_op_A,const int rows_op_B,const int cols_op_B,double *d_C);

  void div_along_axis_onDev(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols, int dim_div, int size_vec);
  void mulAdd_direct_onDev(double* dev_lhs, const double* dev_rhs, const double factor, int size, int threads_block);
  void mulAdd_direct_cpu(double* dev_lhs, const double* dev_rhs, const double factor, int size);

  void add_along_col_direct_onDev(double* dev_mat, const double* dev_vec, int rows, int cols);
  void add_along_col_direct_cpu(double* dev_mat, const double* dev_vec, int rows, int cols);


  // host functions
  void matrix_hadamard_cpu (double* res, const double* lhs, const double* rhs, int size);
  void matrix_add_cpu      (double* res, const double* lhs, const double* rhs, int size);
  void matMul              (const double *A,const double *B,int M,int N, int K,double *C);
  void matrix_transpose_cpu(double* out, double* inp, int rows, int cols);
  void mulAdd_cpu(double* res, const double* lhs, const double* rhs, const double factor, int size);
  void add_reduce_dim_cpu(const double* mat_in,double *vec_out, int rows,int cols, int dim_red,int dim_vec);
  void add_along_axis_cpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_add, int size_vec);
  void div_along_axis_cpu(const double* mat_in,const double *vec,double* mat_out, int rows,int cols, int dim_div, int size_vec);

  // special helper functions
  void matrix_scalar_cpu     (double* out, const double* inp, double factor, int size);
  void matrix_hadamard_gpu_test_dev(double* res,const double* lhs,const double* rhs,int     size,int     threads_block,int op_p_th);
#ifdef __cplusplus
}
#endif
#endif // _MATRIX_OPERATOR_H_
