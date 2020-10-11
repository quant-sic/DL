#ifndef _KERNEL_UTILS_H_
#define _KERNEL_UTILS_H_

// #ifdef __cplusplus
// extern "C" {
// #endif

  // enums
  enum pointwise_names{RELU_FUNC=0,SIGMOID_FUNC=1,D_RELU_FUNC=2,D_SIGMOID_FUNC=3};
  enum comb_pointwise_names{ADD=0,DIV=1,MUL=2,EXP_MAX=3,COMP_MAX=4,D_CAT_CROSS_ENT=5,D_RMS=6,SUMM_CAT=7,SUMM_RMS=8,D_CCE_SOFTMAX=9};

  // function typedefs
  typedef double (*get_element_func)(const double *,const int,const int,const int,const int);
  typedef double (*pointwise)(double);
  typedef double (*comb_pointwise)(double,double);

  // __host__ __device__ Functions
  __device__ __host__ double add_func(double a,double b);
  __device__ __host__ double div_func(double a,double b);
  __device__ __host__ double mul_func(double a,double b);
  __host__ __device__ double mulAdd(double lhs, double rhs,double factor);

  __device__ __host__ double sigmoid(double x);
  __device__ __host__ double d_sigmoid(double x);
  __device__ __host__ double relu(double x);
  __device__ __host__ double d_relu(double x);
  __device__ __host__ double exp_max(double in,double max);
  __device__ __host__ double comp_max(double x,double y);

  __host__ __device__ double d_cat_cross_ent(double in,double target);
  __host__ __device__ double d_rms_func(double in,double target);
  __host__ __device__ double d_cce_softmax(double in,double target);
  __host__ __device__ double summand_cat_cross_ent(double in,double target);
  __host__ __device__ double summand_rms(double in,double target);


  __host__ __device__ double get_element(const double *mat,const int row_stride,const int col_stride,const int o_row,const int o_col);
  __host__ __device__ double get_element_transposed(const double *mat,const int row_stride,const int col_stride,const int o_row,const int o_col);


  //___________________________________________________________________________________________________
  // kernel functions
  __global__ void comb_pointwise_1d_kernel(double* res,const double* lhs,const double* rhs,int     size,int func_int);
  __global__ void apply_pointwise_kernel(const double *dev_in, double *dev_out, int size,int func_int);


  __global__ void hadamard_func_kernel(const double *dev_da,double *dev_z,double *dev_dz,int size,int func_int);
  __global__ void matrix_hadamard_kernel(double* res, const double* lhs, const double* rhs, int size);
  __global__ void matrix_add_kernel     (double* res, const double* lhs, const double* rhs, int size);
  __global__ void mat_transpose_kernel(const double *mat_in, double *mat_out, int rows, int cols);
  __global__ void matMul_kernel1(const double *d_A,const double *d_B,int M, int N,int K,double *d_C);
  __global__ void matMul_kernel2(const double *d_A_T,const double *d_B,int M, int N,int K,double *d_C);
  __global__ void matMul_kernel_dsm(const double *d_A,const double *d_B,int M, int N,int K,double *d_C);
  __global__ void matMul_kernel_dsm_coa(const double *d_A_T,const double *d_B,int M, int N,int K,double *d_C);
  __global__ void matMul_kernel_sm(const double *d_A,const double *d_B,int M, int N,int K,double *d_C);
  __global__ void mulAdd_kernel(double *dev_res,const double* dev_lhs,const double* dev_rhs,const double factor,int size);
  __global__ void matMul_kernel_sm_tr(const double * __restrict__ d_A,const double * __restrict__ d_B,int A_TRANSP,int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,double *d_C);
  __global__ void matMul_kernel_sm_tr2(const double *__restrict__ d_A,const double * __restrict__ d_B,int A_TRANSP,int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,double *d_C);
  __global__ void matMul_kernel_sm_tr3(const double *__restrict__ d_A,const double * __restrict__ d_B,int A_TRANSP,int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,double *d_C);

  __global__ void mulAdd_direct_kernel(double* lhs, const double* rhs, const double factor, int size);
  __global__ void add_along_col_direct_kernel(double* dev_mat, const double* dev_vec, int rows, int cols);
  
  
  
  __global__ void add_reduce_rows_kernel(const double * __restrict__  dev_mat_in, double * __restrict__  dev_vec_out, const int rows, const int cols);
  __global__ void add_reduce_cols_kernel(const double* dev_mat_in,double *dev_vec_out, int rows,int cols);
  
  
  
  
  __global__ void func_along_axis_x_kernel(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols,int func_int);
  __global__ void func_along_axis_y_kernel(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols,int func_int);
  __global__ void get_max_kernel(const double * __restrict__ data, int length,double *dev_res);

  __global__ void combine_pointwise_kernel_actsoft(const double* dev_in,const double dev_max,double* dev_out,int size);
  __global__ void d_softmax_activation_kernel(const double * __restrict__ dev_softmax, double *dev_delta, int batchsize,int neurons_out);
  __global__ void softmax_backprop_kernel(const double * __restrict__ dev_da,const double * __restrict__ dev_d_softmax,int batchsize, int neurons_out,double *dev_dz);

  __global__ void add_reduce_rows_func_kernel(const double * __restrict__  dev_in, const double * __restrict__  dev_target, double* dev_res, const int size,int func_int);

  __global__ void d_softmax_activation_unrolled_kernel(const double * __restrict__ dev_softmax, double *dev_delta, int batchsize,int neurons_out);
  __global__ void softmax_backprop_mm_kernel(const double * __restrict__ dev_da,const double * __restrict__ dev_d_softmax,int batchsize, int neurons_out,double *dev_dz);
  __global__ void softmax_backprop_T_kernel(const double * __restrict__ dev_da_T,const double * __restrict__ dev_d_softmax,int batchsize, int neurons_out,double *dev_dz);


  // minor hack. these functions are defined in costfunctions.cu.
  double sum_func_array(const double *in, const double* target, comb_pointwise func, int size);
  double sum_func_array_onDev(const double *dev_in,const double* dev_target,int func_int,int size);

// #ifdef __cplusplus
// }
// #endif

#endif // _KERNEL_UTILS_H_
