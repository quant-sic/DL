#ifndef _MAT_MUL_H_
#define _MAT_MUL_H_

enum mat_op_enum{NORMAL=0,TRANSPOSED=1};

template<typename real_type>
extern void mat_mul_tr_onDev(const real_type *d_A, const real_type *d_B,const int A_TRANSP,const int B_TRANSP,const int rows_op_A,const int cols_op_A,const int rows_op_B,const int cols_op_B,real_type *d_C);

template<typename real_type>
extern void mat_mul_coa_onDev(const real_type *d_A, const real_type *d_B, int M,int N,int K,real_type *d_C);

template<typename real_type>
extern void mat_mul_cpu(const real_type *A,const real_type *B,int M,int N, int K,real_type *C);

template<typename real_type>
extern __global__ void mat_mul_coa_kernel(const real_type *d_A_T,const real_type *d_B,int M, int N,int K,real_type *d_C);

template <typename real_type>
extern __global__ void mat_mul_tr_kernel(const real_type *__restrict__ d_A,const real_type * __restrict__ d_B,const int A_TRANSP,const int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,real_type *d_C);

#endif // _MAT_MUL_H_