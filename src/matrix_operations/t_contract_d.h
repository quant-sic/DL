#ifndef T_CONTRACT_D_H
#define T_CONTRACT_D_H

#include "../global.h"
#include <cuda_runtime.h>

extern void matMul_gpu1(const double *A, const double *B, int M,int N,int K,double *C);
extern void matMul_gpu2(const double *A, const double *B, int M,int N,int K,double *C);
extern void matMul_gpu_sm(const double *A, const double *B, int M,int N,int K,double *C);
extern void matMul_gpu_dsm(const double *A, const double *B, int M,int N,int K,double *C,int threads_block);
extern void matMul_gpu_sm_coa(const double *A, const double *B, int M,int N,int K,double *C);
extern void matMul_cublas(const double *A, const double *B, int M,int N,int K,double *C);
extern void mat_transpose_gpu(const double* mat_in, double* mat_out, int rows, int cols);
// extern void matMul(double *A,double *B,int M,int N, int K,double *C);

#endif
