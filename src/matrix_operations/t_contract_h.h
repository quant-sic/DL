#ifndef T_CONTRACT_H_H
#define T_CONTRACT_H_H

#include "tensor.h"

extern void Tensor_prod(Tensor A,Tensor B,int dimA, int dimB,Tensor *C);
extern void Tensor_prod2(Tensor A,Tensor B,int dimA, int dimB,Tensor *C);
extern void Tensor_prod3(Tensor A,Tensor B,int dimA, int dimB,Tensor *C);

// extern void matMul(double *A,double *B,int M,int N, int K,double *C);
extern void CM(Tensor *T);

#endif
