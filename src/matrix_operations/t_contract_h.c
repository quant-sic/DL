#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include <assert.h>
#include "t_contract_h.h"


// fill contraction map
void CM(Tensor *T){

  int ndim=(*T).ndim;

  int **ctr= (int **)malloc(ndim*sizeof(int*));
  for(int i =0;i<ndim;i++){
    ctr[i]=(int *)calloc((*T).dim_sizes[i],sizeof(int));
  }

  int *factors= (int *)malloc(ndim*sizeof(int));
  for(int i =0;i<ndim;i++){
    factors[i]=1;
  }
  for(int i =ndim-2;i>=0;i--){
    factors[i]=factors[i+1]*(*T).dim_sizes[i+1];
  }

  for(int k =0;k<ndim;k++){

    for(int i =0;i<(*T).nelem;i++){

        int ki=(i/factors[k])%(*T).dim_sizes[k];
        int idx=0;
        for(int l =0;l<ndim;l++){
            idx+=((i/factors[l])%(*T).dim_sizes[l])*factors[l];
        }
        (*T).CM[k][ki][ctr[k][ki]++]+=idx;
    }
  }
}


void Tensor_prod3(Tensor A,Tensor B,int dimA, int dimB,Tensor *C){

  assert(A.dim_sizes[dimA]==B.dim_sizes[dimB]);
  int sds=A.dim_sizes[dimA];

  int C_step=B.nelem/B.dim_sizes[dimB];

  double interm_sum=0;
  for(int j =0;j<A.nelem/A.dim_sizes[dimA];j++){
    for(int l =0;l<B.nelem/B.dim_sizes[dimB];l++){
      interm_sum=0;
      for(int i =0;i<sds;i++){
          interm_sum+=(A.data[A.CM[dimA][i][j]]*B.data[B.CM[dimB][i][l]]);
      }

      (*C).data[j*C_step+l]=interm_sum;
    }
  }
}


// matrix method
void Tensor_prod(Tensor A,Tensor B,int dimA, int dimB,Tensor *C){

    if(A.ndim==2 && B.ndim==2 && dimA==1 && dimB==0){

      assert(A.dim_sizes[dimA]==B.dim_sizes[dimB]);
      int sds=A.dim_sizes[dimA];

      int i,j,k;
      double interm_sum;

      for (i=0;i<A.dim_sizes[0];i++){

          for (j=0;j<B.dim_sizes[1];j++){

              interm_sum=0;

              for (k=0;k<sds;k++){
                  interm_sum+=A.data[i*sds+k]*B.data[k*B.dim_sizes[1]+j];

              }
              (*C).data[i*B.dim_sizes[1]+j]=interm_sum;
          }
      }
    }
}

/*
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
*/
