// TIMES THE CBLAS DGEMM FUNCTION



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "test_matrix_operator.h"
#include <assert.h>
#include <cblas.h>
#include "common.h"
#include <float.h>


int main(int argc, char **argv)
{
 double start,t1,t;

 FILE *fp = fopen("../analysis/matMulTimesCBlas.txt", "w");

 fprintf(fp,"N\tCBlas\n");

 int threads_block;

 int max_shift=12;

 for(int i=0;i<=max_shift;i++){

   t1=DBL_MAX;

   int N=1<<i;
   int dimsA[2]={N,N};
   int dimsB[2]={N,N};
   int A_nelem=dimsA[0]*dimsA[1];
   int B_nelem=dimsB[0]*dimsB[1];
   int C_nelem=dimsA[0]*dimsB[1];

   double *A = (double *)malloc(A_nelem*sizeof(double));
   double *B = (double *)malloc(B_nelem*sizeof(double));
   double *C = (double *)malloc(C_nelem*sizeof(double));

   // best of 5
   for(int k=8;k<=32;k*=2){

       threads_block=k*k;
       for (int j=0;j<5;j++){

             create_random_matrix(A,A_nelem,0,10);
             create_random_matrix(B,B_nelem,0,10);

             // invoke cblas
             start=seconds();
             cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,dimsA[0],dimsB[1],dimsA[1],1.0,A,dimsA[1],B,dimsB[1],0.0,C,dimsA[0]);
             t=seconds()-start;
             t1=(t<t1) ? t : t1 ;


     }
     fprintf(fp,"%d\t%f\n",N,t1);

   }


   free(A);
   free(B);
   free(C);

 }

 fclose (fp);

  return EXIT_SUCCESS;
}
