// standard c headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>

// own c headers
#include "common.h"
#include "global.h"
#include "matrix_operator.h"
#include "matrix_operator_gpu.h"
#include "test_matrix_operator.h"

// define thresholds
#define MATMUL_COMP(K) (sqrt(2*K)*DBL_EPSILON)




int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Testing MatrixMultiplication at ");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("Performs the following checks:\n\n - matMul on HOST and matMul_gpu1, matMul_gpu2 matMul_gpu_dsm, matMul_gpu_dsm_coa, matMul_cublas on Device\n - HOST and DEVICE same result and All yield same result\n");
  printf(" - ADD, Scale and Matrix Multiply compatible (Distributivgesetz)\n");
  printf(" - ONE, Scale and Matrix Multiply + Transpose and Transpose compatible\n");
  printf(" - Multiply + Transpose and Transpose compatible\n");
  printf("\n_________________________________________________\n");


  srand(seconds());   // Initialization, should only be called once.
  double *A,*B,*B_T,*B_T_T,*C1,*C2,*C3,*C4,*C5,*C6,*C7,*C8,*A_T;
  double *A2,*B2,*A3,*B3,*As,*Bs,*A2s,*B2s,*one_m;
  int same_result;

  int threads_block=1024;


  for(int M=1;M<=256;M*=2){
    for(int N=1;N<=256;N*=2){
      for(int K=1;K<=256;K*=2){

        int A_nelem=M*K;
        int B_nelem=K*N;
        int C_nelem=M*N;

        A = (double *)malloc(A_nelem*sizeof(double));
        B = (double *)malloc(B_nelem*sizeof(double));
        A2 = (double *)malloc(A_nelem*sizeof(double));
        B2 = (double *)malloc(B_nelem*sizeof(double));
        As = (double *)malloc(A_nelem*sizeof(double));
        Bs = (double *)malloc(B_nelem*sizeof(double));
        A2s = (double *)malloc(A_nelem*sizeof(double));
        B2s = (double *)malloc(B_nelem*sizeof(double));
        A3 = (double *)malloc(A_nelem*sizeof(double));
        B3 = (double *)malloc(B_nelem*sizeof(double));
        A_T = (double *)malloc(A_nelem*sizeof(double));
        B_T = (double *)malloc(B_nelem*sizeof(double));
        B_T_T = (double *)malloc(B_nelem*sizeof(double));
        C1 = (double *)malloc(C_nelem*sizeof(double));
        C2 = (double *)malloc(C_nelem*sizeof(double));
        C3 = (double *)malloc(C_nelem*sizeof(double));
        C4 = (double *)malloc(C_nelem*sizeof(double));
        C5 = (double *)malloc(C_nelem*sizeof(double));
        C6 = (double *)malloc(C_nelem*sizeof(double));
        C7 = (double *)malloc(C_nelem*sizeof(double));
        C8 = (double *)malloc(C_nelem*sizeof(double));


        create_random_matrix(A,A_nelem,0,5);
        create_random_matrix(B,B_nelem,0,5);



        matMul(A, B,M,N,K,C1);
        matMul_gpu2(A, B,M,N,K,C3,threads_block);
        matMul_gpu_sm_tr(A, B, NORMAL,NORMAL, M, K, K, N, C7);



        same_result=1;
        same_result*=double_equal(C1,C3,C_nelem,MATMUL_COMP(K));
        same_result*=double_equal(C1,C7,C_nelem,MATMUL_COMP(K));



        if (!same_result){
            printf("For M:%d,N:%d,K:%d Methods do not yield the same result\n",M,N,K);
            print_out_matrix(C1,M,N);
            print_out_matrix(C3,M,N);
            print_out_matrix(C7,M,N);


            return EXIT_FAILURE;
        }


        // ____________________________________________________________________________
        // check if double transposing yields original array
        mat_transpose_gpu(B, B_T, K, N,threads_block);
        mat_transpose_gpu(B_T, B_T_T, N, K,threads_block);
        same_result=double_equal(B,B_T_T,B_nelem,DBL_EPSILON);

        if (!same_result){
           printf("For M:%d,N:%d,K:%d Double Transposing does not yield original Matrix\n",M,N,K);
           print_out_matrix(B,K,N);
           print_out_matrix(B_T_T,K,N);
           return EXIT_FAILURE;
        }

        // _____________________________________________________________________________
        // check add matrixmultiply consistency
        int max=10;
        create_random_matrix(A2,A_nelem,0,5);
        create_random_matrix(B2,B_nelem,0,5);
        double alpha,beta,gamma,delta;

        // get scalars
        alpha=(max*(double)rand()/(double)RAND_MAX);
        beta=(max*(double)rand()/(double)RAND_MAX);
        gamma=(max*(double)rand()/(double)RAND_MAX);
        delta=(max*(double)rand()/(double)RAND_MAX);

        // first scale then add then multiply variant
        matrix_scalar_cpu(As,A,alpha,A_nelem);
        matrix_scalar_cpu(A2s,A2,beta,A_nelem);
        matrix_scalar_cpu(Bs,B,gamma,B_nelem);
        matrix_scalar_cpu(B2s,B2,delta,B_nelem);
        matrix_add_cpu(A3,As,A2s,A_nelem);
        matrix_add_cpu(B3,Bs,B2s,B_nelem);
        matMul_gpu_sm_tr(A3, B3, NORMAL,NORMAL, M, K, K, N, C1);

        // first multiply then scale then add variant
        matMul_gpu_sm_tr(A, B, NORMAL,NORMAL, M, K, K, N, C2);
        matMul_gpu_sm_tr(A2, B2, NORMAL,NORMAL, M, K, K, N, C3);
        matMul_gpu_sm_tr(A, B2, NORMAL,NORMAL, M, K, K, N, C4);
        matMul_gpu_sm_tr(A2, B, NORMAL,NORMAL, M, K, K, N, C5);
        matrix_scalar_cpu(C2,C2,alpha*gamma,C_nelem);
        matrix_scalar_cpu(C3,C3,beta*delta,C_nelem);
        matrix_scalar_cpu(C4,C4,alpha*delta,C_nelem);
        matrix_scalar_cpu(C5,C5,beta*gamma,C_nelem);
        matrix_add_cpu(C6,C2,C3,C_nelem);
        matrix_add_cpu(C6,C6,C4,C_nelem);
        matrix_add_cpu(C6,C6,C5,C_nelem);

        // check for equal result
        if (!double_equal(C6,C1,C_nelem,sqrt(4*(1+2*K))*DBL_EPSILON)){
           printf("For M:%d,N:%d,K:%d ADD and MM not compatible\n",M,N,K);
           printf("%e ; %e\n",sqrt(4*(1+2*K))*DBL_EPSILON,max_abs_diff(C1,C6,C_nelem) );
           return EXIT_FAILURE;
        }


        // check if scaled One is consistent with multiplication and scale and transpose matrix consistent with multiply transpose
        if(M==N && N==K){
            one_m = (double *)malloc(N*N*sizeof(double));

            double alpha=(double)(max*(double)rand()/(double)RAND_MAX);

            ONE_Matrix(one_m,N,alpha);

            mat_transpose_gpu(B, B_T, K, N,threads_block);
            matrix_scalar_cpu(B_T,B_T,alpha,B_nelem);
            matMul_gpu_sm_tr(one_m, B, NORMAL,TRANSPOSED, N, N, N, N, C1);

            mat_transpose_gpu(A, A_T, M, K,threads_block);
            matrix_scalar_cpu(A_T,A_T,alpha,A_nelem);
            matMul_gpu_sm_tr(A, one_m, TRANSPOSED,NORMAL, N, N, N, N, C2);

            // check for equal result
            if (!double_equal(B_T,C1,B_nelem,MATMUL_COMP(N))){
               printf("For M:%d,N:%d,K:%d Transpose B and scale and MM_tr not compatible\n",M,N,K);
               printf("%e ; %e\n",10*sqrt(4*K)*DBL_EPSILON,max_abs_diff(C1,C6,C_nelem) );
               return EXIT_FAILURE;
            }
            if (!double_equal(A_T,C2,B_nelem,MATMUL_COMP(N))){
               printf("For M:%d,N:%d,K:%d Transpose A and scale and MM_tr not compatible\n",M,N,K);
               printf("%e ; %e\n",10*sqrt(4*K)*DBL_EPSILON,max_abs_diff(C1,C6,C_nelem) );
               return EXIT_FAILURE;
            }
            free(one_m);
        }


        //checks tranpose in combination with matmul_sm_tr /-ind
        if(M==N){

          C1 = (double *)malloc(M*M*sizeof(double));
          C2 = (double *)malloc(M*M*sizeof(double));
          C7 = (double *)malloc(M*M*sizeof(double));

          C3 = (double *)malloc(K*K*sizeof(double));
          C4 = (double *)malloc(K*K*sizeof(double));
          C8 = (double *)malloc(K*K*sizeof(double));


          mat_transpose_gpu(A, A_T, M, K,threads_block);

          matMul_gpu2(A2, A_T,M,M,K,C1,threads_block);
          matMul_gpu_sm_tr(A2, A, NORMAL,TRANSPOSED, M, K, K, M,C2);

          if (!double_equal(C1,C2,M*M,MATMUL_COMP(K))){
             printf("For M:%d,N:%d,K:%d A*A_T same result %d\n",M,N,K);
             return EXIT_FAILURE;
          }

          matMul_gpu2(A_T, A2,K,K,M,C3,threads_block);
          matMul_gpu_sm_tr(A, A2, TRANSPOSED,NORMAL, K, M, M, K,C4);

          if (!double_equal(C3,C4,K*K,MATMUL_COMP(M))){
             printf("For M:%d,N:%d,K:%d A*A_T not same result\n",M,N,K);
             return EXIT_FAILURE;
          }

          C5 = (double *)malloc(K*K*sizeof(double));
          C6 = (double *)malloc(K*K*sizeof(double));

          mat_transpose_gpu(B, B_T, K, N,threads_block);
          matMul_gpu2(A_T, B_T,K,K,M,C5,threads_block);
          matMul_gpu_sm_tr(A, B, TRANSPOSED,TRANSPOSED, K, M, M, K,C6);

          if (!double_equal(C5,C6,K*K,MATMUL_COMP(M))){
             printf("For M:%d,N:%d,K:%d A_T*B_T not same result\n",M,N,K);
             return EXIT_FAILURE;
          }
        }



        free(A);
        free(B);
        free(A2);
        free(B2);
        free(As);
        free(Bs);
        free(A2s);
        free(B2s);
        free(A3);
        free(B3);
        free(C1);
        free(C2);
        free(C3);
        free(C4);
        free(C5);
        free(C6);
        free(C7);
        free(C8);
        free(B_T);
        free(B_T_T);
        free(A_T);

      }
    }
  }


  printf("All Checks successfull\n");



  return EXIT_SUCCESS;
}
