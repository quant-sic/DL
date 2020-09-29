//#define MAIN_PROGRAM

// c standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cblas.h>
#include <float.h>
#include <sys/time.h>

// own c headers
#include "../common.h"
#include "../matrix_operations/tensor.h"
#include "../matrix_operations/matrix_operator.h"
#include "../matrix_operations/matrix_operator_gpu.h"
#include "../matrix_operations/test_matrix_operator.h"

// cublas headers
#include "cublas_v2.h"
#include <cuda_runtime.h>

#include <cblas.h>







int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nTiming Matrix Multiplication at");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("Performs the following timings:\n\n - matMul on HOST and matMul_gpu1, matMul_gpu2 matMul_gpu_dsm, matMul_gpu_dsm_coa, matMul_cublas,matMul_gpu_sm_tr,matMul_gpu_sm_tr_ind on Device with and without copying\n \n");
  printf("\n_________________________________________________\n");

// GPU Functions
  srand(seconds());   // Initialization, should only be called once.

  double start,t,t1,t2,t3,t4,t5,t6,t7,t8,t9;

  FILE *fp = fopen("../analysis/matMulTimes.txt", "w");

  fprintf(fp,"N\tT_B\tMM1\tMM2\tDSM\tDSM_COA\tcuBlas\tSM\tSM_tr\tCPU\tSM_trInd\n");
  printf("N\tT_B\tMM1\tMM2\tDSM\tDSM_COA\tcuBlas\tSM\tSM_tr\tCPU\tSM_trInd\n");

  int threads_block;
  //maximum shift for maximum dimension size
  int max_shift=11;

  // loop over dimension sizes
  for(int i=0;i<=max_shift;i++){

    // set times on max for every size
    t1=t2=t3=t4=t5=t6=t7=t8=t9=DBL_MAX;

    // set up dimensions and arrays
    int N=1<<i;
    int dimsA[2]={N,N};
    int dimsB[2]={N,N};
    int A_nelem=dimsA[0]*dimsA[1];
    int B_nelem=dimsB[0]*dimsB[1];
    int C_nelem=dimsA[0]*dimsB[1];

    double *A = (double *)malloc(A_nelem*sizeof(double));
    double *B = (double *)malloc(B_nelem*sizeof(double));
    double *C = (double *)malloc(C_nelem*sizeof(double));

    // set sqrt(threads_block)
    for(int k=8;k<=32;k*=2){
        threads_block=k*k;

        // best of 3
        for (int j=0;j<3;j++){

              create_random_matrix(A,A_nelem,0,10);
              create_random_matrix(B,B_nelem,0,10);


              start=seconds();
              matMul_gpu1(A, B, dimsA[0],dimsB[1],dimsA[1],C,threads_block);
              t=seconds()-start;
              t1=(t<t1) ? t : t1 ;


              start=seconds();
              matMul_gpu2(A, B, dimsA[0],dimsB[1],dimsA[1],C,threads_block);
              t=seconds()-start;
              t2=(t<t2) ? t : t2 ;

              start=seconds();
              matMul_gpu_dsm(A, B, dimsA[0],dimsB[1],dimsA[1],C,threads_block);
              t=seconds()-start;
              t3=(t<t3) ? t : t3 ;

              start=seconds();
              matMul_gpu_sm(A, B, dimsA[0],dimsB[1],dimsA[1],C);
              t=seconds()-start;
              t6=(t<t6) ? t : t6 ;

              start=seconds();
              matMul_gpu_dsm_coa(A, B, dimsA[0],dimsB[1],dimsA[1],C,threads_block);
              t=seconds()-start;
              t4=(t<t4) ? t : t4 ;



              start=seconds();
              matMul_cublas(A, B, dimsA[0],dimsB[1],dimsA[1],C,threads_block);
              t=seconds()-start;
              t5=(t<t5) ? t : t5 ;

              start=seconds();
              matMul_gpu_sm_tr(A, B, NORMAL,NORMAL,dimsA[0],dimsA[1],dimsB[0],dimsB[1],C);
              t=seconds()-start;
              t7=(t<t7) ? t : t7 ;

              if (i<=9){
                start=seconds();
                matMul(A, B, dimsA[0],dimsB[1],dimsA[1],C);
                t=seconds()-start;
                t8=(t<t8) ? t : t8 ;
              }

              start=seconds();
              matMul_gpu_sm_tr_ind(A, B, NORMAL,NORMAL,dimsA[0],dimsA[1],dimsB[0],dimsB[1],C);
              t=seconds()-start;
              t9=(t<t9) ? t : t9 ;

      }
      // print to file
      printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",N,threads_block,t1,t2,t3,t4,t5,t6,t7,t8,t9);
      fprintf(fp,"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",N,threads_block,t1,t2,t3,t4,t5,t6,t7,t8,t9);

    }


    free(A);
    free(B);
    free(C);

  }

  fclose (fp);

// onDev functions

FILE *fp2 = fopen("../analysis/matMulTimesOnDev.txt", "w");

fprintf(fp2,"N\tT_B\tMM1\tMM2\tDSM\tDSM_COA\tcuBlas\tSM\tSM_tr\tCPU\tSM_trInd\n");

printf("\nonDev\n");

for(int i=0;i<=max_shift;i++){

  t1=t2=t3=t4=t5=t6=t7=t8=t9=DBL_MAX;

  int N=1<<i;
  int dimsD[2]={N,N};
  int dimsE[2]={N,N};
  int D_nelem=dimsD[0]*dimsD[1];
  int E_nelem=dimsE[0]*dimsE[1];
  int F_nelem=dimsD[0]*dimsE[1];

  double *D = (double *)malloc(D_nelem*sizeof(double));
  double *E = (double *)malloc(E_nelem*sizeof(double));
  double *F = (double *)malloc(F_nelem*sizeof(double));

  double *dev_D = (double *)malloc(D_nelem*sizeof(double));
  double *dev_E = (double *)malloc(E_nelem*sizeof(double));
  double *dev_F = (double *)malloc(F_nelem*sizeof(double));

  CHECK(cudaMalloc((void**)&dev_D, D_nelem*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_E, E_nelem*sizeof(double)));
  CHECK(cudaMalloc((void**)&dev_F, F_nelem*sizeof(double)));

  for(int k=8;k<=32;k*=2){
      threads_block=k*k;

      // best of 3
      for (int j=0;j<3;j++){

        create_random_matrix(D,D_nelem,0,10);
        create_random_matrix(E,E_nelem,0,10);

        copy_host_to_device_double(D,dev_D,D_nelem);
        copy_host_to_device_double(E,dev_E,E_nelem);


        start=seconds();
        matMul_onDev1(dev_D, dev_E, dimsD[0],dimsE[1],dimsD[1],dev_F,threads_block);
        t=seconds()-start;
        t1=(t<t1) ? t : t1 ;


        start=seconds();
        matMul_onDev2(dev_D, dev_E, dimsD[0],dimsE[1],dimsD[1],dev_F,threads_block);
        t=seconds()-start;
        t2=(t<t2) ? t : t2 ;


        start=seconds();
        matMul_dsm_onDev(dev_D, dev_E, dimsD[0],dimsE[1],dimsD[1],dev_F,threads_block);
        t=seconds()-start;
        t3=(t<t3) ? t : t3 ;


        start=seconds();
        matMul_sm_onDev(dev_D, dev_E, dimsD[0],dimsE[1],dimsD[1],dev_F);
        t=seconds()-start;
        t6=(t<t6) ? t : t6 ;

        start=seconds();
        matMul_dsm_coa_onDev(dev_D, dev_E, dimsD[0],dimsE[1],dimsD[1],dev_F,threads_block);
        t=seconds()-start;
        t4=(t<t4) ? t : t4 ;

        cublasStatus_t stat;
        cublasHandle_t handle;

        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("CUBLAS initialization failed\n");
        }
        const double alpha=1.0;
        const double beta=0.0;
        // Invoke kernel
        start=seconds();
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimsE[1], dimsD[0], dimsE[0],&alpha,(const double *)dev_E, dimsE[1],(const double *)dev_D, dimsD[1],&beta,(double *)dev_F, dimsE[1]);
        CHECK(cudaDeviceSynchronize());
        cublasDestroy(handle);
        t=seconds()-start;
        t5=(t<t5) ? t : t5 ;

        start=seconds();
        matMul_sm_onDev_tr(dev_D, dev_E, NORMAL,NORMAL,dimsD[0],dimsD[1],dimsE[0],dimsE[1],dev_F);
        t=seconds()-start;
        t7=(t<t7) ? t : t7 ;

        if (i<=9){
          start=seconds();
          matMul(D, E, dimsD[0],dimsE[1],dimsD[1],F);
          t=seconds()-start;
          t8=(t<t8) ? t : t8 ;
        }

        start=seconds();
        matMul_sm_onDev_tr_ind(dev_D, dev_E, NORMAL,NORMAL,dimsD[0],dimsD[1],dimsE[0],dimsE[1],dev_F);
        t=seconds()-start;
        t9=(t<t9) ? t : t9 ;
    }

    // print to file
    printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",N,threads_block,t1,t2,t3,t4,t5,t6,t7,t8,t9);
    fprintf(fp2,"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",N,threads_block,t1,t2,t3,t4,t5,t6,t7,t8,t9);
  }


  free(D);
  free(E);
  free(F);
  CHECK(cudaFree(dev_D));
  CHECK(cudaFree(dev_E));
  CHECK(cudaFree(dev_F));

  }

  fclose (fp2);


  return EXIT_SUCCESS;
}
