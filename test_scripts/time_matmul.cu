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
#include "common.h"
#include "test_matrix_operator.h"
#include "mat_mul.h"

// cublas headers
#include "cublas_v2.h"
#include <cuda_runtime.h>







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

  //maximum shift for maximum dimension size
  int max_shift=11;

// onDev functions

  FILE *fp2 = fopen("analysis/matMulTimesOnDev.txt", "w");

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

        // best of 3
        for (int j=0;j<3;j++){

          create_random_matrix(D,D_nelem,0,10);
          create_random_matrix(E,E_nelem,0,10);

          copy_host_to_device_double(D,dev_D,D_nelem);
          copy_host_to_device_double(E,dev_E,E_nelem);


          start=seconds();
          mat_mul_coa_onDev<double>(dev_D, dev_E, dimsD[0],dimsE[1],dimsD[1],dev_F);
          t=seconds()-start;
          t2=(t<t2) ? t : t2 ;


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
          mat_mul_tr_onDev<double>(dev_D, dev_E, NORMAL,NORMAL,dimsD[0],dimsD[1],dimsE[0],dimsE[1],dev_F);
          t=seconds()-start;
          t7=(t<t7) ? t : t7 ;

          if (i<=9){
            start=seconds();
            mat_mul_cpu<double>(D, E, dimsD[0],dimsE[1],dimsD[1],F);
            t=seconds()-start;
            t8=(t<t8) ? t : t8 ;
          }

      }

      // print to file
      printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",N,t1,t2,t3,t4,t5,t6,t7,t8,t9);
      fprintf(fp2,"%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",N,t1,t2,t3,t4,t5,t6,t7,t8,t9);
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
