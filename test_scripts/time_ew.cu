// standard c headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cblas.h>
#include <float.h>

// own c headers
#include "common.h"
#include "global.h"
#include "matrix_operator.h"
#include "matrix_operator_gpu.h"
#include "kernel_utils.h"
#include "activations.h"
#include "costfunctions.h"


int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Testing Tensor Contraction/ MatrixMultiplication at ");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("Performs the following checks:\n\n - matMul on HOST and matMul_gpu1, matMul_gpu2 matMul_gpu_dsm, matMul_gpu_dsm_coa, matMul_cublas on Device\n - HOST and DEVICE same result> All yield same result\n\n_________________________________________________\n");


  srand(seconds());   // Initialization, should only be called once.
  double start;

  // time pointwise_combine problem
  double t,t1,t2,t3;
  int size;
  double *res1,*res2,*lhs,*rhs;
  double *dev_res1,*dev_lhs,*dev_rhs;

  FILE *fp_c = fopen("analysis/copying.txt", "w");
  fprintf(fp_c,"N\tTimeHtD\tTimeDtH\n");

  for(size=1;size<=(1<<22);size<<=1){

    lhs=(double *)malloc(size*sizeof(double));
    for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);

    CHECK(cudaMalloc((void**)&dev_lhs, size*sizeof(double)));

    t1=t2=DBL_MAX;

    for(int i=0;i<5;i++){
      start=seconds();
      copy_host_to_device_double(lhs,dev_lhs,size);
      t=seconds()-start;
      t1=t<t1?t:t1;
    }
    for(int i=0;i<5;i++){
      start=seconds();
      copy_device_to_host_double(dev_lhs,lhs,size);
      t=seconds()-start;
      t2=t<t2?t:t2;
    }
    fprintf(fp_c,"%d\t%e\t%e\n",size,t1,t2);

  }
  fclose (fp_c);





  // 1.Difference between using __device__ function and normal

  size=1<<20;
  res1=(double *)malloc(size*sizeof(double));
  res2=(double *)malloc(size*sizeof(double));
  lhs=(double *)malloc(size*sizeof(double));
  rhs=(double *)malloc(size*sizeof(double));
  for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);
  for(int i =0;i<size;i++) rhs[i]=(5.0*(double)rand()/(double)RAND_MAX);


  FILE *fp_pw = fopen("analysis/pointwise.txt", "w");
  fprintf(fp_pw,"N\tTIME_onDEV\tTIME_HOST\n");

  for(size=1;size<=(1<<22);size<<=1){

    res1=(double *)malloc(size*sizeof(double));
    lhs=(double *)malloc(size*sizeof(double));
    for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);

    CHECK(cudaMalloc((void**)&dev_res1, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_lhs, size*sizeof(double)));

    copy_host_to_device_double(lhs,dev_lhs,size);

    t1=t2=t3=DBL_MAX;
    for (int i=0;i<5;i++){
      start=seconds();
      relu_activation_gpu(dev_lhs,dev_res1,size);
      t=seconds()-start;
      t1=t<t1?t:t1;

      start=seconds();
      relu_activation_cpu(lhs,res1,size);
      t=seconds()-start;
      t2=t<t2?t:t2;

    }
    fprintf(fp_pw,"%d\t%e\t%e\n",size,t1,t2);


  }
  fclose (fp_pw);


  // pw scale analysis
  FILE *fp_pw_an = fopen("analysis/pointwise_analysis.txt", "w");
  fprintf(fp_pw_an,"N\tTIME_onDEV_T1\tTIME_HOST\tTIME_onDEV_Tp\n");

  for(size=1;size<=(1<<22);size<<=1){

    res1=(double *)malloc(size*sizeof(double));
    lhs=(double *)malloc(size*sizeof(double));
    for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);

    CHECK(cudaMalloc((void**)&dev_res1, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_lhs, size*sizeof(double)));

    copy_host_to_device_double(lhs,dev_lhs,size);

    t1=t2=t3=DBL_MAX;

    for (int i=0;i<3;i++){

      start=seconds();
      dim3 grid=pointwise_grid(size);
      apply_pointwise_kernel<<<1,1>>>(dev_lhs, dev_res1, size,RELU);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaGetLastError());
      t=seconds()-start;
      t1=t<t1?t:t1;

      start=seconds();
      relu_activation_cpu(lhs,res1,size);
      t=seconds()-start;
      t2=t<t2?t:t2;

    }
    for(int n_threads=1;n_threads<=(1<<15);n_threads*=2){

      int threads_block=(n_threads>BS_1D ? BS_1D :n_threads);
      dim3 grid((n_threads+threads_block-1)/threads_block);

      for (int i=0;i<3;i++){

        start=seconds();
        apply_pointwise_kernel<<<grid,threads_block>>>(dev_lhs, dev_res1, size,RELU);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        t=seconds()-start;
        t3=t<t3?t:t3;

      }

      fprintf(fp_pw_an,"%d\t%d\t%e\t%e\t%e\n",size,n_threads,t1,t2,t3);
    }

  }
  fclose (fp_pw_an);


  FILE *fp_cpw = fopen("analysis/comb_pointwise.txt", "w");
  fprintf(fp_cpw,"N\tOP_P_T\tT_B\tTIME_DEVICE\tTIME_HOST\tTIME_onDEV\n");

  for(size=1;size<=(1<<22);size<<=1){

    res1=(double *)malloc(size*sizeof(double));
    res2=(double *)malloc(size*sizeof(double));
    lhs=(double *)malloc(size*sizeof(double));
    rhs=(double *)malloc(size*sizeof(double));
    for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);
    for(int i =0;i<size;i++) rhs[i]=(5.0*(double)rand()/(double)RAND_MAX);

    CHECK(cudaMalloc((void**)&dev_res1, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_lhs, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_rhs, size*sizeof(double)));

    copy_host_to_device_double(lhs,dev_lhs,size);
    copy_host_to_device_double(rhs,dev_rhs,size);

    for(int op_p_th=1;op_p_th<50;op_p_th++){
      for(int threads_block=64;threads_block<=1024;threads_block*=2){
        t1=t2=t3=DBL_MAX;
        for (int i=0;i<5;i++){

          start=seconds();
          matrix_hadamard_cpu(res1,lhs,rhs,size);
          t=seconds()-start;
          t2=t<t2?t:t2;

          start=seconds();
          matrix_hadamard_onDev(dev_res1,dev_lhs,dev_rhs,size,threads_block);
          t=seconds()-start;
          t3=t<t3?t:t3;
        }
        fprintf(fp_cpw,"%d\t%d\t%d\t%e\t%e\t%e\n",size,op_p_th,threads_block,t1,t2,t3);

      }
    }
  }
  fclose (fp_cpw);

  // measure times for true efficiency true speed up, speed up, efficiency and scale up
  FILE *fp_cpw_an = fopen("analysis/comb_pointwise_analysis.txt", "w");
  fprintf(fp_cpw_an,"N\tOP_P_T\tT_B\tTIME_DEVICE\tTIME_HOST\tTIME_onDEV\n");

    // for n=size
    for(size=1<<10;size<=1<<22;size<<=3){

      res1=(double *)malloc(size*sizeof(double));
      res2=(double *)malloc(size*sizeof(double));
      lhs=(double *)malloc(size*sizeof(double));
      rhs=(double *)malloc(size*sizeof(double));
      for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);
      for(int i =0;i<size;i++) rhs[i]=(5.0*(double)rand()/(double)RAND_MAX);

      CHECK(cudaMalloc((void**)&dev_res1, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&dev_lhs, size*sizeof(double)));
      CHECK(cudaMalloc((void**)&dev_rhs, size*sizeof(double)));

      copy_host_to_device_double(lhs,dev_lhs,size);
      copy_host_to_device_double(rhs,dev_rhs,size);

      int op_p_th=1;

        // p
        for(int n_threads=1;n_threads<=(1<<15);n_threads*=2){
          t1=t2=t3=DBL_MAX;
          for (int i=0;i<5;i++){

            // T(1)
            start=seconds();
            comb_pointwise_1d_kernel<<<1, 1>>>(dev_res1, dev_lhs, dev_rhs, size,MUL);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            t=seconds()-start;
            t1=t<t1?t:t1;

            // T'
            start=seconds();
            matrix_hadamard_cpu(res1,lhs,rhs,size);
            t=seconds()-start;
            t2=t<t2?t:t2;

            // T(p)
            int threads_block=(n_threads>BS_1D ? BS_1D :n_threads);
            int n_blocks=((n_threads+threads_block-1)/threads_block);
            start=seconds();
            comb_pointwise_1d_kernel<<<n_blocks, threads_block>>>(dev_res1, dev_lhs, dev_rhs, size,MUL);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            t=seconds()-start;
            t3=t<t3?t:t3;
          }
          fprintf(fp_cpw_an,"%d\t%d\t%d\t%e\t%e\t%e\n",size,op_p_th,n_threads,t1,t2,t3);
      }
      CHECK(cudaFree(dev_res1));
      CHECK(cudaFree(dev_lhs));
      CHECK(cudaFree(dev_rhs));
      free(res1);
      free(res2);
      free(lhs);
      free(rhs);
    }
    fclose (fp_cpw_an);




}
