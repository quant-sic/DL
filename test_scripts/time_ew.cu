// standard c headers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cblas.h>
#include <float.h>

#include <string>

// own c headers
#include "common.h"
#include "global.h"
#include "relu.h"

#include "common_utils.h"


int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Time EW at ");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));


  srand(seconds());   // Initialization, should only be called once.
  double start;

  // time pointwise_combine problem
  double t,t1,t2,t3;
  int size;
  double *res1,*lhs,*rhs;
  double *dev_res1,*dev_lhs,*dev_rhs;

  FILE *fp_c = fopen("/users/stud/dechentf/DL/analysis/copying.txt", "w");
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


 size=1<<20;
  res1=(double *)malloc(size*sizeof(double));
  lhs=(double *)malloc(size*sizeof(double));
  rhs=(double *)malloc(size*sizeof(double));
  for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);
  for(int i =0;i<size;i++) rhs[i]=(5.0*(double)rand()/(double)RAND_MAX);




  size=1<<20;
  res1=(double *)malloc(size*sizeof(double));
  lhs=(double *)malloc(size*sizeof(double));
  rhs=(double *)malloc(size*sizeof(double));
  for(int i =0;i<size;i++) lhs[i]=(5.0*(double)rand()/(double)RAND_MAX);
  for(int i =0;i<size;i++) rhs[i]=(5.0*(double)rand()/(double)RAND_MAX);


  FILE *fp_pw = fopen("/users/stud/dechentf/DL/analysis/pointwise.txt", "w");
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
      relu_activation_onDev<double>(dev_lhs,dev_res1,size);
      t=seconds()-start;
      t1=t<t1?t:t1;

      start=seconds();
      relu_activation_cpu<double>(lhs,res1,size);
      t=seconds()-start;
      t2=t<t2?t:t2;

    }
    fprintf(fp_pw,"%d\t%e\t%e\n",size,t1,t2);


  }
  fclose (fp_pw);





  FILE *fp_cpw = fopen("analysis/comb_pointwise.txt", "w");
  fprintf(fp_cpw,"N\tOP_P_T\tT_B\tTIME_DEVICE\tTIME_HOST\tTIME_onDEV\n");

  for(size=1;size<=(1<<22);size<<=1){

    res1=(double *)malloc(size*sizeof(double));
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
          combine_pointwise_cpu(lhs,rhs,res1,size,mul_functor<double>());
          t=seconds()-start;
          t2=t<t2?t:t2;

          start=seconds();
          combine_pointwise_onDev(dev_lhs,dev_rhs,dev_res1,size,mul_functor<double>());
          t=seconds()-start;
          t3=t<t3?t:t3;
        }
        fprintf(fp_cpw,"%d\t%d\t%d\t%e\t%e\t%e\n",size,op_p_th,threads_block,t1,t2,t3);

      }
    }
  }
  fclose (fp_cpw);




}
