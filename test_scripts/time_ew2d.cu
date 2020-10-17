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
#include "pw2d_comp.h"
#include "common_utils.h"


int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nTiming 2D pointwise at ");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("Performs the following Timings:\n \n - Add along axis on HOST and DEVICE\n\n_________________________________________________\n");


  srand(seconds());   // Initialization, should only be called once.
  double start;

  // time pointwise_combine problem
  double t,t1,t2,t3,t4;

  double *dev_mat_in,*dev_mat_out,*dev_vec;
  double *mat_in,*mat_out,*vec;



  FILE *fp_ew2d = fopen("analysis/fp_ew2d.txt", "w");
  fprintf(fp_ew2d,"N_ROWS\tN_COLS\tMRTIME_onDEV\tMRTIME_HOST\tCT_DtH\tCT_HtD\n");



  for(int cols=1;cols<=(1<<13);cols<<=1){
    for(int rows=1;rows<=(1<<13);rows<<=1){

      t1=t2=t3=t4=DBL_MAX;

      int size_mat_in=cols*rows;
      int size_mat_out=size_mat_in;
      mat_in=(double *)malloc(size_mat_in*sizeof(double));
      mat_out=(double *)malloc(size_mat_out*sizeof(double));
      vec=(double *)malloc(cols*sizeof(double));

      for(int i =0;i<size_mat_in;i++) mat_in[i]=((double)rand()/(double)RAND_MAX);
      for(int i =0;i<cols;i++) vec[i]=((double)rand()/(double)RAND_MAX);

      CHECK(cudaMalloc((void**)&dev_mat_in, size_mat_in*sizeof(double)));
      CHECK(cudaMalloc((void**)&dev_mat_out, size_mat_out*sizeof(double)));
      CHECK(cudaMalloc((void**)&dev_vec, cols*sizeof(double)));

      copy_host_to_device_double(mat_in,dev_mat_in,size_mat_in);
      copy_host_to_device_double(vec,dev_vec,cols);

      for (int i=0;i<5;i++){
        start=seconds();
        func_along_axis_onDev<double>(add_functor<double>(),dev_mat_in,dev_vec,dev_mat_out, rows,cols, 0, cols);
        t=seconds()-start;
        t1=t<t1?t:t1;

        start=seconds();
        func_along_axis_cpu<double>(add_functor<double>(),mat_in,vec,mat_out, rows,cols, 0, cols);
        t=seconds()-start;
        t2=t<t2?t:t2;

        start=seconds();
        copy_device_to_host_double(dev_mat_in,mat_in,size_mat_in);
        copy_device_to_host_double(dev_vec,vec,cols);
        t=seconds()-start;
        t3=t<t3?t:t3;

        start=seconds();
        copy_host_to_device_double(mat_out,dev_mat_out,size_mat_out);
        t=seconds()-start;
        t4=t<t4?t:t4;

      }
      fprintf(fp_ew2d,"%d\t%d\t%e\t%e\t%e\t%e\n",rows,cols,t1,t2,t3,t4);

      CHECK(cudaFree(dev_mat_in));
      CHECK(cudaFree(dev_mat_out));
      CHECK(cudaFree(dev_vec));
      free(mat_in);
      free(mat_out);
      free(vec);
    }
  }
  fclose (fp_ew2d);

}
