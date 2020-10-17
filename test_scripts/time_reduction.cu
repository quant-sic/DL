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
#include "cce_cost.h"
#include "reduce.h"


int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nTimes Reductions at ");
  printf("device %d: %s \n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  printf("_________________________________________________\n");


  srand(seconds());   // Initialization, should only be called once.
  double start;

  // time pointwise_combine problem
  double t,t1,t2,t3,t4;
  int size;
  double *res1,*lhs,*rhs;
  double *dev_res1,*dev_lhs,*dev_rhs;
  double *dev_mat_in,*dev_vec_out;
  double *mat_in,*vec_out;


  // cost reduction using categorical categorical_crossentropy
  FILE *fp_rowred = fopen("analysis/rowredcost.txt", "w");
  fprintf(fp_rowred,"N\tCostTIME_onDEV\tCostTIME_HOST\n");

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

    t1=t2=t3=t4=DBL_MAX;
    for (int i=0;i<5;i++){
      start=seconds();
      cce_onDev<double>(dev_lhs,dev_rhs,size);
      t=seconds()-start;
      t1=t<t1?t:t1;

      start=seconds();
      cce<double>(lhs,rhs,size);
      t=seconds()-start;
      t2=t<t2?t:t2;

    }
    fprintf(fp_rowred,"%d\t%e\t%e\n",size,t1,t2);
}

  // matrix row reduction
  FILE *fp_rowred_mr = fopen("analysis/rowredmr.txt", "w");
  fprintf(fp_rowred_mr,"N_ROWS\tN_COLS\tMRTIME_onDEV\tMRTIME_HOST\tCT_DtH_M\tCT_HtD_V\n");


  for(int cols=1;cols<=(1<<13);cols<<=1){
    for(int rows=1;rows<=(1<<13);rows<<=1){

      t1=t2=t3=t4=DBL_MAX;

      int size_mat_in=cols*rows;
      mat_in=(double *)malloc(size_mat_in*sizeof(double));
      vec_out=(double *)malloc(rows*sizeof(double));

      for(int i =0;i<size_mat_in;i++) mat_in[i]=((double)rand()/(double)RAND_MAX);

      CHECK(cudaMalloc((void**)&dev_mat_in, size_mat_in*sizeof(double)));
      CHECK(cudaMalloc((void**)&dev_vec_out, rows*sizeof(double)));

      copy_host_to_device_double(mat_in,dev_mat_in,size_mat_in);

      for (int i=0;i<5;i++){
        start=seconds();
        add_reduce_dim_onDev(dev_mat_in,dev_vec_out,rows,cols,1,rows);
        t=seconds()-start;
        t1=t<t1?t:t1;

        start=seconds();
        add_reduce_dim_cpu(mat_in,vec_out,rows,cols,1,rows);
        t=seconds()-start;
        t2=t<t2?t:t2;

        start=seconds();
        copy_device_to_host_double(dev_mat_in,mat_in,size_mat_in);
        t=seconds()-start;
        t3=t<t3?t:t3;

        start=seconds();
        copy_host_to_device_double(vec_out,dev_vec_out,rows);
        t=seconds()-start;
        t4=t<t4?t:t4;

      }
      fprintf(fp_rowred_mr,"%d\t%d\t%e\t%e\t%e\t%e\n",rows,cols,t1,t2,t3,t4);

      CHECK(cudaFree(dev_mat_in));
      CHECK(cudaFree(dev_vec_out));
      free(mat_in);
      free(vec_out);
    }
  }
  fclose (fp_rowred_mr);



  // scale analysis matrix row reduction
  FILE *fp_rowred_mr_an = fopen("analysis/rowredmr_analysis.txt", "w");
  fprintf(fp_rowred_mr_an,"N_ROWS\tN_COLS\tN_Threads\tMRTIME_onDEV_T1\tMRTIME_HOST\tMRTIME_onDEV_Tp\n");


  for(int cols=(1<<8);cols<=(1<<13);cols<<=1){
    for(int rows=(1<<8);rows<=(1<<13);rows<<=1){

      t1=t2=DBL_MAX;

      int size_mat_in=cols*rows;
      mat_in=(double *)malloc(size_mat_in*sizeof(double));
      vec_out=(double *)malloc(rows*sizeof(double));

      for(int i =0;i<size_mat_in;i++) mat_in[i]=((double)rand()/(double)RAND_MAX);

      CHECK(cudaMalloc((void**)&dev_mat_in, size_mat_in*sizeof(double)));
      CHECK(cudaMalloc((void**)&dev_vec_out, rows*sizeof(double)));

      copy_host_to_device_double(mat_in,dev_mat_in,size_mat_in);

      for (int i=0;i<3;i++){

        start=seconds();
        CHECK(cudaMemset(dev_vec_out, 0, rows*sizeof(double)));
        dim3 grid(1,rows);
        add_reduce_rows_kernel<<<grid,1>>>(dev_mat_in, dev_vec_out,rows,cols);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        t=seconds()-start;
        t1=t<t1?t:t1;

        start=seconds();
        add_reduce_dim_cpu(mat_in,vec_out,rows,cols,1,rows);
        t=seconds()-start;
        t2=t<t2?t:t2;
      }



      for(int n_threads=1;n_threads<=(1<<11);n_threads<<=1){

        t3=DBL_MAX;
        int threads_block=(n_threads>BS_R_RED_2D ? BS_R_RED_2D :n_threads);
        dim3 n_blocks(1,rows);

        for (int i=0;i<3;i++){

          start=seconds();
          CHECK(cudaMemset(dev_vec_out, 0, rows*sizeof(double)));
          add_reduce_rows_kernel<<<n_blocks,threads_block>>>(dev_mat_in, dev_vec_out,rows,cols);
          CHECK(cudaDeviceSynchronize());
          CHECK(cudaGetLastError());
          t=seconds()-start;
          t3=t<t3?t:t3;

        }
        fprintf(fp_rowred_mr_an,"%d\t%d\t%d\t%e\t%e\t%e\t%e\n",rows,cols,threads_block,t1,t2,t3);
      }
      CHECK(cudaFree(dev_mat_in));
      CHECK(cudaFree(dev_vec_out));
      free(mat_in);
      free(vec_out);
    }
  }
  fclose (fp_rowred_mr_an);




  // matric col reduction
  FILE *fp_colred_mr = fopen("analysis/colredmr.txt", "w");
  fprintf(fp_colred_mr,"N_ROWS\tN_COLS\tMRTIME_onDEV\tMRTIME_HOST\tCT_DtH_M\tCT_HtD_V\n");


  for(int cols=1;cols<=(1<<13);cols<<=1){
    for(int rows=1;rows<=(1<<13);rows<<=1){

    t1=t2=t3=t4=DBL_MAX;

    int size_mat_in=cols*rows;
    mat_in=(double *)malloc(size_mat_in*sizeof(double));
    vec_out=(double *)malloc(cols*sizeof(double));

    for(int i =0;i<size_mat_in;i++) mat_in[i]=((double)rand()/(double)RAND_MAX);

    CHECK(cudaMalloc((void**)&dev_mat_in, size_mat_in*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_vec_out, cols*sizeof(double)));

    copy_host_to_device_double(mat_in,dev_mat_in,size_mat_in);

    for (int i=0;i<5;i++){
      start=seconds();
      add_reduce_dim_onDev(dev_mat_in,dev_vec_out,rows,cols,0,cols);
      t=seconds()-start;
      t1=t<t1?t:t1;

      start=seconds();
      add_reduce_dim_cpu(mat_in,vec_out,rows,cols,0,cols);
      t=seconds()-start;
      t2=t<t2?t:t2;

      start=seconds();
      copy_device_to_host_double(dev_mat_in,mat_in,size_mat_in);
      t=seconds()-start;
      t3=t<t3?t:t3;

      start=seconds();
      copy_host_to_device_double(vec_out,dev_vec_out,cols);
      t=seconds()-start;
      t4=t<t4?t:t4;

    }
    fprintf(fp_colred_mr,"%d\t%d\t%e\t%e\t%e\t%e\n",rows,cols,t1,t2,t3,t4);

    CHECK(cudaFree(dev_mat_in));
    CHECK(cudaFree(dev_vec_out));
    free(mat_in);
    free(vec_out);
  }
  }
  fclose (fp_colred_mr);




  CHECK(cudaFree(dev_res1));
  CHECK(cudaFree(dev_lhs));
  CHECK(cudaFree(dev_rhs));
  free(res1);
  free(lhs);
  free(rhs);
}
