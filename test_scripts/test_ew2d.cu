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
#include "test_matrix_operator.h"
#include "reduce.h"
#include "pw2d_comp.h"
#include "common_utils.h"

// define thresholds
#define ADD_REDUCE_DIM_THRESHOLD(dim_red,cols,rows) (10000*sqrt((dim_red ? cols:rows))*DBL_EPSILON)
#define ADD_ALONG_AXIS_THRESHOLD (sqrt(2)*DBL_EPSILON)
#define DIV_ALONG_AXIS_THRESHOLD (sqrt(2)*DBL_EPSILON)


int main(int argc, char **argv)
{

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nTesting EW2D at ");
  printf("device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));


  srand(seconds());   // Initialization, should only be called once.



  printf("\n\nPerforms the following checks:\n\n - ADD_ALONG_AXIS, ADD_REDUCE_DIM on HOST and Device\n - HOST and DEVICE yield the same result\n - ADD_ALONG_AXIS and 0 are consistent\n\n_________________________________________________\n");

  double *vec_out,*vec_out_tmp,*mat_in;
  int size_vec;
  double *mat_out,*mat_out_tmp,*vec;


  for(int rows=1;rows<=1<<10;rows*=2){
    for(int cols=1;cols<=1<<10;cols*=2){

        int size_mat_in=cols*rows;
        mat_in=(double *)malloc(size_mat_in*sizeof(double));


        for(int dim_red =0;dim_red<2;dim_red++){
          create_random_matrix(mat_in,size_mat_in,-1,1);

          size_vec=(dim_red ? rows:cols);

          vec_out=(double *)malloc(size_vec*sizeof(double));
          vec_out_tmp=(double *)malloc(size_vec*sizeof(double));

          add_reduce_dim_cpu<double>(mat_in,vec_out,rows,cols,dim_red,size_vec);
          add_reduce_dim_onDev<double>(mat_in,vec_out_tmp,rows,cols,dim_red,size_vec);


          // // check for equal result
          if (!double_equal(vec_out,vec_out_tmp,size_vec,ADD_REDUCE_DIM_THRESHOLD(dim_red,cols,rows))){
             printf("For rows:%d,cols:%d ADD_REDUCE_DIM not same result on Host and Device\n",rows,cols);
             printf("Threshold: %e; Max diff: %e\n\n",ADD_REDUCE_DIM_THRESHOLD(dim_red,cols,rows), max_abs_diff(vec_out,vec_out_tmp,size_vec));
          }
          free(vec_out);
          free(vec_out_tmp);
        }

        for(int dim_add=0;dim_add<2;dim_add++){

          // check if gpu and cpu give same result
          create_random_matrix(mat_in,size_mat_in,-1,1);
          size_vec=(dim_add ? rows:cols);

          mat_out=(double *)malloc(rows*cols*sizeof(double));
          mat_out_tmp=(double *)malloc(rows*cols*sizeof(double));
          vec=(double *)malloc(size_vec*sizeof(double));

          create_random_matrix(vec,size_vec,-1,1);

          func_along_axis_cpu<double>(add_functor<double>(),mat_in,vec,mat_out, rows,cols,dim_add,size_vec);
          func_along_axis_onDev<double>(add_functor<double>(),mat_in,vec,mat_out_tmp, rows,cols,dim_add,size_vec);

          if (!double_equal(mat_out_tmp,mat_out,rows*cols,ADD_ALONG_AXIS_THRESHOLD)){
             printf("For rows:%d,cols:%d ADD_ALONG_AXIS not same result on Host and Device\n",rows,cols);
             printf("Threshold: %e; Max diff: %e\n\n",ADD_ALONG_AXIS_THRESHOLD, max_abs_diff(mat_out_tmp,mat_out,rows*cols));
          }

          // check if 0 and add are consistent
          memset(vec,0,size_vec*sizeof(double));
          func_along_axis_onDev<double>(add_functor<double>(),mat_in,vec,mat_out_tmp, rows,cols,dim_add,size_vec);

          if (!double_equal(mat_in ,mat_out_tmp,size_vec,ADD_ALONG_AXIS_THRESHOLD)){
             printf("For rows:%d,cols:%d ADD_ALONG_AXIS and 0 not compatible\n\n",rows,cols);
          }

          free(mat_out);
          free(mat_out_tmp);
          free(vec);
        }

        for(int dim_div=0;dim_div<2;dim_div++){
          create_random_matrix(mat_in,size_mat_in,-1,1);

          size_vec=(dim_div ? rows:cols);

          mat_out=(double *)malloc(rows*cols*sizeof(double));
          mat_out_tmp=(double *)malloc(rows*cols*sizeof(double));
          vec=(double *)malloc(size_vec*sizeof(double));


          create_random_matrix(vec,size_vec,1,2); // notice there should not be a zero

          func_along_axis_onDev<double>(div_functor<double>(),mat_in,vec,mat_out, rows,cols,dim_div,size_vec);
          func_along_axis_onDev<double>(div_functor<double>(),mat_in,vec,mat_out_tmp, rows,cols,dim_div,size_vec);

          // // check for equal result
          if (!double_equal(mat_out_tmp,mat_out,rows*cols,DIV_ALONG_AXIS_THRESHOLD)){
             printf("For rows:%d,cols:%d DIV_ALONG_AXIS not same result on Host and Device\n",rows,cols);
             printf("Threshold: %e; Max diff: %e\n\n",DIV_ALONG_AXIS_THRESHOLD, max_abs_diff(mat_out_tmp,mat_out,rows*cols));
          }

          free(mat_out);
          free(mat_out_tmp);
          free(vec);
        }

        free(mat_in);
    }
  }

  printf("Checks done\n" );
  return EXIT_SUCCESS;
}
