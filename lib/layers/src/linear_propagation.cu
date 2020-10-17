/*
  GPU / CPU FUNCTIONS LINEAR PROPAGATION

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 20.08.20
  TO-DO   : 1. FIX FUNCTIONS !!!
            2. instead of allocating storage and allocating matrix -> matrix_mul for transposed matrices !!!
	    3. fix threads_block -> anpassen an matrix size?
  CAUTION :
*/
// c++ headers
#include <iostream>

// own c headers
#include "common.h"
#include "linear_propagation.h"
#include "mat_mul.h"
#include "reduce.h"
#include "pw2d_comp.h"
#include "common_utils.h"
#include "pw_comp.h"

//_________________________________________________________________________________________________
// linear forward propagation cpu // w(neurons_in x neurons_out); a(batchsize x neurons_in)
// z = a*w + b
void linear_forward_cpu(double* w,
			double* a,
			double* z,
			double* b,
			int     neurons_in,
			int     neurons_out,
			int     batchsize,
			int     a_cols)
{
  mat_mul_cpu<double>(a, w, batchsize, neurons_out, neurons_in, z);
  func_along_axis_cpu<double>(add_functor<double>(),z,b,z,batchsize, neurons_out,0,neurons_out);
}

//_________________________________________________________________________________________________
// linear layer backprop cpu //dz(batchsize x neurons_out)
// da = dz * w^T
void linear_backprop_cpu(double* w,
			 double* dz,
			 double* da,
			 int     neurons_in,
			 int     neurons_out,
			 int     batchsize,
			 int     dz_cols)
{
  // -> transpose the w matrix beforehand
  double* wT = (double*) malloc(neurons_in*neurons_out*sizeof(double));
  mat_transpose_cpu<double>(wT, w, neurons_in, neurons_out);
  mat_mul_cpu<double>(dz, wT, batchsize, neurons_in, neurons_out, da);
  free(wT);
}

//_________________________________________________________________________________________________
// linear update weights cpu
void linear_update_weights_cpu(double* dz,
			       double* a,
			       double* w,
			       int     batchsize,
			       int     neurons_out,
			       int     a_rows,
			       int     neurons_in,
			       double  learning_rate)
{
  // transpose the matrix a beforehand
  double* aT = (double*) malloc(batchsize*neurons_in*sizeof(double));
  mat_transpose_cpu<double>(aT, a, batchsize, neurons_in);

  // temporary matrix dw -> try to avoid
  double* dw = (double*) malloc(neurons_in*neurons_out*sizeof(double));
  mat_mul_cpu<double>(aT, dz, neurons_in, neurons_out, batchsize, dw);

  // multiplying dw by scalar and adding to w
  combine_pointwise_cpu<double>(w,dw,w,neurons_in*neurons_out,mul_add_functor<double>(-(double)learning_rate/(double)batchsize));

  // free up additional memory
  free(dw);
  free(aT);
}

//_________________________________________________________________________________________________
// linear bias cpu update
void linear_update_bias_cpu(double* dz,
			    double* b,
			    int     batchsize,
			    int     neurons_out,
			    double  learning_rate)
{
  // allocate multiplied dz -> dz is not changed
  double* dz_summed_and_scaled = (double*) malloc(neurons_out*sizeof(double));
  memset(dz_summed_and_scaled, 0, neurons_out*sizeof(double));

  // do the bias update
  add_reduce_dim_cpu<double>(dz, dz_summed_and_scaled, batchsize, neurons_out, 0, neurons_out);
  combine_pointwise_cpu<double>(b,dz_summed_and_scaled,b,neurons_out,mul_add_functor<double>(-(double)learning_rate/(double)batchsize));

  // freeing up space
  free(dz_summed_and_scaled);
}

//_________________________________________________________________________________________________
// linear forward propagation gpu
void linear_forward_gpu(double* w,
			double* a,
			double* z,
			double* b,
			int     neurons_in,
			int     neurons_out,
			int     batchsize,
			int     a_cols)
{
  // multiply w*a
  mat_mul_tr_onDev<double>(a, w, 0, 0, batchsize,neurons_in , neurons_in, neurons_out, z);
  func_along_axis_onDev<double>(add_functor<double>(),z,b,z, batchsize, neurons_out, 0,neurons_out);
}

//_________________________________________________________________________________________________
// linear layer backprop cpu
// da = w^T * dz
void linear_backprop_gpu(double* w,
			 double* dz,
			 double* da,
			 int     neurons_in,
			 int     neurons_out,
			 int     batchsize,
			 int     dz_cols)
{
  // get rid of dz_cols -> cleaner
  mat_mul_tr_onDev<double>(dz, w, 0, 1, batchsize, neurons_out, neurons_out, neurons_in, da);
}

//_________________________________________________________________________________________________
// linear update weights cpu
void linear_update_weights_gpu(double* dz,
			       double* a,
			       double* w,
			       int     batchsize,
			       int     neurons_out,
			       int     neurons_in,
			       double  learning_rate,
			       double* helper_storage)
{
	mat_mul_tr_onDev<double>(a, dz, 1, 0, neurons_in, batchsize, batchsize, neurons_out, helper_storage);
	combine_pointwise_onDev<double>(w,helper_storage,w,neurons_in*neurons_out,mul_add_functor<double>(-(double)learning_rate/(double)batchsize));
}

//_________________________________________________________________________________________________
// linear bias cpu update
void linear_update_bias_gpu(double* dz,
			    double* b,
			    int     batchsize,
			    int     neurons_out,
			    double  learning_rate,
			    double* helper_storage)
{
  	add_reduce_dim_onDev<double>(dz, helper_storage, batchsize, neurons_out, 0, neurons_out);
	combine_pointwise_onDev<double>(b,helper_storage,b,neurons_out,mul_add_functor<double>(-(double)learning_rate/(double)batchsize));
}
