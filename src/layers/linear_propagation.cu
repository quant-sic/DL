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
#include "../common.h"
#include "../matrix_operations/matrix_operator.h"
#include "linear_propagation.h"

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
  matMul(a, w, batchsize, neurons_out, neurons_in, z);
  add_along_col_direct_cpu(z, b, batchsize, neurons_out);
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
  matrix_transpose_cpu(wT, w, neurons_in, neurons_out);
  matMul(dz, wT, batchsize, neurons_in, neurons_out, da);
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
  matrix_transpose_cpu(aT, a, batchsize, neurons_in);

  // temporary matrix dw -> try to avoid
  double* dw = (double*) malloc(neurons_in*neurons_out*sizeof(double));
  matMul(aT, dz, neurons_in, neurons_out, batchsize, dw);

  // multiplying dw by scalar and adding to dw
  mulAdd_direct_cpu(w, dw, -(double)learning_rate/(double)batchsize, neurons_in*neurons_out);

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
  add_reduce_dim_cpu(dz, dz_summed_and_scaled, batchsize, neurons_out, 0, neurons_out);
  mulAdd_direct_cpu(b, dz_summed_and_scaled, -(double)learning_rate/(double)batchsize, neurons_out);

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
  matMul_sm_onDev_tr(a, w, 0, 0, batchsize, neurons_in, neurons_in, neurons_out, z);
  add_along_col_direct_onDev(z, b, batchsize, neurons_out);
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
  matMul_sm_onDev_tr(dz, w, 0, 1, batchsize, neurons_out, neurons_out, neurons_in, da);
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
  matMul_sm_onDev_tr(a, dz, 1, 0, neurons_in, batchsize, batchsize, neurons_out, helper_storage);
  mulAdd_direct_onDev(w, helper_storage, -(double)learning_rate/(double)batchsize, neurons_in*neurons_out, 64);
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
  add_reduce_dim_onDev(dz, helper_storage, batchsize, neurons_out, 0, neurons_out);
  mulAdd_direct_onDev(b, helper_storage,-(double)learning_rate/(double)batchsize, neurons_out, 64);
}
