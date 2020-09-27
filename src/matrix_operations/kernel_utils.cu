#include "../global.h"
#include "kernel_utils.h"
#include <assert.h>

// __device__ __host__ functions
// ____________________________________________________________________________________________________________________________________
// ____________________________________________________________________________________________________________________________________
// algebraic
__device__ __host__ double add_func(double a,double b){
  return a+b;
}

__device__ __host__ double div_func(double a,double b){
  return a/b;
}

// multiplies two double values
__device__ __host__ double mul_func(double a,double b){
  return a*b;
}

__host__ __device__ double mulAdd(double lhs, double rhs,double factor){
  return lhs+factor*rhs;
}


// activations
__device__ __host__ double sigmoid(double x){
  return 1/(1+exp(-x));
}

__device__ __host__ double d_sigmoid(double x){
  return exp(-x)/((1+exp(-x))*(1+exp(-x)));
}

__device__ __host__ double relu(double x){
  return (x>0 ? x : 0);
}

__device__ __host__ double d_relu(double x){
  return (x>0 ? 1 : 0);
}

__device__ __host__ double exp_max(double in,double max){
  return exp(in-max);
}
__device__ __host__ double comp_max(double x,double y){
  return (x>y?x:y);
}

// Indexing
__forceinline__ __host__ __device__ double get_element(const double *mat,const int row_stride,const int col_stride,const int o_row,const int o_col){
  return mat[o_row*row_stride+o_col];
}
__forceinline__ __host__ __device__ double get_element_transposed(const double *mat,const int row_stride,const int col_stride,const int o_row,const int o_col){
  return mat[o_col*row_stride+o_row];
}


// costfunctions
__host__ __device__ double d_cat_cross_ent(double in,double target){
    if(!target){
      return 0;
    }else{
      return -target / in;
    }
}

__host__ __device__ double d_rms_func(double in,double target){
  return in - target;
}

__host__ __device__ double d_cce_softmax(double in,double target){
  return in - target;
}

__host__ __device__ double summand_cat_cross_ent(double in,double target){
    if(!target){
      return 0;
    }else{
      return -target * log(in);
    }
}
__host__ __device__ double summand_rms(double in,double target){
  return 0.5*(in - target)*(in - target);
}


// Arrays that contain the __device__ functions to be used in the kernels
__device__ get_element_func dev_get_element_func[2]={&get_element,&get_element_transposed};
__device__ comb_pointwise dev_comb_pointwise_func[10]={&add_func,&div_func,&mul_func,&exp_max,&comp_max,&d_cat_cross_ent,&d_rms_func,&summand_cat_cross_ent,&summand_rms,&d_cce_softmax};
__device__ pointwise dev_pointwise_func[4]={&relu,&sigmoid,&d_relu,&d_sigmoid};



// kernels
// ____________________________________________________________________________________________________________________________________
// ____________________________________________________________________________________________________________________________________
// Pointwise manipulation and combination kernels


// combines two arrays pontwise
__global__ void comb_pointwise_1d_kernel(double* res,const double* lhs,const double* rhs,int     size,int func_int){
    // get desired function
    comb_pointwise func= dev_comb_pointwise_func[func_int];
    // traverse array elements
    for (int idx=blockIdx.x*blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x*gridDim.x){
        // apply function
        res[idx]=func(lhs[idx],rhs[idx]);
    }
}

// applies a pointwise function and stores the result in another array
__global__ void apply_pointwise_kernel(const double *dev_in, double *dev_out, int size,int func_int){

  // get pointwise function
  pointwise func=dev_pointwise_func[func_int];

  // apply fuction pointwise
  for (int idx=blockIdx.x*blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x*gridDim.x){
    dev_out[idx] = func(dev_in[idx]);
  }
}

// as the softmax function takes a constant as the element to be combined with, there needs to be a special kernel
// besides that a modified combine pointwise kernel
__global__ void combine_pointwise_kernel_actsoft(const double* dev_in,const double dev_max,double* dev_out,int size){


    for (int idx=blockIdx.x*blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x*gridDim.x)
      dev_out[idx] = exp(dev_in[idx]-dev_max);
}

// Hadamrd kernel without the device function -> serves as comparison
__global__ void matrix_hadamard_kernel(double* res,const double* lhs,const double* rhs,int     size){

    for (int idx=blockIdx.x*blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x*gridDim.x){
      res[idx]=lhs[idx]*rhs[idx];
    }
}

// Hadamard kernel with an extra function applied to the rhs
// essentially combine pointwise
__global__ void hadamard_func_kernel(const double *dev_da,double *dev_z,double *dev_dz,int size,int func_int){

  pointwise func=dev_pointwise_func[func_int];
  int idx=blockIdx.x*blockDim.x + threadIdx.x;

  for (; (idx < size); idx += blockDim.x*gridDim.x)
    dev_dz[idx] = dev_da[idx]*func(dev_z[idx]);
}

// Multiply add kernel
// essentially combine pointwise
__global__ void mulAdd_kernel(double *dev_res,const double* dev_lhs,const double* dev_rhs,const double factor,int size)
{
    for (int idx=blockIdx.x*blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x*gridDim.x){
        dev_res[idx] = mulAdd(dev_lhs[idx],dev_rhs[idx],factor);
    }
}

// mull add direct kernel -> no in between result (lhs is the result)
__global__ void mulAdd_direct_kernel(double* lhs,const double* rhs,const double factor,int size)
{
  for (int idx=blockIdx.x*blockDim.x + threadIdx.x; (idx < size); idx += blockDim.x*gridDim.x){
    lhs[idx] += rhs[idx]*factor;
  }
}









// ____________________________________________________________________________________________________________________________________
// Reduction kernels


// Row reduction kernel
// kernel assumes 1 block assigned per row, use block-striding methodology
// assumes blocksize is power of 2
__global__ void add_reduce_rows_kernel(const double * __restrict__  dev_mat_in, double * __restrict__  dev_vec_out, const int rows, const int cols) {

  __shared__ double tmp[BS_R_RED_2D];

  tmp[threadIdx.x] = 0;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) // block-stride
    tmp[threadIdx.x] += dev_mat_in[(blockIdx.y * cols) + i];
  __syncthreads();

  for (int i = blockDim.x>>1; i > 0; i>>=1){
    if (threadIdx.x < i) tmp[threadIdx.x] += tmp[threadIdx.x+i];
    __syncthreads();
  }

  if (!threadIdx.x) dev_vec_out[blockIdx.y] = tmp[0];
}



// coll reduction kernel
// kernel assumes one thread assigned per column sum
// launch number of columns threads
__global__ void add_reduce_cols_kernel(const double* dev_mat_in,double *dev_vec_out, int rows,int cols){

    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if ( idx < cols){
      // add onto intermediate result
      double tmp = 0;
      for (int j = 0; j < rows; j++) tmp += dev_mat_in[(j*cols)+idx];

      // save in output vector
      dev_vec_out[idx] = tmp;
    }
}

// kernel that gets the maximum value for every row
__global__ void get_max_kernel(const double * __restrict__ data, int length,double *dev_res){

    __shared__ double tmp[BS_R_RED_1D];

    tmp[threadIdx.x] = 0;

    for (int i = threadIdx.x; i < length; i += blockDim.x) // block-stride
      tmp[threadIdx.x] = (data[i]>tmp[threadIdx.x] ? data[i]:tmp[threadIdx.x]);
    __syncthreads();

    for (int i = blockDim.x>>1; i > 0; i>>=1){
      if (threadIdx.x < i) tmp[threadIdx.x] = (tmp[threadIdx.x+i]>tmp[threadIdx.x]?tmp[threadIdx.x+i]:tmp[threadIdx.x]);
      __syncthreads();
    }

    if (!threadIdx.x) *dev_res = tmp[0];
}

// Row reduction kernel
// kernel assumes 1 block assigned per row, use block-striding methodology
// assumes blocksize is power of 2
// Also there is an elementwise combination included
__global__ void add_reduce_rows_func_kernel(const double * __restrict__  dev_in, const double * __restrict__  dev_target, double* dev_res, const int size,int func_int) {

  // get combine pointwise funtion
  comb_pointwise func = dev_comb_pointwise_func[func_int];

  // set up shared memory
  __shared__ double tmp[BS_R_RED_1D];

  tmp[threadIdx.x] = 0;

  // apply function and sum into shared memory
  for (int i = threadIdx.x; i < size; i += blockDim.x) // block-stride
    tmp[threadIdx.x] += func(dev_in[i],dev_target[i]);
  __syncthreads();

  // reduce in place in shared memory
  for (int i = blockDim.x>>1; i > 0; i>>=1){
    if (threadIdx.x < i) tmp[threadIdx.x] += tmp[threadIdx.x+i];
    __syncthreads();
  }
  // assign result
  if (!threadIdx.x) *dev_res = tmp[0];
}




// ____________________________________________________________________________________________________________________________________
// 2D elementwise combination kernels

// takes a vector and a matrix and combines them along x axis. Dimensions of course have to match -> see onDev function
// x axis are the columns
__global__ void func_along_axis_x_kernel(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols,int func_int){

  // get desired function;
  comb_pointwise func= dev_comb_pointwise_func[func_int];

  // traverse 2D elements
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < cols); idx += blockDim.x*gridDim.x){
    for (int idy = blockIdx.y * blockDim.y + threadIdx.y; (idy < rows); idy += blockDim.y*gridDim.y){
         // apply elementwise cmbination
         dev_mat_out[idy*cols+idx]=func(dev_mat_in[idy*cols+idx],dev_vec[idy]);
    }
  }
}

// takes a vector and a matrix and combines them along y axis. Dimensions of course have to match -> see onDev function
// y axis are the rows
__global__ void func_along_axis_y_kernel(const double* dev_mat_in,const double *dev_vec,double* dev_mat_out, int rows,int cols,int func_int){

 comb_pointwise func= dev_comb_pointwise_func[func_int];

 for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < cols); idx += blockDim.x*gridDim.x){
   for (int idy = blockIdx.y * blockDim.y + threadIdx.y; (idy < rows); idy += blockDim.y*gridDim.y){
     dev_mat_out[idy*cols+idx]=func(dev_mat_in[idy*cols+idx],dev_vec[idx]);
   }
 }
}


// kernel for adding along col direct
__global__ void add_along_col_direct_kernel(double*       dev_mat,const double* dev_vec,int           rows,int           cols)
{
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; (idx < cols); idx += blockDim.x*gridDim.x)
    for (int idy = blockIdx.y * blockDim.y + threadIdx.y; (idy < rows); idy += blockDim.y*gridDim.y)
      dev_mat[idy*cols+idx] += dev_vec[idx];
}



// kernel that transposes a matrix
__global__ void mat_transpose_kernel(const double *mat_in, double *mat_out, int rows, int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}



// ____________________________________________________________________________________________________________________________________
// Matrix Multiplication kernels

// naive Matrix Multiplication kernel
__global__ void matMul_kernel1(const double *d_A,const double *d_B,int M, int N,int K,double *d_C){

  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

  if(i<M && j<N){

    double sum=0;

    for(int ki=0; ki < K;ki++){

      sum+=d_A[i*K+ki]*d_B[ki*N+j];

    }

    d_C[i*N+j]=sum;
  }
}

// naive Matrix Multiplication kernel with coalesced global memory access.
// A needs to be given transposed
__global__ void matMul_kernel2(const double *d_A_T,const double *d_B,int M, int N,int K,double *d_C){

  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int i = blockIdx.y*blockDim.y+threadIdx.y;

  if(i<M && j<N){

    double sum=0;

    for(int ki=0; ki < K;ki++){

      sum+=d_A_T[ki*M+i]*d_B[ki*N+j];

    }

    d_C[i*N+j]=sum;
  }
}

// Tiled Matrix Multiplication with dynamic shared memory
__global__ void matMul_kernel_dsm(const double *d_A,const double *d_B,int M, int N,int K,double *d_C){

  double CValue = 0;

  // should be the same
  assert(blockDim.x==blockDim.y);
  const int block_dim=blockDim.x;

  int row = blockIdx.y*block_dim + threadIdx.y;
  int col = blockIdx.x*block_dim + threadIdx.x;

  // set up shared memory
  extern __shared__ double Sm[];
  double *As=Sm;
  double *Bs=(double *)&Sm[block_dim*block_dim];

  for (int kk = 0; kk < (block_dim + K - 1)/block_dim; kk++) {

       // load Tiles into shred memory
       if (kk*block_dim + threadIdx.x < K && row < M)
           As[threadIdx.y*block_dim+threadIdx.x] = d_A[row*K + kk*block_dim + threadIdx.x];
       else
           As[threadIdx.y*block_dim+threadIdx.x] = 0.0;

       if (kk*block_dim + threadIdx.y < K && col < N)
           Bs[threadIdx.y*block_dim+threadIdx.x] = d_B[(kk*block_dim + threadIdx.y)*N + col];
       else
           Bs[threadIdx.y*block_dim+threadIdx.x] = 0.0;

       __syncthreads();

       // act out matrix multiplication per 2 submatrices of kk
       for (int n = 0; n < block_dim; ++n) CValue += As[threadIdx.y*block_dim+n] * Bs[n*block_dim+threadIdx.x];

       // all submatrix results will have been added to CValue
       __syncthreads();
  }

  // store in result array
  if (row < M && col < N) d_C[(row*N) +col] = CValue;
}


// Tiled Matrix Multiplication with dynamic shared memory and coalesced global memory access
// d_A needs to be given transposed
__global__ void matMul_kernel_dsm_coa(const double *d_A_T,const double *d_B,int M, int N,int K,double *d_C){

	double CValue = 0;

  // should be the same
  assert(blockDim.x==blockDim.y);
  const int block_dim=blockDim.x;

  int row = blockIdx.y*block_dim + threadIdx.y;
  int col = blockIdx.x*block_dim + threadIdx.x;

  // set up shared memory
  extern __shared__ double Sm[];
  double *As=Sm;
  double *Bs=(double *)&Sm[block_dim*block_dim];


  for (int kk = 0; kk < (block_dim + K - 1)/block_dim; kk++) {

    // load Tiles into shred memory
       if (kk*block_dim + threadIdx.x < K && row < M)
           As[threadIdx.y*block_dim+threadIdx.x] = d_A_T[(kk*block_dim + threadIdx.x)*M + row];
       else
           As[threadIdx.y*block_dim+threadIdx.x] = 0.0;

       if (kk*block_dim + threadIdx.y < K && col < N)
           Bs[threadIdx.y*block_dim+threadIdx.x] = d_B[(kk*block_dim + threadIdx.y)*N + col];
       else
           Bs[threadIdx.y*block_dim+threadIdx.x] = 0.0;

       __syncthreads();

       // act out matrix multiplication per 2 submatrices of kk
       for (int n = 0; n < block_dim; ++n) CValue += As[threadIdx.y*block_dim+n] * Bs[n*block_dim+threadIdx.x];

       // all submatrix results will have been added to CValue
       __syncthreads();
  }
  // store in result array
  if (row < M && col < N) d_C[(row*N) +col] = CValue;

}


// Tiled Matrix Multiplication with static shared memory
// Blocksize and therefore shared memory size is determined by BS_2D
__global__ void matMul_kernel_sm(const double *d_A,const double *d_B,int M, int N,int K,double *d_C){

  double CValue = 0;

  int row = blockIdx.y*BS_2D + threadIdx.y;
  int col = blockIdx.x*BS_2D + threadIdx.x;

  // set up shared memory
  __shared__ double As[BS_2D][BS_2D];
  __shared__ double Bs[BS_2D][BS_2D];

  for (int kk = 0; kk < (BS_2D + K - 1)/BS_2D; kk++) {

       // load Tiles into shred memory
       if (kk*BS_2D + threadIdx.x < K && row < M)
           As[threadIdx.y][threadIdx.x] = d_A[row*K + kk*BS_2D + threadIdx.x];
       else
           As[threadIdx.y][threadIdx.x] = 0.0;

       if (kk*BS_2D + threadIdx.y < K && col < N)
           Bs[threadIdx.y][threadIdx.x] = d_B[(kk*BS_2D + threadIdx.y)*N + col];
       else
           Bs[threadIdx.y][threadIdx.x] = 0.0;

       __syncthreads();

       // act out matrix multiplication per 2 submatrices of kk
       for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

       // all submatrix results will have been added to CValue
       __syncthreads();
  }

  // store in result array
  if (row < M && col < N) d_C[(row*N) +col] = CValue;

}


// Tiled Matrix Multiplication with static shared memory
// Blocksize and therefore shared memory size is determined by BS_2D
// kernel uses indexing functions to multiply transposed matrices
__global__ void matMul_kernel_sm_tr(const double *__restrict__ d_A,const double * __restrict__ d_B,int A_TRANSP,int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,double *d_C){

  get_element_func A_func=dev_get_element_func[A_TRANSP];
  get_element_func B_func=dev_get_element_func[B_TRANSP];

  double CValue = 0;

  int row = blockIdx.y*BS_2D + threadIdx.y;
  int col = blockIdx.x*BS_2D + threadIdx.x;

  __shared__ double As[BS_2D][BS_2D];
  __shared__ double Bs[BS_2D][BS_2D];

  for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {

       if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A){
           As[threadIdx.y][threadIdx.x] = A_func(d_A,cols_A,rows_A,row,(kk*BS_2D + threadIdx.x));
        }
       else
           As[threadIdx.y][threadIdx.x] = 0.0;

       if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B){
           Bs[threadIdx.y][threadIdx.x] = B_func(d_B,cols_B,rows_B,(kk*BS_2D + threadIdx.y),col);
       }else
           Bs[threadIdx.y][threadIdx.x] = 0.0;

       __syncthreads();

       for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

       __syncthreads();
  }

  if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

}


// matmul_kernel_sm_tr ohne indexing Funktionen
__global__ void matMul_kernel_sm_tr3(const double *__restrict__ d_A,const double * __restrict__ d_B,const int A_TRANSP,const int B_TRANSP ,const int rows_op_A,const int cols_op_B,const int cols_op_A,const int rows_A,const int cols_A,const int rows_B,const int cols_B,double *d_C){

  double CValue = 0;

  int row = blockIdx.y*BS_2D + threadIdx.y;
  int col = blockIdx.x*BS_2D + threadIdx.x;

  __shared__ double As[BS_2D][BS_2D];
  __shared__ double Bs[BS_2D][BS_2D];

  if (A_TRANSP && B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[(kk*BS_2D + threadIdx.x)*cols_A+row];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[col*cols_B+(kk*BS_2D + threadIdx.y)];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

  }else if(A_TRANSP && !B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[(kk*BS_2D + threadIdx.x)*cols_A+row];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[(kk*BS_2D + threadIdx.y)*cols_B+col];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

  }else if(!A_TRANSP && !B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[row*cols_A+(kk*BS_2D + threadIdx.x)];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[(kk*BS_2D + threadIdx.y)*cols_B+col];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;

  }else if(!A_TRANSP && B_TRANSP){

    for (int kk = 0; kk < (BS_2D + cols_op_A - 1)/BS_2D; kk++) {
         if (kk*BS_2D + threadIdx.x < cols_op_A && row < rows_op_A) As[threadIdx.y][threadIdx.x] = d_A[row*cols_A+(kk*BS_2D + threadIdx.x)];
         else As[threadIdx.y][threadIdx.x] = 0.0;

         if (kk*BS_2D + threadIdx.y < cols_op_A && col < cols_op_B) Bs[threadIdx.y][threadIdx.x]=d_B[col*cols_B+(kk*BS_2D + threadIdx.y)];
         else Bs[threadIdx.y][threadIdx.x] = 0.0;
         __syncthreads();
         for (int n = 0; n < BS_2D; ++n) CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
         __syncthreads();
    }
    if (row < rows_op_A && col < cols_op_B) d_C[(row*cols_op_B) +col] = CValue;
  }

}



// ____________________________________________________________________________________________________________________________________
// Softmax kernels

// Fills in 3 Tensor that is the differential of the softmax
__global__ void d_softmax_activation_kernel(const double * __restrict__ dev_softmax, double *dev_delta, int batchsize,int neurons_out){

  // assign softmax to 3 tensor d_softmax
  for (unsigned int k=blockIdx.x*blockDim.x+threadIdx.x; (k < batchsize); k += blockDim.x*gridDim.x){
    for (  unsigned int i=blockIdx.y*blockDim.y+threadIdx.y; (i < neurons_out); i += blockDim.y*gridDim.y){
      for (    unsigned int j=blockIdx.z*blockDim.z+threadIdx.z; (j < neurons_out); j += blockDim.z*gridDim.z){
          dev_delta[k*neurons_out*neurons_out+i*neurons_out+j] = (i==j)*dev_softmax[k*neurons_out+i] -dev_softmax[k*neurons_out+i]*dev_softmax[k*neurons_out+j];
      }
    }
  }
}

__global__ void d_softmax_activation_unrolled_kernel(const double * __restrict__ dev_softmax, double *dev_delta, int batchsize,int neurons_out){

  // assign softmax to 3 tensor d_softmax
  int j,k_i;
  for (unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x; (idx < batchsize*neurons_out*neurons_out); idx += blockDim.x*gridDim.x){
      k_i=idx/neurons_out;
      j=idx%neurons_out;

      dev_delta[idx]=((k_i%neurons_out)==j)*dev_softmax[k_i] -dev_softmax[k_i]*dev_softmax[(k_i/neurons_out)*neurons_out+j];
  }
}

// acts out multiplications on submatrices of a 3 tensor dev_d_softmax with matrix dev_da
__global__ void softmax_backprop_kernel(const double * __restrict__ dev_da,const double * __restrict__ dev_d_softmax,int batchsize, int neurons_out,double *dev_dz){

  //get sample and output neuron
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int k = blockIdx.y*blockDim.y+threadIdx.y;

  const double *dev_d_softmax_b=&dev_d_softmax[k*neurons_out*neurons_out];

  if(k<batchsize && j<neurons_out){

    // sum
    double sum=0;
    for(int ki=0; ki < neurons_out;ki++){
      sum+=dev_da[k*neurons_out+ki]*dev_d_softmax_b[ki*neurons_out+j];
    }
    // store result of matrix multiplication
    dev_dz[k*neurons_out+j]=sum;
  }
}


// uses coalesced global memory access for softmax backprop kernel
__global__ void softmax_backprop_T_kernel(const double * __restrict__ dev_da_T,const double * __restrict__ dev_d_softmax,int batchsize, int neurons_out,double *dev_dz){

  //get sample and output neuron
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int k = blockIdx.y*blockDim.y+threadIdx.y;

  const double *dev_d_softmax_b=&dev_d_softmax[k*neurons_out*neurons_out];

  if(k<batchsize && j<neurons_out){

    // sum
    double sum=0;
    for(int ki=0; ki < neurons_out;ki++){
      sum+=dev_da_T[ki*batchsize+k]*dev_d_softmax_b[ki*neurons_out+j];
    }
    // store result of matrix multiplication
    dev_dz[k*neurons_out+j]=sum;
  }
}
