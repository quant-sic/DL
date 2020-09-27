#ifndef GLOBAL_H
#define GLOBAL_H

#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
   Achtung: BS_1D muss vielfaches von 2 sein!
*/

// define operations to be done by one thread
#define OpTh 10

// define Blocksize for 1D pointwise problems
#define BS_1D 512

// define Blocksize for row reduction problem
#define BS_R_RED_1D 512
#define BS_R_RED_2D 128


// define blocksize for 2D pointwise and Matrix Multiplication problem
// for matrix multiplication it can't be greater than 32
#define BS_2D 32
#define BS_EW_2D 32

// defines Blocksize for 3D elementwise problem
#define BS_3D 10

// declaration of grid functions
dim3 pointwise_grid(int size);
dim3 pointwise2d_grid(int size_x,int size_y);
dim3 pointwise3d_grid(int size_x,int size_y,int size_z);
dim3 matrix_mul_grid(int size_x,int size_y);
dim3 pointwise2d_noOpTh_grid(int size_x,int size_y);

// grid for row reduction
dim3 row_red_grid(int size_x, int size_y);
dim3 col_red_grid(int size_x, int size_y);

// declation for block getter functions
dim3 get_pointwise_block();
dim3 get_pointwise2d_block();
dim3 get_pointwise3d_block();
dim3 get_col_red_block();
dim3 get_row_red_1d_block();
dim3 get_row_red_2d_block();
dim3 get_matrix_mul_block();




#ifdef __cplusplus
}
#endif

#endif
