#include "global.h"

dim3 pointwise_block(BS_1D,1);
dim3 pointwise2d_block(BS_EW_2D,BS_EW_2D);
dim3 pointwise3d_block(BS_3D,BS_3D,BS_3D);
dim3 col_red_block(BS_1D,1);
dim3 row_red_1d_block(BS_R_RED_1D,1);
dim3 row_red_2d_block(BS_R_RED_2D,1);
dim3 matrix_mul_block(BS_2D,BS_2D);


dim3 pointwise_grid(int size){
  dim3 pointwise_grid((size+(OpTh*pointwise_block.x)-1)/(OpTh*pointwise_block.x),1);
  return pointwise_grid;
}

dim3 pointwise2d_grid(int size_x,int size_y){
  dim3 pointwise2d_grid((size_x+(OpTh*pointwise2d_block.x)-1)/(OpTh*pointwise2d_block.x),(size_y+(OpTh*pointwise2d_block.y)-1)/(OpTh*pointwise2d_block.y));
  return pointwise2d_grid;
}

dim3 pointwise2d_noOpTh_grid(int size_x,int size_y){
  dim3 pointwise2d_noOpTh_grid((size_x+(pointwise2d_block.x)-1)/(pointwise2d_block.x),(size_y+(pointwise2d_block.y)-1)/(pointwise2d_block.y));
  return pointwise2d_noOpTh_grid;
}

// grid for row reduction
dim3 row_red_grid(int size_x, int size_y){
  dim3 row_red_grid(1,size_y);
  return row_red_grid;
}

dim3 col_red_grid(int size_x, int size_y){
  dim3 col_red_grid((size_x+col_red_block.x-1)/col_red_block.x,1);
  return col_red_grid;
}

dim3 pointwise3d_grid(int size_x,int size_y,int size_z){
  dim3 pointwise3d_grid((size_x+(OpTh*pointwise3d_block.x)-1)/(OpTh*pointwise3d_block.x),(size_y+(OpTh*pointwise3d_block.y)-1)/(OpTh*pointwise3d_block.y),(size_z+(OpTh*pointwise3d_block.z)-1)/(OpTh*pointwise3d_block.z));
  return pointwise3d_grid;
}

dim3 matrix_mul_grid(int size_x,int size_y){
  dim3 matrix_mul_grid((size_x+matrix_mul_block.x-1)/matrix_mul_block.x,(size_y+matrix_mul_block.y-1)/matrix_mul_block.y);
  return matrix_mul_grid;
}


dim3 get_pointwise_block(){
  return pointwise_block;
}

dim3 get_pointwise2d_block(){
  return pointwise2d_block;
}
dim3 get_pointwise3d_block(){
  return pointwise3d_block;
}

dim3 get_row_red_1d_block(){
  return row_red_1d_block;
}

dim3 get_row_red_2d_block(){
  return row_red_2d_block;
}

dim3 get_col_red_block(){
  return col_red_block;
}

dim3 get_matrix_mul_block(){
  return matrix_mul_block;
}
