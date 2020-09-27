/*
  LINEAR LAYER PROPAGATION HEADER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 20.08.2020
  TO-DO   : 
  CAUTION :
*/

#ifndef _LINEAR_PROPAGATION_H_
#define _LINEAR_PROPAGATION_H_

#ifdef __cplusplus
extern "C" {
#endif
  
  //_________________________________________________________________________________________________
  // cpu functions
  void linear_forward_cpu       (double* w,  double* a,  double* z, double* b, int w_rows, int w_cols, int a_rows, int a_cols);
  void linear_backprop_cpu      (double* w,  double* dz, double* da, int w_rows, int w_cols, int dz_rows, int dz_cols);
  void linear_update_weights_cpu(double* dz, double* a,  double* w, int dz_rows, int dz_cols, int a_rows, int a_cols, double learning_rate);
  void linear_update_bias_cpu   (double* dz, double* b,  int dz_rows, int dz_cols, double learning_rate);
  
  //_________________________________________________________________________________________________
  // gpu functions
  void linear_forward_gpu       (double* w,  double* a,  double* z, double* b, int w_rows, int w_cols, int a_rows, int a_cols);
  void linear_backprop_gpu      (double* w,  double* dz, double* da, int w_rows, int w_cols, int dz_rows, int dz_cols);
  void linear_update_weights_gpu(double* dz, double* a,  double* w, int dz_rows, int dz_cols, int a_cols, double learning_rate, double* helper_storage);
  void linear_update_bias_gpu   (double* dz, double* b,  int dz_rows, int dz_cols, double learning_rate, double* helper_storage);
  
#ifdef __cplusplus
}
#endif

#endif // _LINEAR_PROPAGATION_H_
