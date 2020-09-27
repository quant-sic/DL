/*
  TEST THE LAYERS FOR HOST AND DEVICE
  
  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    : 
  TO-DO   :
  CAUTION :
*/

// c++ standard headers
#include <iostream>

// own cpp headers
#include "test_layers.h"
#include "../layers/layers.h"
#include "../layers/sigmoid.h"
#include "../layers/softmax.h"
#include "../layers/relu.h"
#include "../layers/linear.h"

int main(int agrc, char** argv) {

  // loop over all the layers
  int rows = 5;
  int cols = 3;
  double min = -1.;
  double max = +1.;
  double learn_rate = 0.1;
  
  //_____________________________________________________________________________
  // RELU
  // identity relu -> result should be the same
  relu relu_layer("relu");
  matrix mat_iden = create_i_matrix(rows);
  matrix mat_exp = mat_iden;

  // forward
  test_layer_in_out_forward_backward(relu_layer, mat_iden, mat_exp, learn_rate, true,  true);
  test_layer_in_out_forward_backward(relu_layer, mat_iden, mat_exp, learn_rate, false, true);
  
  // backwrd
  test_layer_in_out_forward_backward(relu_layer, mat_iden, mat_exp, learn_rate, true,  false);
  test_layer_in_out_forward_backward(relu_layer, mat_iden, mat_exp, learn_rate, false, false);
  
  // complete negative matrix -> should be zero
  matrix mat_neg  = create_one_matrix(rows, cols, -1.);
  matrix mat_zero = create_one_matrix(rows, cols, 0.);

  // forward
  test_layer_in_out_forward_backward(relu_layer, mat_neg, mat_zero, learn_rate, true,  true);
  test_layer_in_out_forward_backward(relu_layer, mat_neg, mat_zero, learn_rate, false, true);

  // backward
  test_layer_in_out_forward_backward(relu_layer, mat_neg, mat_zero, learn_rate, true,  false);
  test_layer_in_out_forward_backward(relu_layer, mat_neg, mat_zero, learn_rate, false, false);

  // random matrix all negative become zero else one or x
  matrix mat_ran = create_random_matrix(rows, cols, min, max);
  matrix mat_res_for (rows, cols);
  matrix mat_res_back(rows, cols);
  mat_res_for.alloc();
  mat_res_back.alloc();

  // calculating the expected results
  for (int idx = 0; idx < mat_ran.size(); idx++)
    if (mat_ran[idx] <= 0.)
      mat_res_for[idx] = 0.;
    else
      mat_res_for[idx] = mat_ran[idx];

  // forward
  test_layer_in_out_forward_backward(relu_layer, mat_ran, mat_res_for, learn_rate, true,  true);
  test_layer_in_out_forward_backward(relu_layer, mat_ran, mat_res_for, learn_rate, false, true);

  // backward
  test_layer_in_out_forward_backward(relu_layer, mat_ran, mat_res_for, learn_rate, true,  false);
  test_layer_in_out_forward_backward(relu_layer, mat_ran, mat_res_for, learn_rate, false, false);

  //_________________________________________
  // LINEAR
  // w (neurons_in x neurons_out)
  // b (neruons_out x 1)

  int neurons_in  = 5;
  int neurons_out = 5;
  linear lin_layer("linear", neurons_in, neurons_out);
  matrix mat_ran_lin = create_random_matrix(neurons_in, neurons_out, min, max);
  matrix mat_weights = create_i_matrix(neurons_out);
  lin_layer.set_weights_matrix(mat_weights, false);

  // linear forward
  // w = identity
  // b = 0's
  // -> prop_forward shoudlnt chnage anything (symmetric case)
  test_layer_in_out_forward_backward(lin_layer, mat_ran_lin, mat_ran_lin, learn_rate, true, true);
  mat_weights = lin_layer.get_weights_matrix();
  matrix mat_bias = lin_layer.get_bias_matrix();
  
  test_layer_in_out_forward_backward(lin_layer, mat_ran_lin, mat_ran_lin, learn_rate, false, true);
  mat_weights = lin_layer.get_weights_matrix();
  mat_bias = lin_layer.get_bias_matrix();

  // linear backward
  // do the matrices change? bias and weights?
  test_layer_in_out_forward_backward(lin_layer, mat_ran_lin, mat_ran_lin, learn_rate, true, false);
  test_layer_in_out_forward_backward(lin_layer, mat_ran_lin, mat_ran_lin, learn_rate, false, false);
}
