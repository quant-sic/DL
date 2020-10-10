/*
  LINEAR LAYER CLASS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   : 1. change random number generation on device? for weights?
  CAUTION :
*/

#ifndef _LINEAR_H_
#define _LINEAR_H_

// c++ standard headers
#include <random>

// own c++ headers
#include "layers.h"

// own c / cu headers
#include "linear_propagation.h"
#include "test_matrix_operator.h"

//_________________________________________________________________________________________________
// class for linear layer
class linear : public layer
{
private :
  // matrices
  matrix w;
  matrix b;

  matrix z;
  matrix a;
  matrix da;

  // helper array -> no dynamic allocation for weights and bias update
  matrix helper_storage;

  // neurons
  size_t _neurons_in;
  size_t _neurons_out;
  
  // member fucntions for initalization and propagation
  void init_bias_zeros    (void);
  void init_weights_random(void);

  // propagation functions
  void calc_store_backprop_error(matrix& dz, bool flag_host);
  void calc_store_layer_output  (matrix& a,  bool flag_host);
  void update_weights           (matrix& dz, double learning_rate, bool flag_host);
  void update_bias              (matrix& dz, double learning_rate, bool flag_host);

  // random number generator
  std::default_random_engine gen;
  
public :
  // constructor / destructor
  linear(std::string name, size_t neurons_in, size_t neurons_out);
  ~linear(void);

  // forwward and backward propagation
  matrix& prop_forward (matrix& a, bool flag_host);
  matrix& prop_backward(matrix& dz, double learning_rate = 0.01, bool flag_host = true);

  // layer dimensions
  int neurons_in (void) { return _neurons_in; };
  int neurons_out(void) { return _neurons_out; };

  // getter for bias / weigths
  matrix get_weights_matrix(void) const { return w; };
  matrix get_bias_matrix   (void) const { return b; };

  // setter functions
  void set_weights_matrix(matrix _w, bool flag_host);
  void set_bias_matrix   (matrix _b, bool flag_host);

  // operator overloading
  void print_out(std::ostream& out) const { out << "linear"; };
};

#endif // _LINEAR_H_
