/*
  NEURAL NETWORK CLASS

  AUTHOR  : JANNIS SCHÜRMANN
  DATE    : 18.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

// c++ standard headers
#include <vector>

// own c++ headers
#include "costs.h"
#include "layers.h"
#include "linear.h"
#include "softmax.h"
#include "sigmoid.h"
#include "relu.h"

//_______________________________________________________________________________________________
// class for the neural network
class neural_network
{
private :
  // members of the neural network
  std::vector <layer*> v_layers;
  costs* _cost;

  // matrices and learning rate
  matrix y;
  matrix dy;
  double learning_rate;

  // flag for running on host or device
  bool flag_host = true;

  // used in python flag
  bool flag_py = false;
  
public :
  // constructor / destructor
  neural_network (double learning_rate = 0.01, bool flag_host = true,bool flag_py=true);
  ~neural_network(void);

  // forward and backward propagation
  matrix prop_forward (matrix x);
  void   prop_backward(matrix predict, matrix target);

  // layer operations
  void add_layer(layer* layer) { v_layers.push_back(layer); };
  std::vector <layer*> get_layers(void) const { return v_layers; };

  // cost operations
  void set_cost(costs* cost) { _cost = cost; }
  costs* get_cost(void) const { return _cost; };

  // getter for the host flag
  bool get_flag_host(void) const { return flag_host; };
  
  // operator overloading
  friend std::ostream& operator <<(std::ostream& out, const neural_network& nn);

  // function to produce a network naming short for file names
  std::string produce_naming_short(std::vector <int>& v_neurons);
};
#endif // _NEURAL_NETWORK_H_
