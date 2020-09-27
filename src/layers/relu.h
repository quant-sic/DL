/*
  CLASS FOR THE RELU ACTIVATION LAYER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _RELU_H_
#define _RELU_H_

#include "layers.h"

//_________________________________________________________________________________________________
// class for the relu layer
class relu : public layer
{
private :
  matrix a;
  matrix z;
  matrix dz;

public :
  // contructor destructor
  relu (std::string name) { this->_name = name; 	this->_type = RELU;};
  ~relu(void) {};
  
  // back and forward propagation
  matrix& prop_forward (matrix& z, bool flag_host = true);
  matrix& prop_backward(matrix& da, double learning_rate = 0.01, bool flag_host = true);

  // operatro overloading
  void print_out(std::ostream& out) const { out << "relu"; };
};

#endif // _RELU_H_
