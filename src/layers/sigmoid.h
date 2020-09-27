/*
  CLASS FILE FOR THE SIGMOID

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include "layers.h"

//_________________________________________________________________________________________________
// class for the sigmoid layer
class sigmoid : public layer
{
private :
  matrix a;
  matrix z;
  matrix dz;

public :
  // constructor / destructor
  sigmoid (std::string name) { this->_name = name; 	this->_type = SIGMOID;};
  ~sigmoid(void) {};

  // back and forward propagation
  matrix& prop_forward (matrix& z, bool flag_host = true);
  matrix& prop_backward(matrix& da, double learning_rate = 0.01, bool flag_host = true);

  // operator overloading
  void print_out(std::ostream& out) const { out << "sigmoid"; };
};

#endif // _SIGMOID_H_
