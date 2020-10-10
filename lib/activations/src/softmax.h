/*
  SOFTMAX LAYER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 26.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "layers.h"

//_________________________________________________________________________________________________
// class for the softmax layer
class softmax : public layer
{
private :
  matrix a;
  matrix z;
  matrix dz;

public :
  // constructor / destructor
  softmax(std::string name) { this->_name = name; 	this->_type = SOFTMAX;};
  ~softmax(void) {};

  // back and forward propagation
  matrix& prop_forward (matrix& z, bool flag_host = true);
  matrix& prop_backward(matrix& da, double learning_rate = 0.01, bool flag_host = true);

  // operator overloading
  void print_out(std::ostream& out) const { out << "softmax"; };
};

#endif // _SOFTMAX_H_
