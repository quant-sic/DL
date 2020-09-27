/*
  NEURAL NETWORK LAYER BASE CLASS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   :
  CAUTION :
*/

#ifndef _LAYERS_H_
#define _LAYERS_H_

#include <string>
#include "../neural_network/matrix.h"
#include "activations.h"

//enum Layers
//_________________________________________________________________________________________________
enum layer_names{LINEAR,RELU,SIGMOID,SOFTMAX};

//_________________________________________________________________________________________________
// base class for the different layers
class layer
{
protected :
  std::string _name;
  int _type;

public :
  // destructor
  virtual ~layer(void) = 0;

  // forward backward prop
  virtual matrix& prop_forward (matrix& A, bool flag_host) = 0;
  virtual matrix& prop_backward(matrix& dZ, double learning_rate, bool flag_host) = 0;

  // get the name of the related layer
  std::string get_name(void) { return this->_name; };

  // operator overloading
  virtual void print_out(std::ostream& out) const = 0;
  friend std::ostream& operator <<(std::ostream& out, const layer& l) { l.print_out(out); return out; };

  // get layer type
  int get_type(void) { return this->_type; };
};

#endif // _LAYERS_H_
