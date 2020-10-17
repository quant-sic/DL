/*
  CLASS FILE FOR COSTS

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 20.08.2020
  TO-DO   : 1. costs -> parent class -> inherit for different cost types
            2.
  CAUTION :
*/

#ifndef _COSTS_H_
#define _COSTS_H_

// own c++ headers
#include "matrix.h"

// cost type enum
enum cost_names {MSE, CCE, CCE_SOFT};

//_______________________________________________________________________________________________
// class for the bce_cost
class costs
{
protected:
  int _type;

public :
  // destructor
  virtual ~costs(void) {};

  // memeber functions
  virtual double cost (matrix predict, matrix target, bool flag_host) = 0;
  virtual matrix dcost(matrix predict, matrix target, matrix dy, bool flag_host) = 0;

  // operator overloading
  virtual void   print_out(std::ostream& out) const = 0;
  friend std::ostream& operator <<(std::ostream& out, const costs& c) { c.print_out(out); return out; };

  // get cost type
  int get_type(void) { return this->_type; };
};

#endif // _COSTS_H_
