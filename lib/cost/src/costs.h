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

// own c headers
#include "costfunctions.h"

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
  virtual ~costs(void) = 0;

  // memeber functions
  virtual double cost (matrix predict, matrix target, bool flag_host) = 0;
  virtual matrix dcost(matrix predict, matrix target, matrix dy, bool flag_host) = 0;

  // operator overloading
  virtual void   print_out(std::ostream& out) const = 0;
  friend std::ostream& operator <<(std::ostream& out, const costs& c) { c.print_out(out); return out; };

  // get cost type
  int get_type(void) { return this->_type; };
};

//_______________________________________________________________________________________________
// categorical cross entropy cost
class cce_soft_cost : public costs
{
public :
  // constructor
  cce_soft_cost (void) {this->_type = CCE_SOFT;};
  ~cce_soft_cost(void) {};

  // functions to calculate
  double cost (matrix predict, matrix target, bool flag_host);
  matrix dcost(matrix predict, matrix target, matrix dy, bool flag_host);

  // operator overloading
  //friend std::ostream& operator <<(std::ostream& out, const cce_soft_cost& cce_soft);
  void print_out(std::ostream& out) const { out << "cce_soft"; };
};
#endif // _COSTS_H_
