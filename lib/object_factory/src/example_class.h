/*
  module for the example class to use with the object factory

  author  : archer
  date    :
  to-do   :
  caution :

  program : just a simple class with different types and numbers of data members.
*/

#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

// c++ standard headers
#include <string>
#include <ostream>
#include <iomanip>

// c++ own headers

//_______________________________________________________________________________________________
// example class
class example
{
private :
  // string variables
  std::string _name = "archer", _type = "agent";
  
  // double values
  double _min = -12., _max = 12., _mean = 0.;

  // int values
  int _size = 24;
  
public :
  // constructor / destructor
  example(void) {};
  ~example(void) {};

  // setters for all the different members
  void set_name(std::string name) { _name = name; };
  void set_type(std::string type) { _type = type; };
  void set_min (double min)       { _min  = min;  };
  void set_max (double max)       { _max  = max;  };
  void set_mean(double mean)      { _mean = mean; };
  void set_size(int size)         { _size = size; };
  
  // ostream operator
  friend std::ostream& operator <<(std::ostream& out, const example& e) {
    out << "name : " << std::setw(12) << e._name << " | "
	<< "type : " << std::setw(12) << e._type << " | "
	<< "min : "  << std::setw(12) << e._min  << " | "
	<< "max : "  << std::setw(12) << e._max  << " | "
	<< "mean : " << std::setw(12) << e._mean << " | "
	<< "size : " << std::setw(12) << e._size;
    return out;
  };
};

#endif // _EXAMPLE_H_
