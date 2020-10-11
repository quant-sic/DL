/*
  playing around with the object factory

  author  : archer
  date    :
  to-do   :
  caution :

  program : just playing around a little bit with the object factory
*/

// c++ standard headers
#include <iostream>

// own headers
#include "timer.h"
#include "example_class.h"
#include "object_factory.h"

// redundant comment about a main function
int main(int argc, char** argv) {
  // vector of member functions and ranges
  range <std::string> r_name(std::vector <std::string> {"krieger", "barry", "pam", "lana", "archer", "ray"});
  range <std::string> r_type(std::vector <std::string> {"agent", "doctor", "enemy", "chef", "assassin"});
  range <double>      r_min (std::vector <double> {0.1, 0.7, 1.2, 1.5, 1.8, 2.0});
  range <double>      r_max (std::vector <double> {1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9});
  range <double>      r_mean(-1.0, 1.0, 10);
    
  // functions for the string variables
  std::vector <range<std::string>> v_range_str = {r_name, r_type};
  std::vector <std::function<void(example&)>> v_func_string = {
    std::bind(&example::set_name, std::placeholders::_1, std::ref(v_range_str[0].value())),
    std::bind(&example::set_type, std::placeholders::_1, std::ref(v_range_str[1].value()))
  };
  std::pair <std::vector <std::function<void(example&)>>&, std::vector <range<std::string>>&> v_fr_string = std::make_pair(std::ref(v_func_string), std::ref(v_range_str));
    
  // functions for the double variables
  std::vector <range<double>> v_range_dbl = {r_min, r_max, r_mean};
  std::vector <std::function<void(example&)>> v_func_double = {
    std::bind(&example::set_min,  std::placeholders::_1, std::ref(v_range_dbl[0].value())),
    std::bind(&example::set_max,  std::placeholders::_1, std::ref(v_range_dbl[1].value())),
    std::bind(&example::set_mean, std::placeholders::_1, std::ref(v_range_dbl[2].value()))
  };
  std::pair <std::vector <std::function<void(example&)>>&, std::vector <range<double>>&> v_fr_double = std::make_pair(std::ref(v_func_double), std::ref(v_range_dbl));

  // produce vector of class instances
  example ex;
  timer time_table;
  time_table.start();
  std::vector <example> v_ex = produce_objects<example, std::string, double>(ex, v_fr_string, v_fr_double);
  time_table.stop();

  std::cout << "time[ms]    : " << time_table.ms() << std::endl; 
  std::cout << "v_ex.size() : " << v_ex.size() << std::endl;
}
