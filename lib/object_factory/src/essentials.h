/*
  MODULE FOR SIMPLE USEFUL FUNCTIONS THAT CAN BE NEEDED MORE OFTEN
  AUTHOR : JANNIS SCHÜRMANN
  DATE   : 22.03.2020
  
  TO-DO : 1. either reuse the find_in_vector function or get rid of it -> this module should be 
             kept as small as possible
*/

#ifndef __ESSENTIALS_H_
#define __ESSENTIALS_H_

// standard c++ headers
#include <array>
#include <vector>
#include <cstdio>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

// functions contained in this module
template <typename T> int  find_in_vector   (std::vector <T>& vec, const T& value);
std::string                execute_command  (const char* command);
std::size_t                get_maximum_width(const std::vector <std::string>& v_names);

//________________________________________________________________________________________________
// function to execute a system command and get the return value as a string
std::string execute_command(const char* command)
{
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command, "r"), pclose);
  if (!pipe)
    throw std::runtime_error("popen() failed!");
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    result += buffer.data();
  return result;
}

//________________________________________________________________________________________________
// find an index in a vector for a given vector and element
// returns -1 if not contained
// returns index of the first found element
template <typename T>
int find_in_vector(std::vector <T>& vec,
		   const T& value)
{
  if (vec.size() == 0)
    return -1;
  auto it = std::find(vec.begin(), vec.end(), value);
  if (it != vec.end())
    return std::distance(vec.begin(), it);
  else
    return -1;
}

//________________________________________________________________________________________________
// get the maximum width / length from a vector of strings
std::size_t get_max_width(const std::vector <std::string>& v_names)
{
  if (v_names.size() == 0)
    return 12; // because i like trouble
  size_t max = v_names[0].length();
  size_t length = 0;
  for (size_t idx = 1; idx < v_names.size(); idx++) {
    length = v_names[idx].length();
    if (length > max)
      max = length;
  }
  return max;
}

#endif
