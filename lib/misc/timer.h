/*
  MODULE FOR TIMING CPU AND REAL TIMES

  AUTHOR  : JANNIS SCHÜRMANN
  DATE    : 20.03.2020
  TO-DO   : 1. time_last_to_end -> wtf is this? like, WTF?
  CAUTION : 
*/

#ifndef __TIMER_H_
#define __TIMER_H_

// c++ standard headers
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>

//_______________________________________________________________________________________________
// helper functions
std::string time_to_hh_mm_ss(int time_s);

//_______________________________________________________________________________________________
// timer class
class timer {
private :
  // the classical Y2262 problem
  std::chrono::time_point <std::chrono::system_clock> t_max = std::chrono::time_point<std::chrono::system_clock>::max();

  // start and end point / duration
  std::chrono::time_point <std::chrono::system_clock> t_start, t_end = t_max, t_last;
  std::chrono::duration <double> t_dur;

  // vector to store multiple time points and their naming
  std::vector<double>      v_time;
  std::vector<std::string> v_section;

  // get_last_to_end
  double time_last_to_end(void);
  
public :
  // constructor
  timer(void) {};

  // times for the segment functions
  void section(std::string section);
  
  // starter and stopper
  void start(void);
  void stop (void);
  void stop (std::string section);
  
  // getter functions
  std::vector <double>      get_times   (void) const {return v_time;};
  std::vector <std::string> get_sections(void) const {return v_section;};
  std::chrono::time_point <std::chrono::system_clock> get_start(void) const {return t_start;};

  // maximum width for table
  int get_maximum_width(void) const;
  
  // duartions in different units
  double s  (void);
  double ms (void);
  double us (void);
  double ns (void);
  //  double min(void);
  // double h  (void);
    
  // overlaoding operator for output
  friend std::ostream& operator <<(std::ostream& out, const timer& t);
};
#endif // __TIMER_H_
