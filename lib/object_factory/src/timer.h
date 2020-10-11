/*
  MODULE FOR TIMING CPU AND REAL TIMES

  AUTHOR  : JANNIS SCHÜRMANN
  DATE    : 20.03.2020
  TO-DO   : 
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

// own c++ headers
#include "essentials.h"

//_______________________________________________________________________________________________
// classes in this module
class timer; // time different parts of the program and stopwatch
class time_profiler;

//_______________________________________________________________________________________________
// helper functions
std::string time_to_hh_mm_ss(int time_s);
std::pair <double, std::string> time_to_biggest_unit_from_ns(double t_ms);

//_______________________________________________________________________________________________
// timer class
class timer {
private :
  // the classical Y2262 problem
  std::chrono::time_point <std::chrono::system_clock> t_max =
    std::chrono::time_point<std::chrono::system_clock>::max();
  
  // start and end point / duration
  std::chrono::time_point <std::chrono::system_clock> t_start, t_end = t_max, t_last;
  std::chrono::duration <double> t_dur;

  // normal timer sections
  std::vector <double>      v_time;
  std::vector <std::string> v_section;
  
public :
  // constructor
  timer(void) {};
  ~timer(void) {};
  
  // times for the segment functions
  void section(std::string section);
  
  // starter and stopper
  void start(void) { t_start = std::chrono::system_clock::now(); t_last  = t_start; };
  void stop (void) { t_end = std::chrono::system_clock::now(); };
  
  // getter functions
  std::vector <double>      get_times   (void) const { return v_time; };
  std::vector <std::string> get_sections(void) const { return v_section; };
  std::chrono::time_point <std::chrono::system_clock> get_start(void) const { return t_start; };
  std::string get_date_and_time(void);
  
  // maximum width for table
  size_t get_maximum_width(void) const;
  
  // duartions in different units
  double s  (void) { return std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count(); };
  double ms (void) { return std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count(); };
  double us (void) { return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count(); };
  double ns (void) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count(); };
  
  // overlaoding operator for output
  friend std::ostream& operator <<(std::ostream& out, const timer& t);
};

//____________________________________________________________________________________________
// get date and time as a string
std::string timer::get_date_and_time(void)
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
  return ss.str();
}

//____________________________________________________________________________________________
// add a segment to the timer
void timer::section(std::string section)
{
  if (t_end != t_max)
    t_last = t_end;
  else
    t_last = t_start;
  t_end = std::chrono::system_clock::now();
  v_time.push_back(std::chrono::duration_cast<std::chrono::seconds>(t_end - t_last).count());
  v_section.push_back(section);
}

//____________________________________________________________________________________________
// output stream operator
std::ostream& operator <<(std::ostream& os,
			  const timer&  t)
{
  // check for section sizes
  size_t width = get_max_width(t.get_sections());
  if ((t.v_time.size() != 0) and
      (t.v_time.size() == t.v_section.size())) {
    os << std::setw(width) << "section" << " |" << std::setw(width) << "time" << std::endl;
    os << "---------------------------------------------" << std::endl;
    
    // write out the sections data
    std::vector <double>      v_tim = t.get_times();
    std::vector <std::string> v_sec = t.get_sections();
    for (size_t i = 0; i < v_tim.size(); i++) {
      os << std::setw(width) << v_sec[i] << " |"
	 << std::setw(width) << time_to_hh_mm_ss(v_tim[i]) << std::endl;
    }
    
    // output and calculation complete time
    std::chrono::time_point <std::chrono::system_clock> t_end = std::chrono::system_clock::now();
    std::chrono::duration <double> t_dur = t_end - t.get_start();
    os << std::setw(width) << "complete time" << " |"
       << std::setw(width) << time_to_hh_mm_ss(std::chrono::duration_cast<std::chrono::seconds>(t_dur).count())
       << std::endl;
    return os;
  }
  return os;
}

//____________________________________________________________________________________________
// time conversion to string for output
std::string time_to_hh_mm_ss(int time_s)
{
  int time_hh = time_s/3600;
  int time_mm = (time_s - time_hh*3600) / 60;
  int time_ss = time_s - time_hh*3600 - time_mm*60;
  std::string time_hh_mm_ss = std::to_string(time_hh) + "h";
  if (time_mm < 10) time_hh_mm_ss += "0";
  time_hh_mm_ss += std::to_string(time_mm) + "m";
  if (time_ss < 10) time_hh_mm_ss += "0";
  time_hh_mm_ss += std::to_string(time_ss) + "s";
  return time_hh_mm_ss;
}

//____________________________________________________________________________________________
// convert a time in ms to the biggest time unit and return the string
std::pair <double, std::string> time_to_biggest_unit_from_ns(double t_ns)
{
  // convert to s if needed
  double t = t_ns;
  std::string unit = "ns";
  if (t >= 1000.) {
    t /= 1000.;
    unit = "us";
  }
  if (t >= 1000.) {
    t /= 1000.;
    unit = "ms";
  }
  if (t >= 1000.) {
    t /= 1000.;
    unit = "s";
  }
  if (t >= 60.) {
    t /= 60.;
    unit = "m";
  }
  if (t >= 60.) {
    t /= 60.;
    unit = "h";
  }
  
  return std::make_pair(t, unit);
}

//____________________________________________________________________________________________
// time profiler class to time functions
// starting a function creates a starting point for that function
// stopping a function calculates the duration and adds it to the section
// unordered_map is probably faster for this, but who cares
class time_profiler {
protected :
  // vectors to store fucntions and times
  std::vector <std::string> _v_func;
  std::vector <double>      _v_time;
  std::vector <std::chrono::time_point <std::chrono::system_clock>> _v_start;
  
public : 
  // constructor / destructor
  time_profiler(void) {};
  ~time_profiler(void) {};

  // start and stop a function
  void start_function(std::string f_name);
  void stop_function(std::string f_name);
  
  // extract operator
  friend std::ostream& operator <<(std::ostream& out, const time_profiler& t_prof);
};

//____________________________________________________________________________________________
// start a section -> add a starting point and a section of not found
void time_profiler::start_function(std::string f_name)
{
  int idx = find_in_vector(_v_func, f_name); 
  if (idx == -1) {
    // not found -> add section and starting point
    _v_func.push_back (f_name);
    _v_start.push_back(std::chrono::system_clock::now());
    _v_time.push_back(0.0);
  } else {
    // found -> set the correct starting point
    _v_start[idx] = std::chrono::system_clock::now();
  }
}

//____________________________________________________________________________________________
// stop a section
void time_profiler::stop_function(std::string f_name)
{
  int idx = find_in_vector(_v_func, f_name);
  if (idx == -1)
    std::cout << "time profiler cant stop function. No function with name " << f_name << std::endl;
  else {
    _v_time[idx] += std::chrono::duration_cast<std::chrono::nanoseconds>
      (std::chrono::system_clock::now() - _v_start[idx]).count();
  }
}

//____________________________________________________________________________________________
// extract operator for time profiler
std::ostream& operator <<(std::ostream&        out,
			  const time_profiler& t_prof)
{
  if (t_prof._v_func.size() == t_prof._v_time.size()) {
    // print header data
    size_t width = get_max_width(t_prof._v_func);
    out << std::setw(width) << "function" << " | time taken\n";
    for (size_t idx = 0; idx < width + 17; idx++)
      out << "-";
    out << "\n";
    
    // print table data
    double t_func;
    std::string u_func;
    for (size_t idx = 0; idx < t_prof._v_func.size(); idx++) {
      out << std::setw(width) << t_prof._v_func[idx] << " | ";
      std::tie(t_func, u_func) = time_to_biggest_unit_from_ns(t_prof._v_time[idx]);
      out << std::setw(12) << t_func << " " << u_func << "\n";
    }
  } else {
    std::cout << __FUNCTION__ << std::endl;
    std::cout << "Cannot output time profiler. Vectors dont have the same sizes." << std::endl;
  }
  return out;
}
#endif // __TIMER_H_
