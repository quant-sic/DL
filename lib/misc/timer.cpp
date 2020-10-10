/*
  MODULE FOR TIMING CPU AND REAL TIMES

  AUTHOR  : JANNIS SCHÜRMANN
  DATE    : 20.03.2020
  TO-DO   : 1. time_last_to_end -> wtf is this? like, WTF?
  CAUTION : 
*/

#include "timer.h"

//____________________________________________________________________________________________
// get maximum width from timer
int timer::get_maximum_width(void) const
{
  if (v_section.size() == 0)
    return 12;
  int max = v_section[0].length();
  for (int i = 1; i < v_section.size(); i++)
    if (v_section[i].length() > max)
      max = v_section[i].length();
  return max;
}

//____________________________________________________________________________________________
// output stream operator
std::ostream& operator <<(std::ostream& os,
			  const timer&  t)
{
  // check for section sizes
  int width = t.get_maximum_width();
  if ((t.v_time.size() != 0) and
      (t.v_time.size() == t.v_section.size())) {
    os << std::setw(width) << "section" << " |" << std::setw(width) << "time" << std::endl;
    for (int i = 0; i < 2*width+2; i++)
      os << "-";
    os << "\n";
    
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
// get_last_to_end
double timer::time_last_to_end(void)
{
  return std::chrono::duration_cast<std::chrono::seconds>(t_end - t_last).count();
}

//____________________________________________________________________________________________
// add a segment to the timer
void timer::section(std::string section)
{
  if (t_end != t_max) {
    t_last = t_end;
  } else {
    t_last = t_start;
  }
  t_end = std::chrono::system_clock::now();
  v_time.push_back(time_last_to_end());
  v_section.push_back(section);
}

//____________________________________________________________________________________________
// start the timer
void timer::start(void)
{
  t_start = std::chrono::system_clock::now();
  t_last  = t_start;
}

//____________________________________________________________________________________________
// stop the timer
void timer::stop(void)
{
  t_end = std::chrono::system_clock::now();
}

//____________________________________________________________________________________________
// stop the timer
void timer::stop(std::string section)
{
  t_end = std::chrono::system_clock::now();
  v_time.push_back((double)s());
  v_section.push_back(section);
}

/*
//____________________________________________________________________________________________
// get the duration from the timer
double timer::h(void)
{
t_dur = t_end - t_start;
return std::chrono::duration_cast<std::ratio<3600>>(t_dur).count();
}


//____________________________________________________________________________________________
// get the duration from the timer
double timer::min(void)
{
t_dur = t_end - t_start;
return std::chrono::duration_cast<std::ratio<60>>(t_dur).count();
}
*/

//____________________________________________________________________________________________
// get the duration from the timer
double timer::s(void)
{
  t_dur = t_end - t_start;
  return std::chrono::duration_cast<std::chrono::seconds>(t_dur).count();
}

//____________________________________________________________________________________________
// get the duration from the timer
double timer::ms(void)
{
  t_dur = t_end - t_start;
  return std::chrono::duration_cast<std::chrono::milliseconds>(t_dur).count();
}

//____________________________________________________________________________________________
// get the duration from the timer
double timer::us(void)
{
  t_dur = t_end - t_start;
  return std::chrono::duration_cast<std::chrono::microseconds>(t_dur).count();
}

//____________________________________________________________________________________________
// get the duration from the timer
double timer::ns(void)
{
  t_dur = t_end - t_start;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t_dur).count();
}
