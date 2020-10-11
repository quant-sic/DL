/*
  module for a range

  author  : archer
  date    :
  to-do   : 1. inherited version of range class for object factory class?
            2. const& value?
	    
  caution : 1. initalizing the range via an interval only works for real numerical types
  
  program : class for a range, which means, either a tuple (not std::tuple) of values or an
            interval [min, max] with N steps in between. the interval version shouldn't be
	    used with non real numerical types. The tuple version can and should be used
	    for other types as well(matrices, strings etc.), for example in combination with
	    the object factory class.
*/

#ifndef _RANGE_H_
#define _RANGE_H_

// c++ standard headers
#include <vector>
#include <cstddef>
#include <ostream>
#include <algorithm>

//________________________________________________________________________________________________
// range class
template <typename _real_type>
class range
{
private :
  std::vector <_real_type> _v_points; // tuple of points ti define the range
  _real_type _value;                  // current value used for the object factory
  
public :
  // tuple constructor
  range(std::vector <_real_type> v_points) {
    std::sort(v_points.begin(), v_points.end());
    auto last = std::unique(v_points.begin(), v_points.end());
    v_points.erase(last, v_points.end()); // too lazy too look up vector inialization
    _v_points = v_points;
  };

  // interval constructor
  range(_real_type min, _real_type max, std::size_t n = 1) : _v_points(n) {
    // 0 would trigger division by 0 -> also, 0 doesn't make sense
    if (n == 0)
      n = 1;
    
    // calculate the points in the range
    _real_type step = (max - min) / (n - 1);
    for (std::size_t idx = 0; idx < n; idx++)
      _v_points[idx] = min + idx * step;
  };

  // destructor
  ~range(void) {};
  
  // getters -> set the reference for the current value
  std::size_t n  (void)            { return _v_points.size(); };
  _real_type  nth(std::size_t idx) { _value = _v_points[idx];    return _v_points[idx];    };
  _real_type  min(void)            { _value = _v_points.front(); return _v_points.front(); };
  _real_type  max(void)            { _value = _v_points.back();  return _v_points.back();  };

  // refernce getter
  _real_type& value(void) { return _value; };
  
  // extract operator
  friend std::ostream& operator <<(std::ostream& out, const range<_real_type>& r) {
    out << "range points : ";
    for (auto val : r._v_points)
      out << val << " ";
    return out;
  }
};

#endif // _RANGE_H_
