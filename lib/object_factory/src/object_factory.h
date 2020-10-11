/*
  module for an object factory

  author  : archer
  date    :
  to-do   : 1. push_back in recursive template complicated

  caution : 1. a template object must be given
            2. value that is referenced with std::bind should be the .value() from the corresponding range
	    3. only works for single valued setter functions from the object class

  program : class for an object factory, which produces via recursion all possible objects.
            the class needs an objects instance and for every changed parameter a member function
	    which changes said parameter and a range of variables. range meaning here an object from
	    the range.h module. An example on how to use this class is given in
	    test_object_factory.cpp.
*/

#ifndef _OBJECT_FACTORY_H_
#define _OBJECT_FACTORY_H_

// c++ standard headers
#include <iostream>
#include <functional>

// c++ own headers
#include "range.h"

//_______________________________________________________________________________________________
// raw recursion for the parameters
template <typename _obj_type, typename _para_type>
void recursion_para(std::vector <_obj_type>&                       v_obj,
		    std::vector <std::function<void(_obj_type&)>>& v_func,
		    std::vector <range<_para_type>>&               v_range,
		    _obj_type&                                     tmpl_obj,
		    std::size_t                                    depth = 0)
{
  if (depth == v_func.size())
    v_obj.push_back(tmpl_obj);
  else {
    for (std::size_t idx = 0; idx < v_range[depth].n(); idx++) {
      v_range[depth].nth(idx);
      v_func[depth](tmpl_obj);
      recursion_para(v_obj, v_func, v_range, tmpl_obj, depth+1);
    }
  }
}

//_______________________________________________________________________________________________
// function to produce the vector of objects with changed parameters
template <typename _obj_type, typename _para_type>//, typename _para_type>
std::vector <_obj_type> produce_objects(_obj_type&                                                                                 tmpl_obj,
					std::pair<std::vector<std::function<void(_obj_type&)>>&, std::vector<range<_para_type>>&>& v_f_r)
{
  std::vector <_obj_type> v_obj;
  recursion_para(v_obj, v_f_r.first, v_f_r.second, tmpl_obj);
  return v_obj;
}


//_______________________________________________________________________________________________
// i mean, lets do this bullshit
template <typename _obj_type, typename _para_type, typename... _para_args>
std::vector <_obj_type> produce_objects(_obj_type& tmpl_obj,
					std::pair<std::vector<std::function<void(_obj_type&)>>&, std::vector<range<_para_type>>&>& v_f_r,
					std::pair<std::vector<std::function<void(_obj_type&)>>&, std::vector<range<_para_args>>&>&... para_args)
{
  // get all objects from lower template
  std::vector <_obj_type> v_obj = produce_objects<_obj_type, _para_args...>(tmpl_obj, para_args...);

  // expand all the objects obtained from lower template
  std::vector <std::vector <_obj_type>> vv_obj(v_obj.size());
  for (std::size_t idx = 0; idx < v_obj.size(); idx++)
    recursion_para(vv_obj[idx], v_f_r.first, v_f_r.second, v_obj[idx]);

  // make one vector again
  std::vector <_obj_type> v_return;
  for (std::size_t idx = 0; idx < vv_obj.size(); idx++)
    for (std::size_t ele = 0; ele < vv_obj[idx].size(); ele++)
      v_return.push_back(vv_obj[idx][ele]);
  
  return v_return;
}

#endif // _OBJECT_FACTORY_H_
