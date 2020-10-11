/*
  TEST THE LAYER IMPLEMENTATION
  
  AUTHOR  : FABIAN DECHENT / JANNIS SCHÜRMANN
  DATE    :
  TO-DO   :
  CAUTION : 
*/

#ifndef _TEST_LAYERS_H_
#define _TEST_LAYERS_H_

#include <float.h>
#include "layers.h"
#include "test_matrices_cpp.h"

//________________________________________________________________________________________________
// funtions to test the layers
template <typename _layer_type>
void test_layer_in_out_forward_backward(_layer_type layer,
					matrix      inp,
					matrix      exp_out,
					double      learning_rate,
					bool        flag_host,
					bool        flag_forward)
{
  // calculate output for the layer
  if (!flag_host) {
    inp.copy_host_to_device();
    exp_out.copy_host_to_device();
  }
  
  matrix out;
  out = layer.prop_forward(inp, flag_host);
  if (!flag_forward)
    out = layer.prop_backward(out, learning_rate, flag_host);
  
  // compare the output and expected output
  bool   accept;
  double min, max, mean_rel;
  std::tie(accept, min, max, mean_rel) = compare_matrices_cpp(out, exp_out, flag_host, DBL_EPSILON);
  
  // output für die Layer
  std::cout << "_______________________" << std::endl;
  if (flag_forward)
    std::cout << "PROP_FORWARD  : " << std::endl;
  else
    std::cout << "PROP_BACKWARD : " << std::endl;
  std::cout << "Layer        : " << layer     << std::endl;
  std::cout << "host/device  : ";
  if (flag_host) std::cout << "host" << std::endl;
  else std::cout << "device" << std::endl;
  if (accept)
    std::cout << "SUCCESS : TRUE" << std::endl;
  else {
    std::cout << "SUCCESS : FALSE" << std::endl;
    std::cout << "min_err : " << min << std::endl;
    std::cout << "max_err : " << max << std::endl;
    std::cout << "rel_err : " << mean_rel << std::endl;
  }
  
  // print out matrices for debugging
  if (!accept) {
    if (!flag_host) {
      std::cout << "Input  : " << std::endl;
      output_device_matrix(inp);
      std::cout << "Output : " << std::endl;
      output_device_matrix(out);
      std::cout << "Exp. Output : " << std::endl;
      output_device_matrix(exp_out);
    } else {
      std::cout << "Input  : " << std::endl;
      output_host_matrix(inp);
      std::cout << "Output : " << std::endl;
      output_host_matrix(out);
      std::cout << "Exp. Output : " << std::endl;
      output_host_matrix(exp_out);
    }
  }
}

#endif // _TEST_LAYERS_H_
