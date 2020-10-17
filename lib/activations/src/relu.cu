/*
  CLASS FOR THE RELU ACTIVATION LAYER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 19.08.2020
  TO-DO   :
  CAUTION : RELU ACTIVATION SEEMS KINDA FISHY
*/

#include "relu.h"

//_________________________________________________________________________________________________
// forward propagation
matrix& relu::prop_forward(matrix& z,
			   bool    flag_host)
{
  this->z = z;
  a.alloc_if_not_allocated(z.rows(), z.cols());
  if (flag_host)
    relu_activation_cpu<double>(z.data_host.get(), a.data_host.get(), z.size());
  else
    relu_activation_onDev<double>(z.data_device.get(), a.data_device.get(), z.size());
  return a;
}

//_________________________________________________________________________________________________
// backward propagation
matrix& relu::prop_backward(matrix& da,
			    double  learning_rate,
			    bool    flag_host)
{
  dz.alloc_if_not_allocated(z.rows(), z.cols());
  if (flag_host)
    relu_activation_backprop_cpu<double>(da.data_host.get(), z.data_host.get(), dz.data_host.get(), z.size());
  else 
    relu_activation_backprop_onDev<double>(da.data_device.get(), z.data_device.get(), dz.data_device.get(), z.size());
  return dz;
}
