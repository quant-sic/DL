/*
  SOFTMAX LAYER

  AUTHOR  : FABIAN DECHANT / JANNIS SCHÜRMANN
  DATE    : 26.08.2020
  TO-DO   : 
  CAUTION :
*/

#include "softmax.h"
#include "activations.h"

//_________________________________________________________________________________________________
// forward propagation
matrix& softmax::prop_forward(matrix& z,
			      bool    flag_host)
{
  this->z = z;
  a.alloc_if_not_allocated(z.rows(), z.cols());
  if (flag_host)
    softmax_activation_cpu(z.data_host.get(), a.data_host.get(), z.rows(), z.cols());
  else
    softmax_activation_onDev(z.data_device.get(), a.data_device.get(), z.rows(), z.cols());
  return a;
}

//_________________________________________________________________________________________________
// backward propagation
matrix& softmax::prop_backward(matrix& da,
			       double  learning_rate,
			       bool    flag_host)
{
  dz.alloc_if_not_allocated(z.rows(), z.cols());
  if (flag_host)
    softmax_activation_backprop_cpu(da.data_host.get(), z.data_host.get(),dz.data_host.get(), z.rows(), z.cols());
  else
    softmax_activation_backprop_onDev(da.data_device.get(), z.data_device.get(), dz.data_device.get(), z.rows(), z.cols());
  return dz;
}
