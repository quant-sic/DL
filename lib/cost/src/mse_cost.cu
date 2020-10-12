#include "mse_cost.h"

//_______________________________________________________________________________________________
// cost calculaion
double mse_cost::cost(matrix predict,
		      matrix target,
		      bool   flag_host)
{
  if (flag_host)
    return rms(predict.data_host.get(), target.data_host.get(), predict.size(), predict.rows());
  else
    return rms_onDev(predict.data_device.get(), target.data_device.get(), predict.size(), predict.rows());
}

//_______________________________________________________________________________________________
// dcost calculation
matrix mse_cost::dcost(matrix predict,
		       matrix target,
		       matrix dy,
		       bool   flag_host)
{
  if (flag_host)
    d_rms(predict.data_host.get(), target.data_host.get(), dy.data_host.get(), predict.size());
  else
    d_rms_onDev(predict.data_device.get(), target.data_device.get(), dy.data_device.get(), predict.size());
  matrix ret = dy;
  return ret;
}