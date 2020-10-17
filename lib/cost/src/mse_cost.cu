#include "mse_cost.h"

//_______________________________________________________________________________________________
// cost calculaion
double mse_cost::cost(matrix predict,
		      matrix target,
		      bool   flag_host)
{
  if (flag_host)
    return mse<double>(predict.data_host.get(), target.data_host.get(), predict.size());
  else
    return mse_onDev<double>(predict.data_device.get(), target.data_device.get(), predict.size());
}

//_______________________________________________________________________________________________
// dcost calculation
matrix mse_cost::dcost(matrix predict,
		       matrix target,
		       matrix dy,
		       bool   flag_host)
{
  if (flag_host)
    d_mse<double>(predict.data_host.get(), target.data_host.get(), dy.data_host.get(), predict.size());
  else
    d_mse_onDev<double>(predict.data_device.get(), target.data_device.get(), dy.data_device.get(), predict.size());
  matrix ret = dy;
  return ret;
}