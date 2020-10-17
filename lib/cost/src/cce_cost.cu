#include "cce_cost.h"

//_______________________________________________________________________________________________
// cost calculaion
double cce_cost::cost(matrix predict,
		      matrix target,
		      bool   flag_host)
{
  // batchsize is the target or predict number of rows
  if (flag_host)
    return cce<double>(predict.data_host.get(), target.data_host.get(), predict.size());
  else
    return cce_onDev<double>(predict.data_device.get(), target.data_device.get(), predict.size());
}

//_______________________________________________________________________________________________
// dcost calculation
matrix cce_cost::dcost(matrix predict,
		       matrix target,
		       matrix dy,
		       bool   flag_host)
{
  if (flag_host)
    d_cce<double>(predict.data_host.get(), target.data_host.get(), dy.data_host.get(), predict.size());
  else
    d_cce_onDev<double>(predict.data_device.get(), target.data_device.get(), dy.data_device.get(), predict.size());
  matrix ret = dy;
  return ret;
}
