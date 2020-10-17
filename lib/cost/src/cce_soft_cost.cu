#include "cce_soft_cost.h"
//_______________________________________________________________________________________________
// cost calculaion
double cce_soft_cost::cost(matrix predict,
			   matrix target,
			   bool   flag_host)
{
  if (flag_host)
    return cce_soft_cpu<double>(predict.data_host.get(), target.data_host.get(), predict.size(), predict.rows());
  else
    return cce_soft_onDev<double>(predict.data_device.get(), target.data_device.get(), predict.size(), predict.rows());
}

//_______________________________________________________________________________________________
// dcost calculation
matrix cce_soft_cost::dcost(matrix predict,
			    matrix target,
			    matrix dy,
			    bool   flag_host)
{
  if (flag_host)
    d_cce_soft_cpu<double>(predict.data_host.get(), target.data_host.get(), dy.data_host.get(), predict.size(), predict.rows());
  else
    d_cce_soft_onDev<double>(predict.data_device.get(), target.data_device.get(), dy.data_device.get(), predict.size(), predict.rows());
  matrix ret = dy;
  return ret;
}