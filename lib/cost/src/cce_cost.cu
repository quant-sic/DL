#include "cce_cost.h"

//_______________________________________________________________________________________________
// cost calculaion
double cce_cost::cost(matrix predict,
		      matrix target,
		      bool   flag_host)
{
  // batchsize is the target or predict number of rows
  if (flag_host)
    return categorical_crossentropy(predict.data_host.get(), target.data_host.get(), predict.size(), predict.rows());
  else
    return categorical_crossentropy_onDev(predict.data_device.get(), target.data_device.get(), predict.size(), predict.rows());
}

//_______________________________________________________________________________________________
// dcost calculation
matrix cce_cost::dcost(matrix predict,
		       matrix target,
		       matrix dy,
		       bool   flag_host)
{
  if (flag_host)
    d_categorical_crossentropy(predict.data_host.get(), target.data_host.get(), dy.data_host.get(), predict.size());
  else
    d_categorical_crossentropy_onDev(predict.data_device.get(), target.data_device.get(), dy.data_device.get(), predict.size());
  matrix ret = dy;
  return ret;
}
