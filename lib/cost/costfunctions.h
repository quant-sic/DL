#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

  // ----------------------------------------------------------------------------------
  // Costfunctions and derivative
  double categorical_crossentropy(const double *in,double *target,int size,int batchsize);
  void   d_categorical_crossentropy(const double *in, double *target, double *delta, int size);
  double rms(const double *in,double *target,int size,int batchsize);
  void   d_rms(const double *in, double *target, double *delta, int size);
  double rms_onDev(const double *dev_in, double *dev_target, int size,int batchsize);
  void d_rms_onDev(const double *dev_in, double *dev_target, double *dev_delta, int size);
  double categorical_crossentropy_onDev(const double *dev_in, double *dev_target, int size,int batchsize);
  void d_categorical_crossentropy_onDev(const double *dev_in, double *dev_target, double *dev_delta, int size);
  double cce_softmax_cpu(const double *in,double *target,int size,int batchsize);
  void d_cce_softmax_cpu(const double *in, double *target, double *delta, int size, int batchsize);
  double cce_softmax_onDev(const double *dev_in, double *dev_target, int size, int batchsize);
  void d_cce_softmax_onDev(const double *dev_in, double *dev_target, double *dev_delta, int size, int batchsize);



#ifdef __cplusplus
}
#endif
#endif //COSTFUNCTIONS_H
