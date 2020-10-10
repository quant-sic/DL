#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
/*
#ifdef MAIN_PROGRAM
   #define EXTERN
#else
   #define EXTERN extern
#endif
*/
#ifdef __cplusplus
extern "C" {
#endif

  // --------------------------------------------------------------------------------
  // Device Kernel Activations and derivatives
  __global__ void relu_activation_kernel     (const double *dev_in, double *dev_out,  int number);
  __global__ void d_relu_activation_kernel   (const double *dev_in, double *dev_delta,int number);
  __global__ void sigmoid_activation_kernel  (const double *dev_in, double *dev_out,  int number);
  __global__ void d_sigmoid_activation_kernel(const double *dev_in, double *dev_delta,int number);
  __global__ void relu_activation_backprop_kernel(const double *dev_da,double *dev_z,double *dev_dz,int number);
  __global__ void sigmoid_activation_backprop_kernel(const double *dev_da,double *dev_z,double *dev_dz,int number);

  // --------------------------------------------------------------------------------
  // Host Activations and derivatives
  void softmax_activation_cpu(const double *in,double *out,int batchsize,int neurons_out);
  void d_softmax_activation_cpu(const double *in, double *delta, int batchsize,int neurons_out);
  void relu_activation_cpu     (const double *in, double *out,  int number);
  void sigmoid_activation_cpu  (const double *in, double *out,  int number);
  void d_relu_activation_cpu   (const double *in, double *delta,int number);
  void d_sigmoid_activation_cpu(const double *in, double *delta,int number);
  void relu_activation_backprop_cpu(const double *da,double *z,double *dz,int number);
  void sigmoid_activation_backprop_cpu(const double *da,double *z,double *dz,int number);
  void softmax_activation_backprop_cpu(const double *da,double *z,double *dz,int neurons_out,int batchsize);
  void d_softmax_activation_cpu(const double *in, double *delta, int batchsize,int neurons_out);


  // Device Activations Derivatives and Backprop
  void relu_activation_gpu(const double *dev_in, double *dev_out, int number);
  void d_relu_activation_gpu(const double *dev_in, double *dev_delta, int number);
  void relu_activation_backprop_gpu(const double *dev_da,double *dev_z,double *dev_dz,int number);
  void sigmoid_activation_gpu(const double *dev_in, double *dev_out, int number);
  void d_sigmoid_activation_gpu(const double *dev_in, double *dev_delta, int number);
  void sigmoid_activation_backprop_gpu(const double *dev_da,double *dev_z,double *dev_dz,int number);
  void softmax_activation_onDev(const double *dev_in,double *dev_out,int batchsize,int neurons_out);
  void softmax_activation_backprop_onDev(const double *dev_da,const double *dev_z,double *dev_dz,int neurons_out,int batchsize);
  void d_softmax_activation_onDev(const double *dev_in, double *dev_delta, int batchsize,int neurons_out);

  double get_max(const double *data, int length);
  double get_max_onDev(const double *dev_in,int size);

  void softmax_activation_backprop_onDev(const double *dev_da,const double *dev_z,double *dev_dz,int neurons_out,int batchsize);

  // --------------------------------------------------------------------------------
  // Activation Function pointer
  typedef void (*act_Func) (double *,const int);

  // --------------------------------------------------------------------------------
  // Activation Struct includes Activation and its derivative
  typedef struct{
    void (*f) (double *,int);
    void (*df)(double *,int);
  } activation;


  // define algorithm thresholds
  #define SOFTMAX_THRESHOLD (1<<12)
  #define SOFTMAX_BB_THRESHOLD (1<<13)
  #define D_SOFTMAX_THRESHOLD (1<<13)

#ifdef __cplusplus
}
#endif

#endif //ACTIVATIONS_H
