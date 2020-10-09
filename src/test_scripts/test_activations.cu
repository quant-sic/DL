#include "../layers/activations.h"
#include "../cost/costfunctions.h"
#include "../matrix_operations/test_matrix_operator.h"
#include "../common.h"
#include <float.h>
#include <math.h>

#include "../layers/tanh.h"


// define thresholds
#define SOFTMAX_COMP_THRESHOLD(neurons_out) ((sqrt(2+2*neurons_out))*DBL_EPSILON)
#define D_SOFTMAX_COMP_THRESHOLD(neurons_out) (sqrt(3)*SOFTMAX_COMP_THRESHOLD(neurons_out))
#define SOFTMAX_BB_COMP_THRESHOLD(neurons_out) (100*sqrt(2*neurons_out)*D_SOFTMAX_COMP_THRESHOLD(neurons_out))
#define SUM_THRESHOLD(size) (sqrt(size)*DBL_EPSILON)
#define RELU_COMP_THRESHOLD DBL_EPSILON
#define RELU_BB_COMP_THRESHOLD DBL_EPSILON
#define SIGMOID_COMP_THRESHOLD (sqrt(2)*DBL_EPSILON)
#define D_SIGMOID_COMP_THRESHOLD (sqrt(7)*DBL_EPSILON)
#define SIGMOID_BB_COMP_THRESHOLD (sqrt(8)*DBL_EPSILON)

// checks if all array elemnts are bigger than zero
int bigger_than_zero(const double *A,int size,double threshold){

    for(int i=0;i<size;i++){
        if(A[i]<(-1*threshold)){return 0;}
    }
    return 1;
}

// checks if all array elements are smaller than 1
int smaller_than_value(const double *A,int size,double value,double threshold){

    for(int i=0;i<size;i++){
        if(A[i]>(value+threshold)){return 0;}
    }
    return 1;
}

void compare_activations(const char* comp_name,double * lhs,double *rhs,int batchsize,int neurons_out,double threshold)
{
  if(!double_equal(lhs,rhs,batchsize*neurons_out,threshold)){
    compare_matrices_error(comp_name, lhs, rhs,batchsize,neurons_out);
  }
}

void check_act_normalisation(const char* name,double *act,int batchsize,int neurons_out)
{
  int normalised=1;
  double sum=0;
  for(int k=0;k<batchsize;k++){
      sum=0;
      for(int i =0;i<neurons_out;i++) sum+=act[k*neurons_out+i];
      normalised*=fabs(sum-1)<SUM_THRESHOLD(neurons_out);
  }
  if(!normalised) printf("%s not normalised for batchsize %d and neurons_out %d\n",name, batchsize,neurons_out);
}

void check_relu_backprop(const char* name,const double *act_in,const double *dz,const double *da,int size){
  int relu_backprop_correct=1;
  for(int i =0;i<size;i++) relu_backprop_correct*=(act_in[i]>0 && dz[i]==da[i])||(act_in[i]<=0 && dz[i]==0);

  if(!relu_backprop_correct) printf("%s not correct\n",name);
}


int main(int argc, char **argv)
{
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using device %d: %s \n\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
  printf("Performs the following checks:\n\n - RELU, SIGMOID, SOFTMAX on HOST and Device\n - RELU correct result and >=0\n - SOFTMAX normalised and >=0\n - SIGMOID >=0 and <= 1/4\n - HOST and DEVICE same result\n\n_________________________________________________\n");




  double *act_in,*act_out_sig,*act_out_relu,*act_out_soft,*tmp,*da,*dz,*tmp2,*act_out_tanh;
  double *dev_act_in,*dev_act_out_sig,*dev_act_out_relu,*dev_act_out_soft,*dev_tmp,*dev_da,*dev_dz,*dev_tmp2,*dev_act_out_tanh;
  double *d_softmax,*tmp_d_softmax,*dev_d_softmax;


  for(int batchsize=1;batchsize<=1<<3;batchsize*=2){
      for(int neurons_out=1;neurons_out<=1<<3;neurons_out*=2){

    // setup
    int size=batchsize*neurons_out;

    act_in=(double *)malloc(size*sizeof(double));
    act_out_sig=(double *)malloc(size*sizeof(double));
    act_out_relu=(double *)malloc(size*sizeof(double));
    act_out_soft=(double *)malloc(size*sizeof(double));
    act_out_tanh=(double *)malloc(size*sizeof(double));
    tmp=(double *)malloc(size*sizeof(double));
    tmp2=(double *)malloc(size*sizeof(double));
    da=(double *)malloc(sizeof(double)*size);
    dz=(double *)malloc(sizeof(double)*size);


    create_random_matrix(act_in,size,-1,1);
    create_random_matrix(da,size,-1,1);

    CHECK(cudaMalloc((void**)&dev_act_in, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_act_out_sig, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_act_out_relu, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_act_out_soft, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_act_out_tanh, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_tmp, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_tmp2, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_da, size*sizeof(double)));
    CHECK(cudaMalloc((void**)&dev_dz, size*sizeof(double)));

    copy_host_to_device_double(act_in, dev_act_in, size);
    copy_host_to_device_double(da, dev_da, size);


    // tanh 
      tanh_activation_cpu<double>(act_in,act_out_tanh,size);
      tanh_activation_onDev<double>(dev_act_in,dev_act_out_tanh,size);
      copy_device_to_host_double(dev_act_out_tanh, tmp, size);
      compare_activations("TANH Result differs beyond threshold on Host and Device",tmp, act_out_tanh,batchsize,neurons_out,RELU_COMP_THRESHOLD*10);

    // _____________________________________________________________________________________________
    // softmax host
    // softmax normalisation
      softmax_activation_cpu(act_in,act_out_soft,batchsize,neurons_out);
      check_act_normalisation("SOFTMAX HOST",act_out_soft,batchsize,neurons_out);

    // on device
      softmax_activation_onDev(dev_act_in,dev_act_out_soft,batchsize,neurons_out);
      copy_device_to_host_double(dev_act_out_soft, tmp, size);
      check_act_normalisation("SOFTMAX DEVICE",tmp,batchsize,neurons_out);

      // compare device and host
      compare_activations("SOFTMAX Result differs beyond threshold on Host and Device",tmp, act_out_soft,batchsize,neurons_out,SOFTMAX_COMP_THRESHOLD(neurons_out));
      if(!bigger_than_zero(tmp,size,SOFTMAX_COMP_THRESHOLD(neurons_out)) || !bigger_than_zero(act_out_soft,size,SOFTMAX_COMP_THRESHOLD(neurons_out))) printf("SOFTMAX smaller than 0 on device or host\n" );


      // d_softmax
      d_softmax=(double *)malloc(batchsize*neurons_out*neurons_out*sizeof(double));
      tmp_d_softmax=(double *)malloc(batchsize*neurons_out*neurons_out*sizeof(double));
      CHECK(cudaMalloc((void**)&dev_d_softmax, batchsize*neurons_out*neurons_out*sizeof(double)));

      d_softmax_activation_cpu(act_in, d_softmax, batchsize,neurons_out);
      d_softmax_activation_onDev(dev_act_in,dev_d_softmax,batchsize,neurons_out);
      copy_device_to_host_double(dev_d_softmax, tmp_d_softmax, batchsize*neurons_out*neurons_out);

      compare_activations("D_SOFTMAX Result differs beyond threshold on Host and Device",tmp_d_softmax, d_softmax,batchsize,neurons_out,D_SOFTMAX_COMP_THRESHOLD(neurons_out));


      // compare Softmax Backprop
      softmax_activation_backprop_cpu(da,act_in,dz,neurons_out,batchsize);
      softmax_activation_backprop_onDev(dev_da,dev_act_in,dev_dz,neurons_out,batchsize);
      copy_device_to_host_double(dev_dz, tmp, size);

      compare_activations("SOFTMAX BACKPROP Result differs beyond threshold on Host and Device",tmp, dz,batchsize,neurons_out,SOFTMAX_BB_COMP_THRESHOLD(neurons_out));


      // _____________________________________________________________
      // relu host
      relu_activation_cpu(act_in,act_out_relu,size);
      relu_activation_backprop_cpu(da,act_in,dz,size);
      check_relu_backprop("RELU BACKPROP HOST",act_in,dz,da,size);

      //relu device
      relu_activation_gpu(dev_act_in,dev_act_out_relu,size);
      copy_device_to_host_double(dev_act_out_relu, tmp, size);

      relu_activation_backprop_gpu(dev_da,dev_act_in,dev_dz,size);
      copy_device_to_host_double(dev_dz, tmp2, size);
      check_relu_backprop("RELU BACKPROP DEVICE",act_in,tmp2,da,size);


      compare_activations("RELU BACKPROP Result differs beyond threshold on Host and Device",tmp2, dz,batchsize,neurons_out,RELU_BB_COMP_THRESHOLD);
      compare_activations("RELU Result differs beyond threshold on Host and Device",tmp, act_out_relu,batchsize,neurons_out,RELU_COMP_THRESHOLD);

      if(!bigger_than_zero(act_out_relu,size,RELU_COMP_THRESHOLD) || !bigger_than_zero(tmp,size,RELU_COMP_THRESHOLD)) printf("RELU smaller than 0 on device or host\n" );

      // _____________________________________________________________
      // sigmoid host
      sigmoid_activation_cpu(act_in,act_out_sig,size);
      sigmoid_activation_backprop_cpu(da,act_in,dz,size);

      // sigmoid device
      sigmoid_activation_gpu(dev_act_in,dev_act_out_sig,size);
      copy_device_to_host_double(dev_act_out_sig, tmp, size);

      sigmoid_activation_backprop_gpu(dev_da,dev_act_in,dev_dz,size);
      copy_device_to_host_double(dev_dz, tmp2, size);

      compare_activations("SIGMOID BACKPROP Result differs beyond threshold on Host and Device",tmp2, dz,batchsize,neurons_out,SIGMOID_BB_COMP_THRESHOLD);
      compare_activations("SIGMOID Result differs beyond threshold on Host and Device",tmp, act_out_sig,batchsize,neurons_out,SIGMOID_COMP_THRESHOLD);

      if(!bigger_than_zero(act_out_sig,size,SIGMOID_COMP_THRESHOLD) || !bigger_than_zero(tmp,size,SIGMOID_COMP_THRESHOLD) || !smaller_than_value(act_out_sig,size,1.0,SIGMOID_COMP_THRESHOLD) || !smaller_than_value(tmp,size,1.0,SIGMOID_COMP_THRESHOLD)){
        printf("SIGMOID out of bounds on device or host\n" );
      }

      // d_sigmoid
      d_sigmoid_activation_cpu(act_in, dz, size);

      // sigmoid device
      d_sigmoid_activation_gpu(dev_act_in, dev_dz, size);
      copy_device_to_host_double(dev_dz, tmp, size);
      compare_activations("D_SIGMOID Result differs beyond threshold on Host and Device",tmp, dz,batchsize,neurons_out,D_SIGMOID_COMP_THRESHOLD);

      if(!bigger_than_zero(dz,size,SIGMOID_BB_COMP_THRESHOLD) || !bigger_than_zero(tmp,size,SIGMOID_BB_COMP_THRESHOLD)|| !smaller_than_value(dz,size,.25,SIGMOID_BB_COMP_THRESHOLD) || !smaller_than_value(tmp,size,.25,SIGMOID_BB_COMP_THRESHOLD)){
        printf("D_SIGMOID out of bounds on device or host\n" );
      }



      free(act_out_sig);
      free(act_in);
      free(act_out_relu);
      free(act_out_soft);
      free(tmp);
      free(da);
      free(dz);
      free(tmp2);
      CHECK(cudaFree(dev_act_in));
      CHECK(cudaFree(dev_act_out_sig));
      CHECK(cudaFree(dev_act_out_relu));
      CHECK(cudaFree(dev_act_out_soft));
      CHECK(cudaFree(dev_tmp));
      CHECK(cudaFree(dev_tmp2));
      CHECK(cudaFree(dev_da));
      CHECK(cudaFree(dev_dz));
      free(d_softmax);
      free(tmp_d_softmax);
      CHECK(cudaFree(dev_d_softmax));
  }
}

printf("All Checks Done\n\n");

}
