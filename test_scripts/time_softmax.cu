#include "activations.h"
#include "costfunctions.h"
#include "common.h"
#include <float.h>
#include <math.h>

int main(int argc, char **argv)
{
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("\nTiming Softmax using device %d: %s \n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
  printf("_________________________________________________\n");


  double *act_in,*act_out_soft,*da,*dz,*delta;
  double *dev_act_in,*dev_act_out_soft,*dev_da,*dev_dz,*dev_delta;
  double start,t,t1,t2,t3,t4,t5,t6,t7,t8,t9;

  FILE *softmax_times = fopen("../analysis/softmax_times_sgpu_kernel.txt", "w");
  fprintf(softmax_times,"SIZE\tS_TIME_onDEV\tS_TIME_HOST\tSB_TIME_onDEV\tSB_TIME_HOST\tCT_DtH\tCT_HtD\tD_STIME_onDEV\tD_STIME_HOST\tBB_OPT_TIME\n");


  for(int size=1;size<=(1<<18);size<<=1){

        t1=t2=t3=t4=t5=t6=t7=t8=t9=DBL_MAX;
        int neurons_out=(int)sqrt(size);
        int batchsize=(int)sqrt(size);


        act_in=(double *)malloc(size*sizeof(double));
        da=(double *)malloc(size*sizeof(double));
        dz=(double *)malloc(size*sizeof(double));
        act_out_soft=(double *)malloc(size*sizeof(double));
        delta=(double *)malloc(batchsize*neurons_out*neurons_out*sizeof(double));


        for(int i =0;i<size;i++) act_in[i]=2*((double)rand()/(double)RAND_MAX)-1;
        for(int i =0;i<size;i++) da[i]=2*((double)rand()/(double)RAND_MAX)-1;

      	CHECK(cudaMalloc((void**)&dev_act_in, size*sizeof(double)));
      	CHECK(cudaMalloc((void**)&dev_da, size*sizeof(double)));
      	CHECK(cudaMalloc((void**)&dev_dz, size*sizeof(double)));
      	CHECK(cudaMalloc((void**)&dev_act_out_soft, size*sizeof(double)));
        CHECK(cudaMalloc((void**)&dev_delta, batchsize*neurons_out*neurons_out*sizeof(double)));

        copy_host_to_device_double(act_in, dev_act_in, size);
        copy_host_to_device_double(da, dev_da, size);

        for(int i=0;i<10;i++){


          start=seconds();
          softmax_activation_onDev(dev_act_in,dev_act_out_soft,batchsize,neurons_out);
          t=seconds()-start;
          t1=t<t1?t:t1;

          start=seconds();
          softmax_activation_cpu(act_in,act_out_soft,batchsize,neurons_out);
          t=seconds()-start;
          t2=t<t2?t:t2;


          start=seconds();
          softmax_activation_backprop_onDev(dev_da,dev_act_in,dev_dz,neurons_out,batchsize);
          t=seconds()-start;
          t3=t<t3?t:t3;

          start=seconds();
          softmax_activation_backprop_cpu(da,act_in,dz,neurons_out,batchsize);
          t=seconds()-start;
          t4=t<t4?t:t4;

          start=seconds();
          copy_device_to_host_double(dev_da, da, size);
          t=seconds()-start;
          t5=t<t5?t:t5;

          start=seconds();
          copy_host_to_device_double(dz, dev_dz, size);
          t=seconds()-start;
          t6=t<t6?t:t6;

          start=seconds();
          d_softmax_activation_onDev(dev_act_in,dev_delta, batchsize,neurons_out);
          t=seconds()-start;
          t7=t<t7?t:t7;

          start=seconds();
          d_softmax_activation_cpu(act_in,delta, batchsize,neurons_out);
          t=seconds()-start;
          t8=t<t8?t:t8;

        }

        fprintf(softmax_times,"%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",size,t1,t2,t3,t4,t5,t6,t7,t8,t9);

        free(act_in);
        free(act_out_soft);
        free(da);
        free(delta);
        free(dz);
        CHECK(cudaFree(dev_act_in));
        CHECK(cudaFree(dev_act_out_soft));
        CHECK(cudaFree(dev_da));
        CHECK(cudaFree(dev_dz));
        CHECK(cudaFree(dev_delta));


   }

fclose(softmax_times);

}
