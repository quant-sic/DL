#ifndef TENSOR_H
#define TENSOR_H

#ifdef MAIN_PROGRAM
   #define EXTERN
#else
   #define EXTERN extern
#endif



typedef struct{
  int ndim;
  int nelem;
  int *dim_sizes;
  double *data;
  int ***CM;
}Tensor;

#undef EXTERN

#endif
