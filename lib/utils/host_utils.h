#ifndef _HOST_UTILS_H_
#define _HOST_UTILS_H_


template<class real_type>
void apply_pointwise(const real_type* in,real_type* out,int size,pointwise<real_type> func){
    for(int i=0;i<size;i++) out[i]=func(in[i]);
}


#endif // _HOST_UTILS_H_
