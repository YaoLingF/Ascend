#include "kernel_operator.h"
#include<cmath>
//#include<iostream>
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelScatterElements{
    using T = TYPE_X;
public:
    __aicore__ inline KernelScatterElements() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, int32_t *shape1, int32_t *shape2, int32_t *shape3, int32_t p, int32_t mode, int32_t dim)
    {
            
            x1Gm.SetGlobalBuffer((__gm__ TYPE_X *)var, shape1[0]*shape1[1]*shape1[2]*shape1[3]);
            x2Gm.SetGlobalBuffer((__gm__ int32_t *)indices, shape2[0]*shape2[1]*shape2[2]*shape2[3]);
            x3Gm.SetGlobalBuffer((__gm__ TYPE_X *)updates, shape3[0]*shape3[1]*shape3[2]*shape3[3]);

            yGm.SetGlobalBuffer((__gm__ TYPE_X *)var_ref, shape1[0]*shape1[1]*shape1[2]*shape1[3]);

            for(int i=0;i<shape1[0];i++)
            {
                for(int j=0;j<shape1[1];j++)
                {
                    for(int k=0;k<shape1[2];k++)
                    {
                        for(int z=0;z<shape1[2];z++)
                        {
                            yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,x1Gm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z));
                        }
                    }
                }
            }
                            
            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                for(int i=0;i<shape2[0];i++)
                {
                    for(int j=0;j<shape2[1];j++)
                    {
                        for(int k=0;k<shape2[2];k++)
                        {
                            for(int z=0;z<shape2[3];z++)
                            {
                                if(dim==0)
                                {
                                    if(mode==1)//add
                                    {
                                        //self[index[i][j][k]][j][k]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,T(st*ad));
                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,T(ad));

                                    }
                                }
                                else if(dim==1)
                                {
                                    if(mode==1)//add
                                    {
                                        //self[i][index[i][j][k]][k]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z,T(st*ad));
                                        
                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z,T(ad));

                                    }
                                }
                                else if(dim==2)
                                {
                                    if(mode==1)//add
                                    {
                                        //self[i][j][index[i][j][k]]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z,T(st*ad));

                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z,T(ad));

                                    }

                                }
                                else//dim==3
                                {
                                    if(mode==1)//add
                                    {
                                        //self[i][j][index[i][j][k]]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index,T(st*ad));

                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        float st = (float)yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index);
                                        float ad = (float)x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index,T(ad));

                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for(int i=0;i<shape2[0];i++)
                {
                    for(int j=0;j<shape2[1];j++)
                    {
                        for(int k=0;k<shape2[2];k++)
                        {
                            for(int z=0;z<shape2[3];z++)
                            {
                                if(dim==0)
                                {
                                    if(mode==1)//add
                                    {
                                        //self[index[i][j][k]][j][k]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st = yGm.GetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        T ad = x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st = yGm.GetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        T ad = x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,T(st*ad));
                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st = yGm.GetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        T ad = x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(index*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+z,T(ad));

                                    }
                                }
                                else if(dim==1)
                                {
                                    if(mode==1)//add
                                    {
                                        //self[i][index[i][j][k]][k]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z,T(st*ad));
                                        
                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+index*shape1[2]*shape1[3]+k*shape1[3]+z,T(ad));

                                    }
                                }
                                else if(dim==2)
                                {
                                    if(mode==1)//add
                                    {
                                        //self[i][j][index[i][j][k]]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z,T(st*ad));

                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+index*shape1[3]+z,T(ad));

                                    }

                                }
                                else//dim==3
                                {
                                    if(mode==1)//add
                                    {
                                        //self[i][j][index[i][j][k]]+=src[i][j][k]
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index,T(st+ad));

                                    }
                                    else if(mode==2)//mul
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index,T(st*ad));

                                    }
                                    else//none
                                    {
                                        int32_t index = x2Gm.GetValue(i*shape2[1]*shape2[2]*shape2[3]+j*shape2[2]*shape2[3]+k*shape2[3]+z);
                                        T st =  yGm.GetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index);
                                        T ad =  x3Gm.GetValue(i*shape3[1]*shape3[2]*shape3[3]+j*shape3[2]*shape3[3]+k*shape3[3]+z);
                                        yGm.SetValue(i*shape1[1]*shape1[2]*shape1[3]+j*shape1[2]*shape1[3]+k*shape1[3]+index,T(ad));

                                    }
                                }
                            }
                        }
                    }
                }
            }

            
            
       
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<int32_t> x2Gm;
    GlobalTensor<TYPE_X> x3Gm;
    GlobalTensor<TYPE_X> yGm;

};


extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelScatterElements<DTYPE_VAR> op;
    op.Init(var, indices, updates, var_ref,tiling_data.shape1,tiling_data.shape2,tiling_data.shape3,tiling_data.p,tiling_data.mode,tiling_data.dim);
}