#include "kernel_operator.h"
#include<cmath>
//#include<iostream>
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelReplicationPad2d{
    using T = TYPE_X;
public:
    __aicore__ inline KernelReplicationPad2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, int32_t *shape, int32_t p)
    {
            
            x1Gm.SetGlobalBuffer((__gm__ TYPE_X *)x, shape[0]*shape[1]*shape[2]);
            x2Gm.SetGlobalBuffer((__gm__ int32_t *)paddings, 2*p);
            

            int32_t l = x2Gm.GetValue(0);
            int32_t r = x2Gm.GetValue(1);
            int32_t top = x2Gm.GetValue(2);
            int32_t down = x2Gm.GetValue(3);

            yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, shape[0]*(shape[1]+top+down)*(shape[2]+l+r));
            
            int32_t s1 = shape[1]+top+down;
            int32_t s2 = shape[2]+l+r;
            
            
                            
            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                for(int i=0;i<shape[0];i++)
                {
                    for(int j=0;j<top;j++)
                    {
                        for(int k=0;k<l;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]));
                        }
                        for(int k=l;k<l+shape[2];k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+k-l));
                        }
                        for(int k=l+shape[2];k<s2;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+shape[2]-1));
                        }
                    }

                    for(int j=top;j<top+shape[1];j++)
                    {
                        for(int k=0;k<l;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(j-top)*shape[2]));
                        }
                        for(int k=l;k<l+shape[2];k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(j-top)*shape[2]+k-l));//kong
                        }
                        for(int k=l+shape[2];k<s2;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(j-top+1)*shape[2]-1));
                        }
                    }

                    for(int j=top+shape[1];j<s1;j++)
                    {
                        for(int k=0;k<l;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(shape[1]-1)*shape[2]));
                        }
                        for(int k=l;k<l+shape[2];k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(shape[1]-1)*shape[2]+k-l));
                        }
                        for(int k=l+shape[2];k<s2;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+shape[1]*shape[2]-1));
                        }
                    }
                    
                }
            }
            else
            {
                for(int i=0;i<shape[0];i++)
                {
                    for(int j=0;j<top;j++)
                    {
                        for(int k=0;k<l;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]));
                        }
                        for(int k=l;k<l+shape[2];k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+k-l));
                        }
                        for(int k=l+shape[2];k<s2;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+shape[2]-1));
                        }
                    }

                    for(int j=top;j<top+shape[1];j++)
                    {
                        for(int k=0;k<l;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(j-top)*shape[2]));
                        }
                        for(int k=l;k<l+shape[2];k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(j-top)*shape[2]+k-l));//kong
                        }
                        for(int k=l+shape[2];k<s2;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(j-top+1)*shape[2]-1));
                        }
                    }

                    for(int j=top+shape[1];j<s1;j++)
                    {
                        for(int k=0;k<l;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(shape[1]-1)*shape[2]));
                        }
                        for(int k=l;k<l+shape[2];k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+(shape[1]-1)*shape[2]+k-l));
                        }
                        for(int k=l+shape[2];k<s2;k++)
                        {
                            yGm.SetValue(i*s1*s2+j*s2+k,x1Gm.GetValue(i*shape[1]*shape[2]+shape[1]*shape[2]-1));
                        }
                    }
                    
                }
                
            }

            
            
       
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> x1Gm;
    GlobalTensor<int32_t> x2Gm;
    GlobalTensor<DTYPE_X> yGm;

};

extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelReplicationPad2d<DTYPE_X> op;
    op.Init(x,paddings,y,tiling_data.shape,tiling_data.p);
}