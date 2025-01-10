#include "kernel_operator.h"
#include<cmath>
//#include<iostream>
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelSoftmax{
    using T = TYPE_X;
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t *shape)
    {
            
            xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, shape[0]*shape[1]*shape[2]);
            yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, shape[0]*shape[1]*shape[2]);
            
            pipe.InitBuffer(inQueueX, BUFFER_NUM, 32 * sizeof(TYPE_X));
            LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
                            
            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                for(int i=0;i<shape[0];i++)
                {
                    for(int j=0;j<shape[2];j++)
                    {
                        float now = 0.0;
                        for(int k=0;k<shape[1];k++)
                        {
                            DataCopy(xLocal, xGm[i*shape[1]*shape[2]+k*shape[2]+j], 32);
                            printf("%f %f ",xGm.GetValue(i*shape[1]*shape[2]+k*shape[2]+j),xLocal.GetValue(0));
                            Exp(xLocal,xLocal,1);
                            half val = xLocal.GetValue(0); 
                            now+=(float)val;
                            //printf("%f %f\n",val,now);
                        }
                        printf("%f\n",now);
                        for(int k=0;k<shape[1];k++)
                        {
                            DataCopy(xLocal, xGm[i*shape[1]*shape[2]+k*shape[2]+j], 32);
                            printf("%f %f\n",xGm.GetValue(i*shape[1]*shape[2]+k*shape[2]+j),xLocal.GetValue(0));
                            Exp(xLocal,xLocal,1);
                            half val = xLocal.GetValue(0);
                            //printf("%f %f\n",val,now);
                            yGm.SetValue(i*shape[1]*shape[2]+k*shape[2]+j,TYPE_X(float(val)/now));
                            //printf("%f\n",yGm.GetValue(i*shape[1]*shape[2]+k*shape[2]+j));
                        }
                      
                    }
                }
            }
            else
            {

                for(int i=0;i<shape[0];i++)
                {
                    for(int j=0;j<shape[2];j++)
                    {
                        
                        float now = 0.0;
                        for(int k=0;k<shape[1];k++)
                        {
                            DataCopy(xLocal, xGm[i*shape[1]*shape[2]+k*shape[2]+j], 32);
                            printf("%f %f ",xGm.GetValue(i*shape[1]*shape[2]+k*shape[2]+j),xLocal.GetValue(0));
                            Exp(xLocal,xLocal,1);
                            float val = xLocal.GetValue(0);                       
                            now+=val;
                        }
                        for(int k=0;k<shape[1];k++)
                        {
                            DataCopy(xLocal, xGm[i*shape[1]*shape[2]+k*shape[2]+j], 32);
                            printf("%f %f ",xGm.GetValue(i*shape[1]*shape[2]+k*shape[2]+j),xLocal.GetValue(0));
                            Exp(xLocal,xLocal,1);
                            float val = xLocal.GetValue(0);
                            yGm.SetValue(i*shape[1]*shape[2]+k*shape[2]+j,val/now);
                        }
                      
                    }
                }
            }

            
            inQueueX.FreeTensor(xLocal);
       
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> yGm;

};



extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSoftmax<DTYPE_X> op;
    op.Init(x,y,tiling_data.shape);
}