

#include "kernel_operator.h"
#include<cmath>
//#include<iostream>
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelMatMulSub{
    using T = TYPE_X;
public:
    __aicore__ inline KernelMatMulSub() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, int32_t M, int32_t K, int32_t N, int32_t keep)
    {
            
            x1Gm.SetGlobalBuffer((__gm__ TYPE_X *)x1, M*K);
            x2Gm.SetGlobalBuffer((__gm__ TYPE_X *)x2, K*N);
            x3Gm.SetGlobalBuffer((__gm__ TYPE_X *)x3, (keep==1?N:M*N));
            yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, M*N);
            
            pipe.InitBuffer(inQueueX, BUFFER_NUM, K * sizeof(float));
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            pipe.InitBuffer(inQueueY, BUFFER_NUM, K * sizeof(float));
            LocalTensor<float> yLocal = inQueueX.AllocTensor<float>();
            
                            
            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                for(int i=0;i<M;i++)
                {
                    for(int j=0;j<N;j++)
                    {
                        for(int k=0;k<K;k++)
                        {
                            float k1 = (float)x1Gm.GetValue(i*K+k);
                            float k2 = (float)x2Gm.GetValue(k*N+j);
                            xLocal.SetValue(k,k1*k2);
                        
                        }
                        ReduceSum(xLocal,xLocal,yLocal,K);
                        float sum = xLocal.GetValue(0);
                        float k3 = (float)x3Gm.GetValue((keep==1?j:i*N+j));
                        sum-=k3;
                        yGm.SetValue(i*N+j,half(sum));
                    }
                }
            }
            else
            {

                for(int i=0;i<M;i++)
                {
                    for(int j=0;j<N;j++)
                    {
                        for(int k=0;k<K;k++)
                        {
                            float k1 = x1Gm.GetValue(i*K+k);
                            float k2 = x2Gm.GetValue(k*N+j);
                            xLocal.SetValue(k,k1*k2);
                        
                        }
                        ReduceSum(xLocal,xLocal,yLocal,K);
                        float sum = xLocal.GetValue(0);
                        float k3 = x3Gm.GetValue((keep==1?j:i*N+j));
                        sum-=k3;
                        yGm.SetValue(i*N+j,sum);
                    }
                }
            }

            
           
       
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<TYPE_X> x1Gm;
    GlobalTensor<TYPE_X> x2Gm;
    GlobalTensor<TYPE_X> x3Gm;
    GlobalTensor<TYPE_X> yGm;

};

extern "C" __global__ __aicore__ void mat_mul_sub(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelMatMulSub<DTYPE_X1> op;
    op.Init(x1,x2,x3,y,tiling_data.M,tiling_data.K,tiling_data.N,tiling_data.keep);
}