#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelNLLLoss{
    using T = TYPE_X;
public:
    __aicore__ inline KernelNLLLoss() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, int32_t *shape, int32_t mode, int32_t ignore)
    {
            int32_t out_size=1;
            if(mode==1)
            {
                out_size=shape[0]*shape[2];
            }
            xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, shape[0]*shape[1]*shape[2]);
            yGm.SetGlobalBuffer((__gm__ int32_t *)target, shape[0]*shape[2]);
            zGm.SetGlobalBuffer((__gm__ TYPE_X *)weight, shape[1]);//n
            oGm.SetGlobalBuffer((__gm__ TYPE_X *)y, out_size);
            float cur;
            float w;
            float sum=0.0;
            float al_w=0.0;

            for(int i=0;i<shape[0];i++)
            for(int j=0;j<shape[2];j++)
            {
                
                cur=-(float)xGm.GetValue(i*shape[1]*shape[2]+yGm.GetValue(i*shape[2]+j)*shape[2]+j);
                w=(float)zGm.GetValue(yGm.GetValue(i*shape[2]+j));
                sum+=cur*w;
                al_w+=w;
                if(mode == 1){
                    oGm.SetValue(i*shape[2]+j,(TYPE_X)(cur*w));
                }
            }
            if(mode == 2){
                sum=sum/(al_w);
                oGm.SetValue(0,(TYPE_X)(sum));
            }
            if(mode == 3){
                oGm.SetValue(0,(TYPE_X)(sum));
            }
        
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<DTYPE_X> oGm;
    GlobalTensor<DTYPE_X> zGm;

};
extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl

    KernelNLLLoss<DTYPE_X> op;
    op.Init(x,target,weight,y,tiling_data.shape,tiling_data.mode,tiling_data.ignore);
}