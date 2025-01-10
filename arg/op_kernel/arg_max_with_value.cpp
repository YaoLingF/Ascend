#include "kernel_operator.h"
#include<type_traits>
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelArgMaxWithValue{
    using T = TYPE_X;
public:
    __aicore__ inline KernelArgMaxWithValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, int32_t *shape,int32_t precal, int32_t sufcal, int32_t prenum, int32_t sufnum)
    {
            int32_t idx = GetBlockIdx();

            int32_t cal = (idx<prenum?precal:sufcal);

            auto startp = idx<prenum? idx*precal:prenum*precal+(idx-prenum)*sufcal;
            int32_t star = idx<prenum? idx*precal:prenum*precal+(idx-prenum)*sufcal;


            
            xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, shape[0]*shape[1]*shape[2]);
            y1Gm.SetGlobalBuffer((__gm__ int32_t *)indice, cal);
            y2Gm.SetGlobalBuffer((__gm__ TYPE_X *)values, cal);

            printf("%d\n",idx);

            
            int32_t t = shape[1]*shape[2];
            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                for(int _=startp;_<startp+cal&&_<shape[0]*shape[2];_++)
                {
                    int i=_/shape[2];
                    int j=_%shape[2];
                    int32_t res = 0;
                    float now = (float)xGm.GetValue(i*t+j);
                    for(int k=1;k<shape[1];k++)
                    {
                        float val = (float)xGm.GetValue(i*t+k*shape[2]+j);
                        if(val>now) now=val,res=k;

                    }

                    printf("%d %d %d\n",idx,_,res);
                    y1Gm.SetValue(_,res);
                    y2Gm.SetValue(_,(TYPE_X)now);
                }
            }
            else
            {

                for(int _=startp;_<startp+cal&&_<shape[0]*shape[2];_++)
                {
                    int i=_/shape[2];
                    int j=_%shape[2];
                    int32_t res = 0;
                    TYPE_X now = xGm.GetValue(i*t+j);
                    for(int k=1;k<shape[1];k++)
                    {
                        TYPE_X val = xGm.GetValue(i*t+k*shape[2]+j);
                        if(val>now) now=val,res=k;

                    }
                    y1Gm.SetValue(_,res);
                    y2Gm.SetValue(_,now);
                }
            }


            //AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(y1Gm);
            //AscendC::DataCacheCleanAndInvalid<DTYPE_X, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(y2Gm);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<int32_t> y1Gm;
    GlobalTensor<DTYPE_X> y2Gm;

};

extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelArgMaxWithValue<DTYPE_X> op;
    op.Init(x,indice,values,tiling_data.shape,tiling_data.precal,tiling_data.sufcal,tiling_data.prenum,tiling_data.sufnum);
}