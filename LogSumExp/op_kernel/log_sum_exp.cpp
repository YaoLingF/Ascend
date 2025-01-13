#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template<typename TYPE_X> class KernelLogSumExp{
    using T = TYPE_X;
public:
    __aicore__ inline KernelLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t *shape,int32_t ts)
    {
            
            xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, shape[0]*shape[1]*shape[2]*shape[3]);
            if(ts==1) yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, shape[0]);
            else if(ts==2) yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, shape[1]*shape[2]);
            else yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, shape[0]*shape[2]);

            printf("%d %d %d\n",shape[0],shape[1],shape[2]);
            
            pipe.InitBuffer(X, BUFFER_NUM, 32 * sizeof(TYPE_X));
            pipe.InitBuffer(Y, BUFFER_NUM, 32 * sizeof(float));
            pipe.InitBuffer(Z, BUFFER_NUM, 32 * sizeof(TYPE_X));
                            
            if constexpr (std::is_same_v<TYPE_X, half>)
            {
                if(ts==1)
                {
                    for(int i=0;i<shape[0];i++)
                    {
                        float now = 0.0;
                        for(int j=0;j<shape[1];j++)
                        {
                            
                            for(int k=0;k<shape[2];k++)
                            {
                                LocalTensor<TYPE_X> x1 = X.AllocTensor<TYPE_X>();
                                DataCopy(x1, xGm[i*shape[1]*shape[2]+j*shape[2]+k], 32);
                                X.EnQue(x1);
                                LocalTensor<TYPE_X> x2 = X.DeQue<TYPE_X>();
                                Exp(x2,x2,32);
                                half val = x2.GetValue(0); 
                                now+=(float)val;
                                X.FreeTensor(x2);
                            }
                        
                        }

                        LocalTensor<float> y1 = Y.AllocTensor<float>();
                        Duplicate(y1,now,32);
                        Y.EnQue(y1);
                        LocalTensor<float> y2 = Y.DeQue<float>();
                        Ln(y2,y2,32);
                        yGm.SetValue(i,(half)y2.GetValue(0));
                        Y.FreeTensor(y2);
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
                                LocalTensor<TYPE_X> x1 = X.AllocTensor<TYPE_X>();
                                DataCopy(x1, xGm[i*shape[1]*shape[2]+k*shape[2]+j], 32);
                                X.EnQue(x1);
                                LocalTensor<TYPE_X> x2 = X.DeQue<TYPE_X>();
                                Exp(x2,x2,32);
                                half val = x2.GetValue(0); 
                                now+=(float)val;

                                X.FreeTensor(x2);
                            }
                            LocalTensor<float> y1 = Y.AllocTensor<float>();
                            Duplicate(y1,now,32);
                            Y.EnQue(y1);
                            LocalTensor<float> y2 = Y.DeQue<float>();
                            Ln(y2,y2,32);
                            yGm.SetValue(i*shape[2]+j,(half)y2.GetValue(0));
                            Y.FreeTensor(y2);
                        }
                    }
                }
            }
            else
            {
                if(ts==2)
                {
                    for(int i=0;i<shape[1];i++)
                    {
                        
                        for(int j=0;j<shape[2];j++)
                        {
                            float now = 0.0;
                            for(int k=0;k<shape[0];k++)
                            {
                                for(int z=0;z<shape[3];z++)
                                {
                                    LocalTensor<TYPE_X> x1 = X.AllocTensor<TYPE_X>();
                                    DataCopy(x1, xGm[k*shape[1]*shape[2]*shape[3]+i*shape[2]*shape[3]+j*shape[3]+z], 32);
                                    X.EnQue(x1);
                                    LocalTensor<TYPE_X> x2 = X.DeQue<TYPE_X>();
                                    Exp(x2,x2,32);
                                    float val = x2.GetValue(0); 
                                    now+=val;
                                    X.FreeTensor(x2);
                                }
                            }
                            LocalTensor<float> y1 = Y.AllocTensor<float>();
                            Duplicate(y1,now,32);
                            Y.EnQue(y1);
                            LocalTensor<float> y2 = Y.DeQue<float>();
                            Ln(y2,y2,32);
                            yGm.SetValue(i*shape[2]+j,y2.GetValue(0));
                            Y.FreeTensor(y2);
                        
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
                                LocalTensor<TYPE_X> x1 = X.AllocTensor<TYPE_X>();
                                printf("%d\n",k);
                                DataCopy(x1, xGm[i*shape[1]*shape[2]+k*shape[2]+j], 32);
                                X.EnQue(x1);
                                LocalTensor<TYPE_X> x2 = X.DeQue<TYPE_X>();
                                //printf("%f %f ",xGm.GetValue(i*shape[1]*shape[2]+k*shape[2]+j),x1.GetValue(0));
                                Exp(x2,x2,32);
                                float val = x2.GetValue(0); 
                                now+=val;

                                X.FreeTensor(x2);
                            }

                            LocalTensor<float> y1 = Y.AllocTensor<float>();
                            Duplicate(y1,now,32);
                            Y.EnQue(y1);
                            LocalTensor<float> y2 = Y.DeQue<float>();
                            Ln(y2,y2,32);
                            yGm.SetValue(i*shape[2]+j,y2.GetValue(0));
                            Y.FreeTensor(y2);
                        
                        }
                    }
                }
            }

       
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> X;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Y,Z;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> yGm;

};
extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelLogSumExp<DTYPE_X> op;
    op.Init(x,y,tiling_data.shape,tiling_data.ts);
}