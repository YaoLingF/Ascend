#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;



template<typename TYPE_X> class KernelNonMaxSuppression{
    using T = TYPE_X;
public:

    __aicore__ inline KernelNonMaxSuppression() {}
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices, int32_t type,int32_t batch, int32_t classes, int32_t num)
    {

            if(num==1024)
            {
                pipe.InitBuffer(X, BUFFER_NUM, 1024*4* sizeof(float));
                xGm.SetGlobalBuffer((__gm__ float *)boxes, batch*num*4);
                yGm.SetGlobalBuffer((__gm__ float *)scores, batch*classes*num);
                zGm.SetGlobalBuffer((__gm__ int32_t *)selected_indices, batch*classes*num*3);
                iouGm.SetGlobalBuffer((__gm__ float *)iou_threshold, 1);
                scoreGm.SetGlobalBuffer((__gm__ float *)score_threshold, 1);
                maxGm.SetGlobalBuffer((__gm__ int32_t *)max_output_boxes_per_class, 1);

                int32_t maxn = maxGm.GetValue(0);

                float scoree = scoreGm.GetValue(0);

                float iou = iouGm.GetValue(0);

                

                int32_t now=0;

                for(int i=0;i<batch;i++)
                {
                    LocalTensor<float> x = X.AllocTensor<float>();
                    DataCopy(x,xGm[i*1024*4],1024*4);
                    X.EnQue(x);
                    LocalTensor<float> xx = X.DeQue<float>();

                    for(int j=0;j<classes;j++)
                    {
                        
                        bool sup[1024]={false};
                        float score[1024];
                        

                        for(int k=0;k<num;k++)
                        {
                            score[k]=yGm.GetValue(i*classes*num+j*num+k);

                        }

                        for(int k=0;k<num;k++)
                        {
                            if(score[k]<scoree) sup[k]=true;
                        }
                        
                        for(int _=0;_<num&&_<maxn;_++)
                        {
                            int32_t index=-1;
                            float top=0;
                            for(int k=0;k<num;k++)
                            {
                                if(sup[k]) continue;
                                if(index==-1) index=k,top=score[k];
                                else if(score[k]>top) index=k,top=score[k];
                            }
                            if(index==-1) break;
                            sup[index]=true;
                            zGm.SetValue(now*3,i);
                            zGm.SetValue(now*3+1,j);
                            zGm.SetValue(now*3+2,index);
                            now++;
                            for(int k=0;k<num;k++)
                            {
                                if(sup[k]) continue;//index k 
                                float x1=xx.GetValue(index*4+1),y1=xx.GetValue(index*4),x2=xx.GetValue(index*4+3),y2=xx.GetValue(index*4+2);

                                float b1x1=x1>x2?x2:x1,b1x2=x1>x2?x1:x2,b1y1=y1>y2?y2:y1,b1y2=y1>y2?y1:y2;

                                x1=xx.GetValue(k*4+1),y1=xx.GetValue(k*4),x2=xx.GetValue(k*4+3),y2=xx.GetValue(k*4+2);

                                float b2x1=x1>x2?x2:x1,b2x2=x1>x2?x1:x2,b2y1=y1>y2?y2:y1,b2y2=y1>y2?y1:y2;

                                x1=b1x1>b2x1?b1x1:b2x1,x2=b1x2>b2x2?b2x2:b1x2;
                                y1=b1y1>b2y1?b1y1:b2y1,y2=b1y2>b2y2?b2y2:b1y2;

                                float inter=(x2-x1>float(0.0)?x2-x1:float(0.0))*(y2-y1>float(0.0)?y2-y1:float(0.0));
                                float b1area=(b1x2-b1x1)*(b1y2-b1y1);
                                float b2area=(b2x2-b2x1)*(b2y2-b2y1);

                                float unionarea=b1area+b2area-inter;

                                float bi=inter/unionarea;

                                if(bi>iou) sup[k]=true;
                                
                            }
                        }
                    }

                    X.FreeTensor(xx);
                }
            }

            else
            {

                xGm.SetGlobalBuffer((__gm__ float *)boxes, batch*num*4);
                yGm.SetGlobalBuffer((__gm__ float *)scores, batch*classes*num);
                zGm.SetGlobalBuffer((__gm__ int32_t *)selected_indices, batch*classes*num*3);
                iouGm.SetGlobalBuffer((__gm__ float *)iou_threshold, 1);
                scoreGm.SetGlobalBuffer((__gm__ float *)score_threshold, 1);
                maxGm.SetGlobalBuffer((__gm__ int32_t *)max_output_boxes_per_class, 1);

                int32_t maxn = maxGm.GetValue(0);

                float scoree = scoreGm.GetValue(0);

                float iou = iouGm.GetValue(0);

                

                int32_t now=0;

                for(int i=0;i<batch;i++)
                {
                    for(int j=0;j<classes;j++)
                    {


                        bool sup[1024]={false};
                        float score[1024];
                        

                        for(int k=0;k<num;k++)
                        {
                            score[k]=yGm.GetValue(i*classes*num+j*num+k);

                            // for(int _=0;_<4;_++)
                            // {
                            //     box[k][_]=xGm.GetValue(i*num*4+k*4+_);
                            // }
                        }

                        for(int k=0;k<num;k++)
                        {
                            if(score[k]<scoree) sup[k]=true;
                        }
                        
                        for(int _=0;_<num&&_<maxn;_++)
                        {
                            int32_t index=-1;
                            float top=0;
                            for(int k=0;k<num;k++)
                            {
                                if(sup[k]) continue;
                                if(index==-1) index=k,top=score[k];
                                else if(score[k]>top) index=k,top=score[k];
                            }
                            if(index==-1) break;
                            sup[index]=true;
                            zGm.SetValue(now*3,i);
                            zGm.SetValue(now*3+1,j);
                            zGm.SetValue(now*3+2,index);
                            now++;
                            for(int k=0;k<num;k++)
                            {
                                if(sup[k]) continue;//index k 
                                float x1=xGm.GetValue(i*num*4+index*4+1),y1=xGm.GetValue(i*num*4+index*4),x2=xGm.GetValue(i*num*4+index*4+3),y2=xGm.GetValue(i*num*4+index*4+2);

                                float b1x1=x1>x2?x2:x1,b1x2=x1>x2?x1:x2,b1y1=y1>y2?y2:y1,b1y2=y1>y2?y1:y2;

                                x1=xGm.GetValue(i*num*4+k*4+1),y1=xGm.GetValue(i*num*4+k*4),x2=xGm.GetValue(i*num*4+k*4+3),y2=xGm.GetValue(i*num*4+k*4+2);

                                float b2x1=x1>x2?x2:x1,b2x2=x1>x2?x1:x2,b2y1=y1>y2?y2:y1,b2y2=y1>y2?y1:y2;

                                x1=b1x1>b2x1?b1x1:b2x1,x2=b1x2>b2x2?b2x2:b1x2;
                                y1=b1y1>b2y1?b1y1:b2y1,y2=b1y2>b2y2?b2y2:b1y2;

                                float inter=(x2-x1>float(0.0)?x2-x1:float(0.0))*(y2-y1>float(0.0)?y2-y1:float(0.0));
                                float b1area=(b1x2-b1x1)*(b1y2-b1y1);
                                float b2area=(b2x2-b2x1)*(b2y2-b2y1);

                                float unionarea=b1area+b2area-inter;

                                float bi=inter/unionarea;

                                if(bi>iou) sup[k]=true;
                                
                            }
                        }
                    }
                }
            }
    }
    
private:
    TPipe pipe;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<int32_t> zGm;

    GlobalTensor<float> iouGm;
    GlobalTensor<float> scoreGm;
    GlobalTensor<int32_t> maxGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> X;
    


};
extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelNonMaxSuppression<float> op;
    op.Init(boxes,scores,max_output_boxes_per_class,iou_threshold,score_threshold,selected_indices,tiling_data.type,tiling_data.batch,tiling_data.classes,tiling_data.num);
}