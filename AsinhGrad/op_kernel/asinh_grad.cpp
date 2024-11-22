#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     
template<typename TYPE_Y, typename TYPE_DY, typename TYPE_Z>
class KernelAsinhGrad {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint32_t prenum, uint32_t sufnum, uint32_t presize, uint32_t sufsize, uint32_t block_size) {

        idx = GetBlockIdx();

        this->blockLength = idx < prenum ? presize : sufsize;
        this->tileLength = block_size;

        auto startPointer = idx < prenum ? idx * presize : prenum * presize + (idx - prenum) * sufsize;
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, this->blockLength);
        dyGm.SetGlobalBuffer((__gm__ TYPE_DY*)dy + startPointer, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + startPointer, this->blockLength);
        this->tileNum = (this->blockLength +this->tileLength - 1)/ this->tileLength;
        
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileLength * sizeof(TYPE_DY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z));
        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(B_x, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_z, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {

        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {

            if(i == loopCount-1)
            {
                CopyIn(i, this->blockLength - (loopCount - 1) * this->tileLength);
                Compute(i, this->blockLength - (loopCount - 1) * this->tileLength);
                CopyOut(i, this->blockLength - (loopCount - 1) * this->tileLength);
            }
            else
            {
                CopyIn(i, this->tileLength);
                Compute(i, this->tileLength);
                CopyOut(i, this->tileLength);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
        LocalTensor<TYPE_DY> dyLocal = inQueueDY.AllocTensor<TYPE_DY>();
        DataCopy(yLocal, yGm[progress * this->tileLength], length);
        DataCopy(dyLocal, dyGm[progress * this->tileLength], length);
        inQueueY.EnQue(yLocal);
        inQueueDY.EnQue(dyLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
        LocalTensor<TYPE_DY> dyLocal = inQueueDY.DeQue<TYPE_DY>();
        LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();

        if constexpr (std::is_same_v<T, half>)
        {
            auto k1 = B_x.Get<float>();
            Cast(k1, yLocal, RoundMode::CAST_NONE, length);
            auto k2 = B_y.Get<float>();
            Cast(k2, dyLocal, RoundMode::CAST_NONE, length);
            auto k3 = B_z.Get<float>();
    
            float inputVal1 = 1.0;
            float inputVal2 = 2.0;

            Exp(k1, k1, length);
            Mul(k3, k1, k1, length);
            Adds(k3, k3, inputVal1, length);
            Muls(k1, k1, inputVal2, length);
            Mul(k1, k1, k2, length);
            Div(k3, k1, k3, length);

            Cast(zLocal, k3, RoundMode::CAST_NONE, length);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            TYPE_Y inputVal1 = 1.0;
            TYPE_Y inputVal2 = 2.0;
            
            Exp(yLocal, yLocal, length);
            Mul(zLocal, yLocal, yLocal, length);
            Adds(zLocal, zLocal, inputVal1, length);
            Muls(yLocal, yLocal, inputVal2, length);
            Mul(yLocal, yLocal, dyLocal, length);
            Div(zLocal, yLocal, zLocal, length);
        }
        
        outQueueZ.EnQue<TYPE_Z>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueDY.FreeTensor(dyLocal);
        
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY,inQueueDY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<QuePosition::VECCALC> B_x,B_y,B_z;
    GlobalTensor<TYPE_Y> yGm;
    GlobalTensor<TYPE_DY> dyGm;
    GlobalTensor<TYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t idx;
};
extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinhGrad<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;
    op.Init(y, dy, z, tiling_data.prenum, tiling_data.sufnum, tiling_data.presize, tiling_data.sufsize, tiling_data.block_size); 
    op.Process();
}