#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     
template<typename TYPE_Y, typename TYPE_DY, typename TYPE_Z>
class KernelAsinhGrad {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelAsinhGrad() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        //this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();

        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, this->blockLength);
        dyGm.SetGlobalBuffer((__gm__ TYPE_DY*)dy + startPointer, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + startPointer, this->blockLength);
        this->tileNum = this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
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
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
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
    
            float inputVal1 = -1;
            float inputVal2 = 0.5;
            Muls(k3, k1, inputVal1, length);
            Exp(k3, k3, length);
            Exp(k1, k1, length);
            Add(k1, k1, k3, length);
            Muls(k1, k1, inputVal2, length);

            Div(k3, k2, k1, length);

            Cast(zLocal, k3, RoundMode::CAST_NONE, length);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            TYPE_Y inputVal1 = -1;
            TYPE_Y inputVal2 = 0.5;
            Muls(zLocal, yLocal, inputVal1, length);
            Exp(zLocal, zLocal, length);
            Exp(yLocal, yLocal, length);
            Add(yLocal, yLocal, zLocal, length);
            Muls(yLocal, yLocal, inputVal2, length);

            Div(zLocal, dyLocal, yLocal, length);
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
};
extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinhGrad<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;
    op.Init(y, dy, z, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain); 
    op.Process();
}