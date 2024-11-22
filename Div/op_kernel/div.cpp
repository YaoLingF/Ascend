#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     

class KernelDiv {

public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t CoreDataNum, uint32_t finalTileNum, uint32_t tileDataNum, uint32_t TailDataNum) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->coreDataNum = CoreDataNum;
        this->tileNum = finalTileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = TailDataNum;

        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2*)x2, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X1*)y, this->coreDataNum);

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X2));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X1));

        if constexpr (std::is_same_v<DTYPE_X1, int8_t>)
        {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int32_t>)
        {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        }
        else if constexpr (std::is_same_v<DTYPE_X1, half>)
        {
            pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
        }

    }
    __aicore__ inline void Process() {

        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();
        DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X1> x1Local = inQueueX1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2Local = inQueueX2.DeQue<DTYPE_X2>();
        LocalTensor<DTYPE_X1> yLocal = outQueueY.AllocTensor<DTYPE_X1>();

        if constexpr (std::is_same_v<DTYPE_X1, int8_t>)
        {
            LocalTensor<half> p1 = tmp1.Get<half>();
            Cast(p1, x1Local, RoundMode::CAST_NONE, this->processDataNum);//int8转为half
            LocalTensor<half> p2 = tmp2.Get<half>();
            Cast(p2, x2Local, RoundMode::CAST_NONE, this->processDataNum);//int8转为half

            Div(p1, p1, p2, this->processDataNum);

            Cast(yLocal, p1, RoundMode::CAST_TRUNC, this->processDataNum);//half转为int8
        }
        else if constexpr (std::is_same_v<DTYPE_X1, int32_t>)
        {
            LocalTensor<float> p1 = tmp1.Get<float>();
            Cast(p1, x1Local, RoundMode::CAST_NONE, this->processDataNum);//int32转为float
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p2, x2Local, RoundMode::CAST_NONE, this->processDataNum);//int32转为float

            Div(p1, p1, p2, this->processDataNum);

            Cast(yLocal, p1, RoundMode::CAST_TRUNC, this->processDataNum);//float->int32
        }
        
        else if constexpr (std::is_same_v<DTYPE_X1, half>)
        {
            LocalTensor<float> p1 = tmp1.Get<float>();
            Cast(p1, x1Local, RoundMode::CAST_NONE, this->processDataNum);//fp16转为float
            LocalTensor<float> p2 = tmp2.Get<float>();
            Cast(p2, x2Local, RoundMode::CAST_NONE, this->processDataNum);//fp16转为float
            // LocalTensor<float> p3 = tmp2.Get<float>();
            // Duplicate(p3, float(1.0), this->processDataNum);
            Muls(p1, p1, float(10000), this->processDataNum);
            Muls(p2, p2, float(10000), this->processDataNum);
            Div(p1, p1, p2, this->processDataNum);

            // Mul(p1, p1, p3, this->processDataNum);

            Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);//float->fp16
        }
        else if constexpr (std::is_same_v<DTYPE_X1, float>)
        {
            Div(yLocal, x1Local, x2Local, this->processDataNum);
        }

        outQueueY.EnQue<DTYPE_X1>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_X1> yLocal = outQueueY.DeQue<DTYPE_X1>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1,inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3;
    GlobalTensor<DTYPE_X1> x1Gm;
    GlobalTensor<DTYPE_X2> x2Gm;
    GlobalTensor<DTYPE_X1> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};


extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelDiv op;
    op.Init(x1, x2, y, tiling_data.CoreDataNum, tiling_data.finalTileNum, tiling_data.tileDataNum, tiling_data.TailDataNum);  
    op.Process();
}