#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
class KernelIsClose {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelIsClose() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, float rtol, float atol, bool nan) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->rtol = rtol;
        this->atol = atol;
        this->nan = nan;

        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        //this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, this->blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, this->blockLength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_result, this->tileLength * sizeof(half));
        pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));
        this->zero = B_zero.Get<half>();
        Duplicate(this->zero, half(0), this->tileLength);
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
        }
        else if constexpr (std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
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
        //CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty = y.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, uint8_t>) 
        {
            assert(nan==false);
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Sub(float_x1,float_x1,float_x2,length);
            Abs(float_x1,float_x1,length);
            Abs(float_x2,float_x2,length);
            Muls(float_x2,float_x2,half(rtol),length);
            Adds(float_x2,float_x2,half(atol),length);
            Compare(bits, float_x1, float_x2, CMPMODE::GT, length);
        }
        else if constexpr (std::is_same_v<T, int32_t>) 
        {
                auto val = B_x1.Get<float>();
                auto float_zero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, length);
                Cast(float_zero, x2, RoundMode::CAST_NONE, length);
                Sub(val,val,float_zero,length);
                Abs(val,val,length);
                Abs(float_zero,float_zero,length);
                Muls(float_zero,float_zero,rtol,length);
                Adds(float_zero,float_zero,atol,length);
                Compare(bits, val, float_zero, CMPMODE::GT, length);
        }
        else if constexpr (std::is_same_v<T, float>) 
        {
                Sub(x1,x1,x2,length);
                Abs(x1,x1,length);
                Abs(x2,x2,length);
                Muls(x2,x2,TYPE_X1(rtol),length);
                Adds(x2,x2,TYPE_X1(atol),length);
                Compare(bits, x1, x2, CMPMODE::GT, length);
        }
        else if constexpr (std::is_same_v<T, half>)
        { //half
                Sub(x1,x1,x2,length);
                Abs(x1,x1,length);
                Abs(x2,x2,length);
                Muls(x2,x2,TYPE_X1(rtol),length);
                Adds(x2,x2,TYPE_X1(atol),length);
                Compare(bits, x1, x2, CMPMODE::GT, length);
        }
        Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Cast(inty, result, RoundMode::CAST_ROUND, length);
        
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    LocalTensor<half> zero;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;

    float rtol;
    float atol;
    bool nan;

};
extern "C" __global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelIsClose<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.rtol, tiling_data.atol, tiling_data.nan);
    op.Process();
}