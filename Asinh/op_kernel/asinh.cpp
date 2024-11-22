#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X, typename TYPE_Y>
class KernelAsinh {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t prenum, uint32_t sufnum, uint32_t presize, uint32_t sufsize, uint32_t block_size) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        idx = GetBlockIdx();

        this->blockLength = idx < prenum ? presize : sufsize;
        this->tileLength = block_size;

        auto startPointer = idx < prenum ? idx * presize : prenum * presize + (idx - prenum) * sufsize;
        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, this->blockLength);
        
        this->tileNum = (this->blockLength +this->tileLength - 1)/ this->tileLength;

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(B_x, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(float));

            pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));
        }
        else
        {
            pipe.InitBuffer(B_zero, this->tileLength * sizeof(float));
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
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        DataCopy(x, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<T, half>)
        {

            auto bits = B_bits.Get<uint8_t>();
            auto zero = B_zero.Get<half>();
            Duplicate(zero, half(0), length);

            Compare(bits, x, zero, CMPMODE::LT, length);


            auto float_x = B_x.Get<float>();
            Cast(float_x, x, RoundMode::CAST_NONE, length);
            auto float_y = B_y.Get<float>();
            float COEFF0 = 1.0;
            float COEFF1 = -1.0;

            Abs(float_x, float_x, length);
            Mul(float_y, float_x, float_x, length);
            Adds(float_y, float_y, COEFF0, length);
            Sqrt(float_y, float_y, length);
            Add(float_y, float_x, float_y, length);
            Ln(float_y, float_y, length);
            Muls(float_x, float_y, COEFF1, length);
            Select(float_y, bits, float_x, float_y, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);

            Cast(y, float_y, RoundMode::CAST_ROUND, length);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            auto bits = B_bits.Get<uint8_t>();
            auto zero = B_zero.Get<float>();
            Duplicate(zero, float(0), length);

            Compare(bits, x, zero, CMPMODE::LT, length);

            DTYPE_X COEFF0 = 1.0;
            DTYPE_X COEFF1 = -1.0;

            Abs(x, x, length);
            Mul(y, x, x, length);
            Adds(y, y, COEFF0, length);
            Sqrt(y, y, length);
            Add(y, y, x, length);
            Ln(y, y, length);
            Muls(x, y, COEFF1, length);

            Select(y, bits, x, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
        }
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_x,B_y,B_bits,B_zero;
    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t idx;


};




extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {


    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinh<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.prenum, tiling_data.sufnum, tiling_data.presize, tiling_data.sufsize, tiling_data.block_size);
    op.Process();

}
