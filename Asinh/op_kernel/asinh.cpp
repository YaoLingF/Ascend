#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X, typename TYPE_Y>
class KernelAsinh {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAsinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        //this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, this->blockLength);
        this->tileNum = this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(B_x, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(float));
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
            
            auto float_x = B_x.Get<float>();
            Cast(float_x, x, RoundMode::CAST_NONE, length);
            auto float_y = B_y.Get<float>();
            float COEFF0 = 1.0;
            Mul(float_y, float_x, float_x, length);
            Adds(float_y, float_y, COEFF0, length);
            Sqrt(float_y, float_y, length);

            Add(float_y, float_x, float_y, length);

            Ln(float_y, float_y, length);

            Cast(y, float_y, RoundMode::CAST_NONE, length);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            assert(false);
            DTYPE_X COEFF0 = 1.0;
            Mul(y, x, x, length);
            Adds(y, y, COEFF0, length);
            Sqrt(y, y, length);

            Add(y, x, y, length);

            Ln(y, y, length);
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
    TBuf<QuePosition::VECCALC> B_x,B_y;
    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;


};




extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {


    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsinh<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();

}
