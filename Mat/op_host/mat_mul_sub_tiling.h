
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatMulSubTilingData)
  TILING_DATA_FIELD_DEF(int32_t, M);
  TILING_DATA_FIELD_DEF(int32_t, K);
  TILING_DATA_FIELD_DEF(int32_t, N);
  TILING_DATA_FIELD_DEF(int32_t, keep);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulSub, MatMulSubTilingData)
}
