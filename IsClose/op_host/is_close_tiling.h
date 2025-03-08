
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IsCloseTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(float, rtol);
  TILING_DATA_FIELD_DEF(float, atol);
  TILING_DATA_FIELD_DEF(bool, nan);
  TILING_DATA_FIELD_DEF(int32_t, ts);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 10, shape);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IsClose, IsCloseTilingData)
}
