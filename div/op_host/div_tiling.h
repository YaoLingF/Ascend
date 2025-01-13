
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivTilingData)
  TILING_DATA_FIELD_DEF(int32_t, size);
  TILING_DATA_FIELD_DEF(int32_t, ts);

  TILING_DATA_FIELD_DEF(uint32_t, CoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, TailDataNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Div, DivTilingData)
}
