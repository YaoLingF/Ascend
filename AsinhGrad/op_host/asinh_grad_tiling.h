
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsinhGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, prenum);
  TILING_DATA_FIELD_DEF(uint32_t, sufnum);
  TILING_DATA_FIELD_DEF(uint32_t, presize);
  TILING_DATA_FIELD_DEF(uint32_t, sufsize);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsinhGrad, AsinhGradTilingData)
}
