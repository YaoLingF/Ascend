
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSumExpTilingData)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 4, shape);
  TILING_DATA_FIELD_DEF(int32_t, ts);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSumExp, LogSumExpTilingData)
}
