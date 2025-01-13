
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NonMaxSuppressionTilingData)
  TILING_DATA_FIELD_DEF(int32_t, type);
  TILING_DATA_FIELD_DEF(int32_t, batch);
  TILING_DATA_FIELD_DEF(int32_t, classes);
  TILING_DATA_FIELD_DEF(int32_t, num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonMaxSuppression, NonMaxSuppressionTilingData)
}
