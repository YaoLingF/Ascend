
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 4, shape1);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 4, shape2);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 4, shape3);
  TILING_DATA_FIELD_DEF(int32_t, p);
  TILING_DATA_FIELD_DEF(int32_t, mode);
  TILING_DATA_FIELD_DEF(int32_t, dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}
