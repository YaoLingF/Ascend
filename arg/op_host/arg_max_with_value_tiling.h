
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, shape);
  TILING_DATA_FIELD_DEF(int32_t, dim);
  TILING_DATA_FIELD_DEF(int32_t, keep);
  TILING_DATA_FIELD_DEF(int32_t, precal);
  TILING_DATA_FIELD_DEF(int32_t, sufcal);
  TILING_DATA_FIELD_DEF(int32_t, prenum);
  TILING_DATA_FIELD_DEF(int32_t, sufnum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
}
