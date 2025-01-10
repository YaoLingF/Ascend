
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReplicationPad2dTilingData)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, shape);
  TILING_DATA_FIELD_DEF(int32_t, p);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReplicationPad2d, ReplicationPad2dTilingData)
}
