
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxTilingData)
  TILING_DATA_FIELD_DEF_ARR(int32_t, 3, shape);
  TILING_DATA_FIELD_DEF(int32_t, dim);
  TILING_DATA_FIELD_DEF(int32_t, keep);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Softmax, SoftmaxTilingData)
}
