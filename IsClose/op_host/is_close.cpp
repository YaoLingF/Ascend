
#include "is_close_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include<cassert>
namespace optiling {
     const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    IsCloseTilingData tiling;
    int32_t NUM = 24;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();
    int32_t ts=0;
    int32_t shape1[10],shape2[10];

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);

    int32_t dim1=x1_shape->GetStorageShape().GetDimNum();
    int32_t dim2=x2_shape->GetStorageShape().GetDimNum();

    if(dim1!=dim2){
        ts=1;

        
         for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) shape1[i]=x1_shape->GetStorageShape().GetDim(i);
         for (int i = 0; i < x2_shape->GetStorageShape().GetDimNum(); i++) shape2[i]=x2_shape->GetStorageShape().GetDim(i);

        sizeofdatatype = 1;
        NUM = 40;
    }else if(dt == ge::DT_FLOAT16){
        sizeofdatatype = 2;
        NUM = 15;
    }
    else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
        NUM = 10;
    }
    else{ //DT_FLOAT
        sizeofdatatype = 4;
        NUM = 6;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;//每个tile大小（单位：元素个数） 对齐32B
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t num = totalLength / block_size;//Tile个数
    uint32_t core_size = num / aivNum * block_size;//每个核处理元素个数
    uint32_t core_remain = totalLength - aivNum * core_size;
    core_remain = (core_remain + ALIGN_NUM -1) / ALIGN_NUM * ALIGN_NUM;

    // uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);//每个核处理元素的个数
    // uint32_t core_remain = totalLength - aivNum * core_size;


    float rtol = *context->GetAttrs()->GetFloat(0);
    float atol = *context->GetAttrs()->GetFloat(1);
    bool nan = *context->GetAttrs()->GetBool(2);

    tiling.set_rtol(rtol);
    tiling.set_atol(atol);
    tiling.set_nan(nan);
    tiling.set_shape(shape2);
    
    

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    tiling.set_ts(ts);
 

    context->SetBlockDim(aivNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class IsClose : public OpDef {
public:
    explicit IsClose(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("rtol").AttrType(OPTIONAL).Float(1e-05);
        this->Attr("atol").AttrType(OPTIONAL).Float(1e-08);
        this->Attr("equal_nan").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(IsClose);
}
