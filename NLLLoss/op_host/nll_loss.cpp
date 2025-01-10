
#include "nll_loss_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t N = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  NLLLossTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz[3]={1,1,1};
  
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
  {
    data_sz[std::min(2,i)] = x1_shape->GetStorageShape().GetDim(i);
    printf("%ld\n",x1_shape->GetStorageShape().GetDim(i));
  }
  printf("%ld\n",x1_shape->GetStorageShape().GetDimNum());

const char* reduction = context->GetAttrs()->GetStr(0);
    int32_t ignore = *context->GetAttrs()->GetInt(1);

    if(x1_shape->GetStorageShape().GetDimNum() == 1)
    {
        data_sz[1]=data_sz[0];
        data_sz[0]=1;
    }
    assert(ignore == -100);

    int32_t mode= 1;
    if(reduction[0] == 'n'){
        mode = 1;
    }
    else if(reduction[0] == 'm'){
        mode = 2;
    }
    else{//sum
        mode = 3;
    }
    printf("%d\n",mode);
    auto dt = context->GetInputTensor(0)->GetDataType();

    tiling.set_shape(data_sz);
    tiling.set_mode(mode);
    tiling.set_ignore(ignore);
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
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
class NLLLoss : public OpDef {
public:
    explicit NLLLoss(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(NLLLoss);
}
