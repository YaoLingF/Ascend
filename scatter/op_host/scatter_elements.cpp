
#include "scatter_elements_tiling.h"
#include "register/op_def_registry.h"
#include <cassert>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ScatterElementsTilingData tiling;

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    const gert::StorageShape* x3_shape = context->GetInputShape(2);

    int32_t p = x2_shape->GetStorageShape().GetDimNum();
    //assert(p<=3);

    int32_t shape1[5]={1,1,1,1,1};
    int32_t shape2[5]={1,1,1,1,1};
    int32_t shape3[5]={1,1,1,1,1};
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    {
        shape1[i]*=x1_shape->GetStorageShape().GetDim(i);
        shape2[i]*=x2_shape->GetStorageShape().GetDim(i);
        shape3[i]*=x3_shape->GetStorageShape().GetDim(i);
    }
    const char* reduction = context->GetAttrs()->GetStr(1);
    int32_t dim = *context->GetAttrs()->GetInt(0);

    int32_t mode= 1;
    if(reduction[0] == 'a'){
        mode = 1;
    }
    else if(reduction[0] == 'm'){
        mode = 3;
    }
    else{//none
        mode = 3;
    }

    tiling.set_shape1(shape1);
    tiling.set_shape2(shape2);
    tiling.set_shape3(shape3);
    tiling.set_p(p);
    tiling.set_mode(mode);
    tiling.set_dim(dim);
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
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").AttrType(OPTIONAL).Int(0);
        this->Attr("reduce").AttrType(OPTIONAL).String("none");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ScatterElements);
}
