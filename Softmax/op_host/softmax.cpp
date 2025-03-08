
#include "softmax_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SoftmaxTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t shape[3]={1,1,1};
    int32_t dim = x1_shape->GetStorageShape().GetDimNum();
    int32_t keep = *context->GetAttrs()->GetInt(0);

    if(keep==-1)
    {
        keep=dim-1;
    }
    
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    {
        
        if(i<keep) shape[0]*=x1_shape->GetStorageShape().GetDim(i);
        else if(i==keep) shape[1]*=x1_shape->GetStorageShape().GetDim(i);
        else shape[2]*=x1_shape->GetStorageShape().GetDim(i);
        printf("%ld\n",x1_shape->GetStorageShape().GetDim(i));
    }
    printf("%ld\n",x1_shape->GetStorageShape().GetDimNum());

    printf("%d %d %d %d %d\n",dim,keep,shape[0],shape[1],shape[2]);

    tiling.set_shape(shape);
    tiling.set_dim(dim);
    tiling.set_keep(keep);
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
class Softmax : public OpDef {
public:
    explicit Softmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Softmax);
}
