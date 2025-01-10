
#include "replication_pad2d_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ReplicationPad2dTilingData tiling;
    
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    const gert::StorageShape* x2_shape = context->GetInputShape(1);

    int32_t p = x2_shape->GetStorageShape().GetDim(0)/2;
    int32_t sum = x1_shape->GetStorageShape().GetDimNum();
    int32_t shape[3]={1,1,1};
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    {
        
        if(i<sum-p) shape[0]*=x1_shape->GetStorageShape().GetDim(i);
        else if(i==sum-p) shape[1]*=x1_shape->GetStorageShape().GetDim(i);
        else shape[2]*=x1_shape->GetStorageShape().GetDim(i);
        printf("%ld\n",x1_shape->GetStorageShape().GetDim(i));
    }

    tiling.set_shape(shape);
    tiling.set_p(p);
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
class ReplicationPad2d : public OpDef {
public:
    explicit ReplicationPad2d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ReplicationPad2d);
}
