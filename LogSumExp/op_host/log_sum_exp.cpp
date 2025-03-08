
#include "log_sum_exp_tiling.h"
#include "register/op_def_registry.h"
#include<cassert>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    LogSumExpTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    auto dt = context->GetInputTensor(0)->GetDataType();
    

    int32_t shape[4]={1,1,1,1};
    const gert::TypedContinuousVector<int64_t> attr0 = *context->GetAttrs()->GetListInt(0);//1 

    bool keep = *context->GetAttrs()->GetBool(1);

    auto x = attr0.GetSize();
    auto y = attr0.GetData();

    const int64_t *z = reinterpret_cast<const int64_t *>(attr0.GetData()+1);

    std::cout<<x<<" "<<(*y)<<" "<<*z<<" "<<*(z+1)<<"\n";
    std::cout << "y address: " << y << "\n";
    std::cout << "z address: " << z << "\n";
    std::cout << "z[0]: " << z[0] << "\n";
    std::cout << "z[1]: " << z[1] << "\n";
    
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    {
        
        if(i<*y) shape[0]*=x1_shape->GetStorageShape().GetDim(i);
        else if(i==*y) shape[1]*=x1_shape->GetStorageShape().GetDim(i);
        else shape[2]*=x1_shape->GetStorageShape().GetDim(i);
        printf("%ld\n",x1_shape->GetStorageShape().GetDim(i));
    }
    
    int32_t ts=0;


    int32_t corenum=1;

    if(x==2)
    {
       
            ts=1;

    }
    

    tiling.set_shape(shape);
    tiling.set_ts(ts);
    context->SetBlockDim(1);
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
class LogSumExp : public OpDef {
public:
    explicit LogSumExp(const char* name) : OpDef(name)
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
        this->Attr("dim").AttrType(OPTIONAL).ListInt({0});
        this->Attr("keep_dim").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(LogSumExp);
}
