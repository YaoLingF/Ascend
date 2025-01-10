
#include "arg_max_with_value_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

   ArgMaxWithValueTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t shape[3]={1,1,1};
    int32_t dim = x1_shape->GetStorageShape().GetDimNum();
    int32_t keep = *context->GetAttrs()->GetInt(0);
    printf("%d %d\n",dim,keep);

    auto dt = context->GetInputTensor(0)->GetDataType();
    int32_t siz=0;
    if(dt == ge::DT_FLOAT16){
        siz = 2;
    }
    else if(dt == ge::DT_UINT8)
    {
        siz = 1;
    }
    else siz = 4;
    int32_t num = 64/siz;
    
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    {
        
        if(i<keep) shape[0]*=x1_shape->GetStorageShape().GetDim(i);
        else if(i==keep) shape[1]*=x1_shape->GetStorageShape().GetDim(i);
        else shape[2]*=x1_shape->GetStorageShape().GetDim(i);
        printf("%ld\n",x1_shape->GetStorageShape().GetDim(i));
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    printf("%d ",aivNum);

    int maxn = 100;
    
    int sum = shape[0]*shape[2];
    sum = (sum+num-1)/num;


    int32_t precal = (sum+maxn-1)/maxn;
    int32_t sufcal = sum/maxn;

    int32_t prenum = sum%maxn;
    int32_t sufnum = maxn-prenum;

    precal*=num;
    sufcal*=num;

    std::cout<<maxn<<" "<<precal<<" "<<sufcal<<" "<<prenum<<" "<<sufnum<<" "<<shape[0]<<" "<<shape[1]<<" "<<shape[2]<<"\n";


    tiling.set_shape(shape);
    tiling.set_dim(dim);
    tiling.set_keep(keep);
    tiling.set_precal(precal);
    tiling.set_sufcal(sufcal);
    tiling.set_prenum(prenum);
    tiling.set_sufnum(sufnum);
    context->SetBlockDim(maxn);
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
class ArgMaxWithValue : public OpDef {
public:
    explicit ArgMaxWithValue(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indice")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dimension").Int();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ArgMaxWithValue);
}
