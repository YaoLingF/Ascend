
#include "asinh_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t N = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    AsinhGradTilingData tiling;
    int32_t NUM = 10;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_FLOAT16){
        sizeofdatatype = 2;
        NUM = 12;
    }
    else{
        sizeofdatatype = 4;
        NUM = 6;
    }
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;//每个32B块可容纳元素个数
    totalLength = (totalLength * sizeofdatatype + 31) / 32;//32B个数

    uint32_t pre32B = (totalLength / N) + (totalLength % N == 0 ? 0 : 1);
    uint32_t suf32B = totalLength / N;

    uint32_t prenum = totalLength % N;
    uint32_t sufnum = N - prenum;

    uint32_t presize = pre32B * ALIGN_NUM;
    uint32_t sufsize = suf32B * ALIGN_NUM;

    uint32_t tiling_size = ub_size / BLOCK_SIZE / NUM;//每个tile多少个32B
    uint32_t block_size = tiling_size * ALIGN_NUM;//每个tile大小(单位：元素个数)


    tiling.set_prenum(prenum);
    tiling.set_sufnum(sufnum);
    tiling.set_presize(presize);
    tiling.set_sufsize(sufsize);
    tiling.set_block_size(block_size);

    std::cout<<aivNum<<" "<<prenum<<" "<<sufnum<<" "<<presize<<" "<<sufsize<<" "<<block_size<<"hh\n";
    context->SetBlockDim(N);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
class AsinhGrad : public OpDef {
public:
    explicit AsinhGrad(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
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

OP_ADD(AsinhGrad);
}
