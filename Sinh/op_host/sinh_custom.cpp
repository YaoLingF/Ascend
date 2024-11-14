#include"sinh_custom_tiling.h"
#include "register/op_def_registry.h"
namespace optiling {
/**
Tiling Func负责对输入数据进行分块（Tile）处理。分块处理的好处在于，可以并行计算不同块中的数据，提升计算效率。
BLOCK_DIM 定义了每次计算操作需要处理的块的数量。
TILE_NUM 定义了在每个计算块中进一步将数据划分为更小的子块。每个子块的数据大小由blocklength/TILE_NUM来决定。
该方法将 totalLength 和 TILE_NUM 此类方法保存在tiling对象中，随后将这些信息写入`RawTilingData`中
**/
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SinhCustomTilingData tiling;
    //考生自行填充
    const uint32_t BLOCK_DIM = 8;
    const uint32_t TILE_NUM = 8;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), 
    context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}
/**
这个函数定义了输入与输出的形状推理逻辑，保证输入和输出的形状是相同的。
**/
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}
/**
该类定义了一个自定义的sinh算子，明确了输入和输出的张量格式和数据类型（DT_FLOAT16），并且指定该算子的推理形状函数是InferShape，Tiling函数是TilingFunc。
最后，通过OP_ADD(SinhCustom)将该算子注册到Ascend编译器中。
**/
namespace ops {
class SinhCustom : public OpDef {
public:
    explicit SinhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(SinhCustom);
}
