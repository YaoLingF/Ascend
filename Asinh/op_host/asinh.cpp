
// #include "asinh_tiling.h"
// #include "register/op_def_registry.h"
// #include "tiling/platform/platform_ascendc.h"
// #include <algorithm>


// namespace optiling {
//     const uint32_t BLOCK_SIZE = 32;
// static ge::graphStatus TilingFunc(gert::TilingContext* context)
// {

//     AsinhTilingData tiling;
//     uint64_t ubSize;
//     auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
//     // auto socVersion = ascendcPlatform.GetSocVersion();
//     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小
//     // auto aivNum = ascendcPlatform.GetCoreNum(); //获取当前硬件平台的核数 此平台为1
//     std::cout<<ubSize<<"\n";

//     //获取输入shape信息
//     uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
//     uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
//     uint32_t inputLength = inputBytes * inputNum; //输入长度

//     //可使用的ub空间 输入3输出1，手动考虑双缓存
//     uint32_t ubDataNumber = 4;
//     // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
//     uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber; //每个ub段可用的空间块数
//     uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes; //每次处理的数据量

//     // Input data for 32B alignment
//     uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE); //输入长度 对齐处理
//     // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
//     uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE;// 输入数据需要多少空间块    
//     //  chunks are calculated and sliced several times using the number of data on each core
//     uint32_t CoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes; //对齐空间后的输入数量
//     uint32_t TileNum = everyCoreInputBlockNum / tileBlockNum;
//     uint32_t finalTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? TileNum : TileNum + 1; //需要循环处理几次
//     // Tail block calculation for  chunks of data
//     uint32_t TailDataNum = CoreDataNum - (tileDataNum * TileNum);
//     TailDataNum = TailDataNum == 0 ? tileDataNum : TailDataNum; //最后一次需要处理的数据量

    
//     tiling.set_CoreDataNum(CoreDataNum);  //对齐空间后的输入数量
//     tiling.set_finalTileNum(finalTileNum);//需要循环处理几次
//     tiling.set_tileDataNum(tileDataNum); //每次处理的数据量
//     tiling.set_TailDataNum(TailDataNum); //最后一次需要处理的数据量

//     std::cout<<CoreDataNum<<"\n"<<finalTileNum<<"\n"<<tileDataNum<<"\n"<<TailDataNum<<"\n"<<inputNum<<"\n"<<inputBytes<<"\n";
    
    
//     context->SetBlockDim(1);
//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
//     size_t *currentWorkspace = context->GetWorkspaceSizes(1);
//     currentWorkspace[0] = 0;
//     return ge::GRAPH_SUCCESS;


// }
// }


#include "asinh_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t N = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AsinhTilingData tiling;

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
        NUM = 10;
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
class Asinh : public OpDef {
public:
    explicit Asinh(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Asinh);
}
