
#include "non_max_suppression_tiling.h"
#include "register/op_def_registry.h"
#include<cassert>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  NonMaxSuppressionTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(1);

  int32_t type = *context->GetAttrs()->GetInt(0);//1 

  int32_t batch = x1_shape->GetStorageShape().GetDim(0);

  int32_t classes = x1_shape->GetStorageShape().GetDim(1);
  
  int32_t num = x1_shape->GetStorageShape().GetDim(2);;

  assert(type==0);


  
  
    //assert(num==1024);
  

  std::cout<<type<<" "<<batch<<" "<<classes<<" "<<num<<"\n";

  tiling.set_type(type);
  tiling.set_batch(batch);
  tiling.set_classes(classes);
  tiling.set_num(num);
  
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
class NonMaxSuppression : public OpDef {
public:
    explicit NonMaxSuppression(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("max_output_boxes_per_class")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("iou_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("score_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("selected_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("center_point_box").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(NonMaxSuppression);
}
