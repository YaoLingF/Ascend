#include "kernel_operator.h"

extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}