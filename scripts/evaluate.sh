#!/bin/bash
source scripts/env.sh

# Run isometric generation
python isometric/src/eval.py --gt_dxf_path $GT_DXF_PATH --pred_dxf_path $PRED_DXF_PATH