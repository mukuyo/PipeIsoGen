#!/bin/bash
source scripts/env.sh

# Run instance segmentation model
python SAM-6D/Instance_Segmentation_Model/run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --cad_type $CAD_TYPE --img_dir $IMG_DIR --cam_path $CAMERA_PATH --pipe_list $PIPE_LIST --run_mode $RUN_MODE
