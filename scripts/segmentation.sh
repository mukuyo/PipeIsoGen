#!/bin/bash
source scripts/env.sh

# Run instance segmentation model
python SAM-6D/Instance_Segmentation_Model/run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH
