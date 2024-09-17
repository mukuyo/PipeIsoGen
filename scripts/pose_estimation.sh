#!/bin/bash
source scripts/env.sh

# Run pose estimation model
python SAM-6D/Pose_Estimation_Model/run_inference_custom.py --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --cad_type $CAD_TYPE --img_dir $IMG_DIR --cam_path $CAMERA_PATH --pipe_list $PIPE_LIST
