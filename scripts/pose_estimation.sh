#!/bin/bash
source scripts/env.sh

# Run pose estimation model
python SAM-6D/Pose_Estimation_Model/run_inference_custom.py --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_dir $SEG_DIR
