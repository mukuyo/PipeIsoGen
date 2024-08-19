#!/bin/bash
source scripts/env.sh

# Run pose estimation model
python SAM-6D/Pose_Estimation_Model/run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $TEE_CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $TEE_SEG_PATH --obj_name "tee"
python SAM-6D/Pose_Estimation_Model/run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $ELBOW_CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $ELBOW_SEG_PATH --obj_name "elbow"
