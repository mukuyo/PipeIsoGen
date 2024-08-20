#!/bin/bash
source scripts/env.sh

# Run pose estimation model
python isometric/src/generate.py --output_dir $OUTPUT_DIR --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --pose_dir $POSE_DIR
