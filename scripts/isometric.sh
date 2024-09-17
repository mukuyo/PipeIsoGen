#!/bin/bash
source scripts/env.sh

# Run pose estimation model
python isometric/src/generate.py --output_dir $OUTPUT_DIR --img_dir $IMG_DIR --cam_path $CAMERA_PATH --pose_dir $POSE_DIR
