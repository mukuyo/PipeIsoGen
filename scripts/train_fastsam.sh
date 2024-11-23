#!/bin/bash
source scripts/env.sh

export TARGET_PIPE="elbow"

# Run instance segmentation model
python SAM-6D/Instance_Segmentation_Model/train_fastsam.py --output_dir $OUTPUT_DIR --target $TARGET_PIPE --epoch 1000 --batch_size 8 --save_period 100
