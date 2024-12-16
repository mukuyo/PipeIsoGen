#!/bin/bash
source scripts/env.sh

# Run instance segmentation model
python SAM-6D/Instance_Segmentation_Model/run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --cad_type $CAD_TYPE --img_dir $IMG_DIR --cam_path $CAMERA_PATH --pipe_list $PIPE_LIST --run_mode $RUN_MODE

# Run pose estimation model
python SAM-6D/Pose_Estimation_Model/run_inference_custom.py --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --cad_type $CAD_TYPE --img_dir $IMG_DIR --cam_path $CAMERA_PATH --pipe_list $PIPE_LIST

# Run pose estimation model
python isometric/src/generate.py --output_dir $OUTPUT_DIR --img_dir $IMG_DIR --cam_path $CAMERA_PATH --pose_dir $POSE_DIR

# Run isometric generation
python isometric/src/eval.py --gt_dxf_path $GT_DXF_PATH --pred_dxf_path $PRED_DXF_PATH