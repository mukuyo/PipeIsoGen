# env.sh

export MODE=sim
export PIPE=test

export CAD_DIR=data/model
export RGB_PATH=data/$MODE/images/$PIPE/rgb.png
export DEPTH_PATH=data/$MODE/images/$PIPE/depth.png
export CAMERA_PATH=data/$MODE/camera.json
export OUTPUT_DIR=data/outputs

# Render env
export BLENDER_PATH=SAM-6D/Render/bin/blender-3.3.21-linux-x64

# Prediction env
export SEGMENTOR_MODEL=fastsam
export SEG_DIR=$OUTPUT_DIR/segmentation/
export POSE_DIR=$OUTPUT_DIR/pose/
