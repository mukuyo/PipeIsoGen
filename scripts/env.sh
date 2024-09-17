# env.sh

export IMAGE=real
export RUN_MODE=eval
export PIPE_MODEL=test
export PIPE_LIST="elbow","tee"
export CAD_TYPE="center-big"

export CAD_DIR=data/cad_model
export IMG_DIR=data/$IMAGE/images/$PIPE_MODEL
export RGB_PATH=data/$IMAGE/images/$PIPE_MODEL/rgb.png
export DEPTH_PATH=data/$IMAGE/images/$PIPE_MODEL/depth.png
export CAMERA_PATH=data/$IMAGE/camera.json
export OUTPUT_DIR=data/outputs

# Render env
export BLENDER_PATH=SAM-6D/Render/bin/blender-3.3.21-linux-x64

# Prediction env
export SEGMENTOR_MODEL=fastsam
