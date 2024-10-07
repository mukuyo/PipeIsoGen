# env.sh

export IMAGE=sim
export RUN_MODE=eval
export PIPE_MODEL=test
export PIPE_LIST="elbow","tee"
export CAD_TYPE="center"

export CAD_DIR=data/cad_model
export IMG_DIR=data/$IMAGE/images/$PIPE_MODEL
export CAMERA_PATH=data/$IMAGE/camera.json
export OUTPUT_DIR=data/outputs

# Render env
export BLENDER_PATH=SAM-6D/Render/bin/blender-3.3.21-linux-x64

# Prediction env
export SEGMENTOR_MODEL=fastsam
