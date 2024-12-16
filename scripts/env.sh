# env.sh

export IMAGE=sim
export RUN_MODE=eval

# PipeV1, PipeV2, test 
export PIPE_MODEL=PipeV1

# Ideal, Azure, L515, D435
export SENSOR_TYPE=D435

export PIPE_LIST="elbow","tee"
export CAD_TYPE="None"

export CAD_DIR=data/cad_model

if [ "$PIPE_MODEL" != "test" ]; then
  export IMG_DIR=data/$IMAGE/$PIPE_MODEL/$SENSOR_TYPE
else
  export IMG_DIR=data/$IMAGE/$PIPE_MODEL
fi

export CAMERA_PATH=data/$IMAGE/camera.json
export OUTPUT_DIR=data/outputs

# Render env
export BLENDER_PATH=SAM-6D/Render/bin/blender-3.3.21-linux-x64

# Prediction env
export SEGMENTOR_MODEL=fastsam

# Evaluation env
export GT_DXF_PATH=data/$IMAGE/$PIPE_MODEL/GT/gt.dxf
export PRED_DXF_PATH=data/outputs/isometric/pipe_no_size.dxf
