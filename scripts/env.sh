# env.sh

export MODE=sim

export CAD_DIR=data/$MODE/model
export RGB_PATH=data/$MODE/rgb/rgb.png
export DEPTH_PATH=data/$MODE/depth/depth.png
export CAMERA_PATH=data/$MODE/camera.json
export OUTPUT_DIR=data/outputs

# Render env
export BLENDER_PATH=SAM-6D/Render/bin/blender-3.3.21-linux-x64

# Segmentaion env
export SEGMENTOR_MODEL=fastsam

# Pose estimation env
export TEE_SEG_PATH=$OUTPUT_DIR/segmentation/tee/detection_ism.json
export ELBOW_SEG_PATH=$OUTPUT_DIR/segmentation/elbow/detection_ism.json
