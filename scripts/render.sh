#!/bin/bash
source scripts/env.sh

# Render Tee CAD templates
blenderproc run SAM-6D/Render/render_custom_templates.py --custom-blender-path $BLENDER_PATH --output_dir $OUTPUT_DIR --cad_path $TEE_CAD_PATH --obj_name "tee" --colorize True  
# Render Elbow CAD templates
blenderproc run SAM-6D/Render/render_custom_templates.py --custom-blender-path $BLENDER_PATH --output_dir $OUTPUT_DIR --cad_path $ELBOW_CAD_PATH --obj_name "elbow" --colorize True  
