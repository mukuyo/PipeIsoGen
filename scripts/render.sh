#!/bin/bash
source scripts/env.sh

# Render Tee CAD templates
blenderproc run SAM-6D/Render/render_custom_templates.py --custom-blender-path $BLENDER_PATH --output_dir $OUTPUT_DIR --cad_dir $CAD_DIR --colorize True  --pipe_list $PIPE_LIST --cad_type $CAD_TYPE
