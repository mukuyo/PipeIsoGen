import json
import os
import cv2
import numpy as np
from labelme import utils

def create_mask_from_labelme_json(json_path, output_mask_path):
    with open(json_path) as f:
        data = json.load(f)
    
    image_data = data.get("imageData")
    if image_data:
        image = utils.img_b64_to_arr(image_data)
    else:
        image_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
        image = cv2.imread(image_path)
    
    shapes = data["shapes"]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    cv2.imwrite(output_mask_path, mask)

# Example usage:
json_dir = "data/mask/label"
output_mask_dir = "data/mask/masks"

os.makedirs(output_mask_dir, exist_ok=True)

for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_dir, json_file)
        mask_path = os.path.join(output_mask_dir, json_file.replace(".json", ".png"))
        create_mask_from_labelme_json(json_path, mask_path)