import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
# import imageio
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23
from ultralytics import YOLO

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

detection_list = []

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    for det in detections:
        if det['score'] >= 0.01:
            mask = rle_to_mask(det["segmentation"])
            edge = canny(mask)
            edge = binary_dilation(edge, np.ones((2, 2)))
            obj_id = det["category_id"]
            temp_id = obj_id - 1

            r = int(255 * colors[temp_id][0])
            g = int(255 * colors[temp_id][1])
            b = int(255 * colors[temp_id][2])
            img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
            img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
            img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
            img[edge, :] = 255

    img = Image.fromarray(np.uint8(img))
    img.save(save_path)

def visualize_all(rgb, detection_list, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    alpha = 0.33

    for detections in detection_list:
        colors = distinctipy.get_colors(len(detections))

        for j, det in enumerate(detections):
            if det['score'] >= 0.01:
                mask = rle_to_mask(det["segmentation"])
                edge = canny(mask)
                edge = binary_dilation(edge, np.ones((2, 2)))
                temp_id = j

                r = int(255 * colors[temp_id][0])
                g = int(255 * colors[temp_id][1])
                b = int(255 * colors[temp_id][2])
                img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
                img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
                img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
                img[edge, :] = 255

    img = Image.fromarray(np.uint8(img))
    img.save(save_path)

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))

    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, cad_dir, cad_type, img_dir, cam_path, pipe_lists, stability_score_thresh):
    conf_path = os.path.join("./configs/")

    with initialize(version_base=None, config_path=conf_path):
        cfg = compose(config_name='run_inference.yaml')
    
    model = []
    pipe_list = pipe_lists.split(',')
    for i, pipe_name in enumerate(pipe_list):
        os.makedirs(os.path.join(args.output_dir, "segmentation"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "segmentation", pipe_name), exist_ok=True)
        
        if segmentor_model == "sam":
            with initialize(version_base=None, config_path=conf_path+"/model"):
                cfg.model = compose(config_name='ISM_sam.yaml')
            cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        elif segmentor_model == "fastsam":
            with initialize(version_base=None, config_path=conf_path+"/model"):
                cfg.model = compose(config_name='ISM_'+pipe_name+'.yaml')
        else:
            raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))
        logging.info("Initializing model")

        model.append(instantiate(cfg.model))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model[i].descriptor_model.model = model[i].descriptor_model.model.to(device)
        model[i].descriptor_model.model.device = device
        # if there is predictor in the model, move it to device
        # if hasattr(model[i].segmentor_model, "predictor"):
        #     model[i].segmentor_model.predictor.model = (
        #         model[i].segmentor_model.predictor.model.to(device)
        #     )
        # else:
        #     model[i].segmentor_model.model.setup_model(device=device, verbose=True)
        logging.info(f"Moving models to {device} done!")
            
        
        logging.info("Initializing template")
        template_dir = os.path.join(output_dir, f"render/{pipe_name}")

        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

        model[i].ref_data = {}
        model[i].ref_data["descriptors"] = model[i].descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        model[i].ref_data["appe_descriptors"] = model[i].descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
    
    save_path = os.path.join(output_dir, "segmentation", "all")
    os.makedirs(save_path, exist_ok=True)
    
    print(img_dir)
    for _img_num, _ in enumerate(tqdm(glob.glob(img_dir + "/rgb/*.png"))):
        # run inference
        img_num = str(_img_num*10)
        detection_list = []
        for i, pipe_name in enumerate(pipe_list):
            os.makedirs(os.path.join(output_dir, "segmentation", pipe_name, img_num), exist_ok=True)

            rgb_path = os.path.join(img_dir, "rgb", "frame"+img_num+'.png')
            depth_path = os.path.join(img_dir, "depth", "frame"+img_num+'.png')

            rgb = Image.open(rgb_path).convert("RGB")

            detections = model[i].segmentor_model.generate_masks(np.array(rgb))
            if detections is None:
                continue
            
            detections = Detections(detections)

            query_decriptors, query_appe_descriptors = model[i].descriptor_model.forward(np.array(rgb), detections)
            # matching descriptors
            (
                idx_selected_proposals,
                pred_idx_objects,
                semantic_score,
                best_template,
            ) = model[i].compute_semantic_score(query_decriptors)
            # update detections
            detections.filter(idx_selected_proposals)
            query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]
            # compute the appearance score
            appe_scores, ref_aux_descriptor= model[i].compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)
            # compute the geometric score
            batch = batch_input_data(depth_path, cam_path, device)
            template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(torch.float32).to(device)
            model[i].ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

            if cad_type == 'None':
                mesh = trimesh.load_mesh(os.path.join(cad_dir, pipe_name+'.ply'))
            else:
                mesh = trimesh.load_mesh(os.path.join(cad_dir, pipe_name+'-'+cad_type+'.ply'))
            model_points = mesh.sample(2048).astype(np.float32) / 1000.0
            model[i].ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
            image_uv = model[i].project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

            geometric_score, visible_ratio = model[i].compute_geometric_score(
                image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model[i].visible_thred
                )

            # final score
            final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

            detections.add_attribute("scores", final_score)
            detections.add_attribute("object_ids", torch.zeros_like(final_score))   
                
            detections.to_numpy()

            save_path = f"{output_dir}/segmentation/{pipe_name}/{img_num}"
            detections.save_to_file(0, 0, 0, os.path.join(save_path, "detection_ism"), "Custom", return_results=False)
            detections = convert_npz_to_json(idx=0, list_npz_paths=[os.path.join(save_path, "detection_ism")+".npz"])
            save_json_bop23(os.path.join(save_path, "detection_ism.json"), detections)

            visualize(rgb, detections, os.path.join(save_path, "vis_ism.png"))
            
            detection_list.append(detections)
        
        rgb = Image.open(rgb_path).convert("RGB")
        visualize_all(rgb, detection_list, os.path.join(os.path.join(output_dir, "segmentation", "all", f"vis_all_ism_{img_num}.png")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_dir", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--cad_type", default='', help="The type of CAD model")
    parser.add_argument("--img_dir", nargs="?", help="Path to RGB image")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--run_mode", default='predict', help="The run mode of the model")
    parser.add_argument("--pipe_list", nargs="?", help="The target pipe names")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()

    run_inference(
        args.segmentor_model, args.output_dir, args.cad_dir, args.cad_type, args.img_dir, args.cam_path, args.pipe_list,
        stability_score_thresh=args.stability_score_thresh, 
    )
