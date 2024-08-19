import gorilla
import argparse
import os
import sys
from PIL import Image
import os.path as osp
import numpy as np
import random
import importlib
import json

import torch
import torchvision.transforms as transforms
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

def display_rgb_image_cv(rgb_image):
    cv2.imshow('RGB Image', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")
    # pem
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="path to pretrain model")
    parser.add_argument("--model",
                        type=str,
                        default="pose_estimation_model",
                        help="path to model file")
    parser.add_argument("--config",
                        type=str,
                        default="./SAM-6D/Pose_Estimation_Model/config/base.yaml",
                        help="path to config file, different config.yaml use different config")
    parser.add_argument("--iter",
                        type=int,
                        default=600000,
                        help="epoch num. for testing")
    parser.add_argument("--exp_id",
                        type=int,
                        default=0,
                        help="")
    
    # input
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_dir", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--seg_dir", nargs="?", help="Path to segmentation information(generated by ISM)")
    parser.add_argument("--det_score_thresh", default=0.5, help="The score threshold of detection")
    
    args_cfg = parser.parse_args()

    return args_cfg

def init(obj_name):
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_iter = args.iter

    cfg.output_dir = args.output_dir
    cfg.cad_path = os.path.join(args.cad_dir, obj_name+'.ply')
    cfg.rgb_path = args.rgb_path
    cfg.depth_path = args.depth_path
    cfg.cam_path = args.cam_path
    cfg.seg_path = os.path.join(args.seg_dir, obj_name+'/detection_ism.json')

    cfg.det_score_thresh = args.det_score_thresh
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return cfg



from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from draw_utils import draw_detections
import pycocotools.mask as cocomask
import trimesh

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz


def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose


def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg):
    dets = []
    with open(seg_path) as f:
        dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation
    for det in dets_:
        if det['score'] > det_score_thresh:
            dets.append(det)
    del dets_
    
    cam_info = json.load(open(cam_path))
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info['depth_scale'] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)
    
    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    print(len(dets))

    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        print(np.sum(flag), inst['scene_id'])
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:   
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        # display_rgb_image_cv(rgb)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    print(all_score)
    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets



if __name__ == "__main__":

    img_list = []
    pred_rot_list = []
    pred_trans_list = []
    model_points_list = []
    K_list = []

    for obj_name in ['tee', 'elbow']:
        cfg = init(obj_name)

        print("=> creating model ...")
        MODEL = importlib.import_module(cfg.model_name)
        model = MODEL.Net(cfg.model)
        model = model.cuda()
        model.eval()
        checkpoint = os.path.join("./SAM-6D/Pose_Estimation_Model/", 'checkpoints', 'sam-6d-pem-base.pth')
        
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

        print("=> extracting templates ...")
        tem_path = os.path.join(cfg.output_dir, 'render/'+obj_name)
        all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, cfg.test_dataset)
        with torch.no_grad():
            all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

        print("=> loading input data ...")
        input_data, img, whole_pts, model_points, detections = get_test_data(
            cfg.rgb_path, cfg.depth_path, cfg.cam_path, cfg.cad_path, cfg.seg_path, 
            cfg.det_score_thresh, cfg.test_dataset
        )
        ninstance = input_data['pts'].size(0)
        
        print("=> running model ...")
        with torch.no_grad():
            input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
            input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
            out = model(input_data)

        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

        print("=> saving results ...")
        os.makedirs(f"{cfg.output_dir}/sam6d_results/{obj_name}", exist_ok=True)
        for idx, det in enumerate(detections):
            detections[idx]['score'] = float(pose_scores[idx])
            detections[idx]['R'] = list(pred_rot[idx].tolist())
            detections[idx]['t'] = list(pred_trans[idx].tolist())

        with open(os.path.join(f"{cfg.output_dir}/sam6d_results/{obj_name}", 'detection_pem.json'), "w") as f:
            json.dump(detections, f)

        print("=> visualizating ...")
        save_path = os.path.join(f"{cfg.output_dir}/sam6d_results/{obj_name}", 'vis_pem.png')
        valid_masks = pose_scores == pose_scores.max()
        K = input_data['K'].detach().cpu().numpy()
        vis_img = visualize(img, pred_rot, pred_trans, model_points*1000, K, save_path)
        vis_img.save(save_path)
        
        img_list.append(img)
        pred_rot_list.append(pred_rot)
        pred_trans_list.append(pred_trans)
        model_points_list.append(model_points)
        K_list.append(K)
        
    for i in range(2):
        save_path = os.path.join(f"{cfg.output_dir}/sam6d_results", 'multi.png')
        
        vis_img = visualize(img, pred_rot, pred_trans, model_points*1000, K, save_path)
        # vis_img.save(save_path)