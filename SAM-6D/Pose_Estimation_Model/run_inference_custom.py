import gorilla
import argparse
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import os.path as osp
import numpy as np
import random
import importlib
import json
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
from math import sin, cos, degrees, radians
from plyfile import PlyData, PlyElement
import open3d as o3d

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
    parser.add_argument("--cad_type", default="", help="The type of CAD model")
    parser.add_argument("--img_dir", nargs="?", help="Path to image")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--pipe_list", default="", help="The list of pipe names")
    parser.add_argument("--det_score_thresh", default=0.05, help="The score threshold of detection")
    
    args_cfg = parser.parse_args()

    return args_cfg

def init():
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
    cfg.cam_path = args.cam_path

    cfg.det_score_thresh = args.det_score_thresh
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return cfg, args.pipe_list.split(','), args.img_dir, args.cad_dir, args.cad_type

from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from draw_utils import draw_detections, draw_detections_all
import pycocotools.mask as cocomask
import trimesh

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

def visualize(rgb, pred_rot, pred_trans, model_points, K, save_dir, id, gt_poses):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, gt_poses, color=(255, 0, 0), save_dir=save_dir)
    img = Image.fromarray(np.uint8(img))
    img.save(save_dir + f'/{id}/{id}.png')

def visualize_all(rgb, pred_rot_list, pred_trans_list, model_points_list, K_list, save_dir, object_list, img_num, gt_pose_list):
    pcl_img, gt_img = draw_detections_all(rgb, pred_rot_list, pred_trans_list, model_points_list, K_list, save_dir, object_list, img_num, gt_pose_list)
    pcl_img = Image.fromarray(np.uint8(pcl_img))
    gt_img = Image.fromarray(np.uint8(gt_img))
    pcl_img.save(os.path.join(save_dir, f'all/vis_pcl_all_{img_num}.png'))
    gt_img.save(os.path.join(save_dir, f'all/vis_gt_all_{img_num}.png'))

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

def project_points(points, rotation, translation, camera_matrix):
    """3Dポイントを2Dへ射影"""
    points_3d = np.dot(points, rotation.T) + translation.transpose()
    points_2d = points_3d[:, :2] / points_3d[:, 2][:, np.newaxis]  # z軸で正規化
    return np.dot(points_2d, camera_matrix[:2, :2]) + camera_matrix[:2, 2]

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

def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg, model_points):
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
    
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']

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

        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]

        flag = np.zeros(4)
        loop_count = 0
        while np.sum(flag) < 40:
            flag = np.linalg.norm(tmp_cloud, axis=1) < (radius * (1.8 + loop_count * 1))
            loop_count += 1.0

        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:   
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets, radius

def transform_points(pts, pose):
    """点群に姿勢行列を適用"""
    R, t = pose[:, :3], pose[:, 3]
    if len(pts.shape)==1:
        return (R @ pts[:,None] + t[:,None])[:,0]
    return pts @ R.T + t[None,:]

def transform_points(temp_gt_rots, k):
    if k == 1:
        temp_gt_rots[:, 0] = -temp_gt_rots[:, 0]
        temp_gt_rots[:, 2] = -temp_gt_rots[:, 2]
    elif k == 2:
        temp_gt_rots[:, 0] = -temp_gt_rots[:, 0]
        temp_gt_rots[:, 1] = -temp_gt_rots[:, 1]
    elif k == 3:
        temp_gt_rots[:, 1] = -temp_gt_rots[:, 1]
        temp_gt_rots[:, 2] = -temp_gt_rots[:, 2]
    elif k == 4:
        temp_gt_rots = temp_gt_rots[:, [1, 2, 0]]
        temp_gt_rots[1, 0] = -temp_gt_rots[1, 0]
        temp_gt_rots[2, 0] = -temp_gt_rots[2, 0]
    elif k == 5:
        temp_gt_rots = temp_gt_rots[:, [1, 2, 0]]
        temp_gt_rots[1, 0] = -temp_gt_rots[1, 0]
        temp_gt_rots[2, 0] = -temp_gt_rots[2, 0]
        temp_gt_rots[0, 2] = -temp_gt_rots[0, 2]
    elif k == 6:
        temp_gt_rots = temp_gt_rots[:, [1, 2, 0]]
        temp_gt_rots[1, 0] = -temp_gt_rots[1, 0]
        temp_gt_rots[2, 0] = -temp_gt_rots[2, 0]
        temp_gt_rots[0, 2] = -temp_gt_rots[0, 2]
        temp_gt_rots[:, 0] = -temp_gt_rots[:, 0]
        temp_gt_rots[:, 2] = -temp_gt_rots[:, 2]
    elif k == 7:
        temp_gt_rots = temp_gt_rots[:, [2, 0, 1]]
        temp_gt_rots[1, 0] = -temp_gt_rots[1, 0]
        temp_gt_rots[2, 0] = -temp_gt_rots[2, 0]
        temp_gt_rots[0, 1] = -temp_gt_rots[0, 1]
    elif k == 8:
        temp_gt_rots = temp_gt_rots[:, [0, 2, 1]]
        temp_gt_rots[:, 0] = -temp_gt_rots[:, 0]
    temp_gt_poses[:, :3] = temp_gt_rots

    return temp_gt_poses
    
if __name__ == "__main__":

    cfg, pipe_list, img_dir, cad_dir, cad_type = init()
    
    np.random.seed(cfg.rd_seed)
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    print("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    model = model.cuda()
    model.eval()
    checkpoint = os.path.join("./SAM-6D/Pose_Estimation_Model/", 'checkpoints', 'sam-6d-pem-base.pth')
    
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    os.makedirs(f"{cfg.output_dir}/pose/all", exist_ok=True)

    print("=> extracting templates ...")
    all_tem_pts = []
    all_tem_feat = []
    gt_pose_list = []
    model_point_list = []
    for i, pipe_name in enumerate(pipe_list):
        gt_pose_list.append([])

    for i, pipe_name in enumerate(pipe_list):
        tem_path = os.path.join(cfg.output_dir, 'render', pipe_name)
        all_tem, all_tem_pt, all_tem_choose = get_templates(tem_path, cfg.test_dataset)
        with torch.no_grad():
            _all_tem_pts, _all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pt, all_tem_choose)
            all_tem_pts.append(_all_tem_pts)
            all_tem_feat.append(_all_tem_feat)
            if cad_type == 'None':
                mesh, _ = trimesh.sample.sample_surface(mesh=trimesh.load_mesh(os.path.join(cad_dir, pipe_name+'.ply')), count=cfg.test_dataset.n_sample_model_point, seed=120)
            else:
                mesh, _ = trimesh.sample.sample_surface(mesh=trimesh.load_mesh(os.path.join(cad_dir, pipe_name+'-'+cad_type+'.ply')), count=cfg.test_dataset.n_sample_model_point, seed=120)
            model_point_list.append(mesh.astype(np.float32) / 1000.0)

            file_path = f'./data/outputs/pose/{pipe_name}/gt_poses.json'
            gt_poses = []
            with open(file_path, 'rb') as f:
                pose = json.load(f)
                for data in pose['poses']:
                    pose = np.array([                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                        [data['rotation']['row0']['x'], data['rotation']['row0']['y'], data['rotation']['row0']['z'], data['translation']['x']],
                        [data['rotation']['row1']['x'], data['rotation']['row1']['y'], data['rotation']['row1']['z'], data['translation']['y']],
                        [data['rotation']['row2']['x'], data['rotation']['row2']['y'], data['rotation']['row2']['z'], data['translation']['z']]
                    ])
                    gt_poses.append(pose)
        gt_pose_list[i] = gt_poses

    for _img_num, _ in enumerate(tqdm(glob.glob(img_dir + "/rgb/*.png"))):
        img_num = str(_img_num * 10)
        img_list = []
        pred_rot_list = []
        pred_trans_list = []
        model_points_list = []
        K_list = []
        predictions_list = []
        for _ in range(len(pipe_list)):
            predictions_list.append([])
        is_empty = False
        for i, pipe_name in enumerate(pipe_list):
            file_path = os.path.join(cfg.output_dir, 'segmentation/', pipe_name, img_num, 'detection_ism.json')
            if os.path.exists(file_path) == False:
                trans = np.zeros((3, 1), dtype=np.float32)
                rot = np.zeros((3, 3), dtype=np.float32)
                predictions_list[i].append(np.hstack((rot, trans)))
                is_empty = True
                print(f"File not found: {file_path}")
                continue

            input_data, img, whole_pts, model_points, detections, radius= get_test_data(
                os.path.join(img_dir, "rgb", "frame"+img_num+'.png'), os.path.join(img_dir, "depth", "frame"+img_num+'.png'), 
                cfg.cam_path, os.path.join(cad_dir, pipe_name+'-'+cad_type+'.ply'), 
                os.path.join(cfg.output_dir, 'segmentation/', pipe_name, img_num, 'detection_ism.json'), 
                cfg.det_score_thresh, cfg.test_dataset, model_point_list[i]
            )
            ninstance = input_data['pts'].size(0)
            
            with torch.no_grad():
                input_data['dense_po'] = all_tem_pts[i].repeat(ninstance,1,1)
                input_data['dense_fo'] = all_tem_feat[i].repeat(ninstance,1,1)
                out = model(input_data)

            if 'pred_pose_score' in out.keys():
                pose_scores = out['pred_pose_score'] * out['score']
            else:
                pose_scores = out['score']
            pose_scores = pose_scores.detach().cpu().numpy()
            pred_rot = out['pred_R'].detach().cpu().numpy()
            pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

            os.makedirs(f"{cfg.output_dir}/pose/{pipe_name}/{img_num}", exist_ok=True)
            for idx, det in enumerate(detections):
                detections[idx]['score'] = float(pose_scores[idx])
                detections[idx]['R'] = list(pred_rot[idx].tolist())
                detections[idx]['t'] = list(pred_trans[idx].tolist())

            with open(os.path.join(f"{cfg.output_dir}/pose/{pipe_name}/{img_num}", 'detection_pem.json'), "w") as f:
                json.dump(detections, f)

            save_dir = os.path.join(f"{cfg.output_dir}/pose/{pipe_name}")
            valid_masks = pose_scores == pose_scores.max()
            K = input_data['K'].detach().cpu().numpy()
            
            gt_rot = []
            gt_trans = []
            gt_pose_data = []
            gt_poses = []

            for j, pose in enumerate(gt_pose_list[i]):
                pose[:, 3] = np.array([pose[:, 3][0], -pose[:, 3][1]-2, pose[:, 3][2]])

            for j, trans in enumerate(pred_trans):
                min_dist = float('inf')
                min_num = 0
                min_trans = np.zeros((3, 1), dtype=np.float32)
                for m, pose in enumerate(gt_pose_list[i]):
                    distance = np.linalg.norm(pose[:, 3] - trans)
                    if distance < min_dist:
                        min_dist = distance
                        min_trans = trans
                        min_num = m
                
                gt_poses.append(gt_pose_list[i][min_num])
                gt_rot.append(gt_pose_list[i][min_num][:, :3])
                gt_trans.append(gt_pose_list[i][min_num][:, 3]) 

            img_list.append(img)
            pred_rot_list.append(pred_rot)
            pred_trans_list.append(pred_trans)
            model_points_list.append(model_points*1000)
            K_list.append(K)

            add_correct_num = 0
            prj_correct_num = 0
            for j in range(len(pred_trans)):
                min_add = float('inf')
                min_prj = float('inf')
                for k in range(8):
                    temp_gt_poses = gt_poses[j].copy()
                    temp_gt_rots = gt_rot[j].copy()

                    temp_gt_rots = transform_points(temp_gt_rots, k)
                        
                    gt_trans_norm = np.linalg.norm(gt_trans[j])
                    pred_trans_norm = np.linalg.norm(pred_trans[j])

                    # スケール係数を計算
                    scale_factor = gt_trans_norm / pred_trans_norm
                    # # 推論された並進ベクトルにスケール変換を適用
                    corrected_pred_trans = pred_trans * scale_factor
                    # # 推論ポーズの並進ベクトルを修正
                    corrected_pred_pose = np.hstack((pred_rot[j], corrected_pred_trans[j].reshape(-1, 1)))

                    # ADDの計算
                    gt_points = transform_points(model_points * 1000, temp_gt_poses)
                    pred_points = transform_points(model_points * 1000, corrected_pred_pose)
                    distances = np.linalg.norm(pred_points - gt_points, axis=1)
                    ADD = np.mean(distances)
                    # if j == 0:
                    #     print(temp_gt_poses[:, :3])
                    #     print(temp_gt_poses[:, 3])
                    #     print(pred_rot[j])
                    #     print(pred_trans[j])

                    true_projection = project_points(model_points*1000, temp_gt_poses[:, :3], temp_gt_poses[:, 3], K)
                    pred_projection = project_points(model_points*1000, pred_rot[j], pred_trans[j], K)
                    reprojection_error = np.linalg.norm(true_projection - pred_projection, axis=1)
                    mean_reprojection_error = np.mean(reprojection_error)

                    if min_add > ADD:
                        min_add = ADD
                        gt_poses[j] = temp_gt_poses
                    if min_prj > mean_reprojection_error:
                        min_prj = mean_reprojection_error

                diameter = radius*2.0*1000
                if min_add < 0.1 * diameter:
                    add_correct_num += 1
                if min_prj < 5:
                    prj_correct_num += 1
                print(f"Num = {j}, ADD = {min_add}, diameter = {0.1 * diameter}, Prj-5 = {min_prj}")

            print(f"ADD正解数: {add_correct_num/len(gt_poses)*100}, Prj-5正解数: {prj_correct_num/len(gt_poses)*100}")
            visualize(img, pred_rot, pred_trans, model_points*1000, K, save_dir, img_num, gt_poses)
            gt_pose_list[i] = gt_poses

        if is_empty is not True:
            visualize_all(img_list[0], pred_rot_list, pred_trans_list, model_points_list, K_list, os.path.join(f"{cfg.output_dir}/pose"), pipe_list, img_num, gt_pose_list)
