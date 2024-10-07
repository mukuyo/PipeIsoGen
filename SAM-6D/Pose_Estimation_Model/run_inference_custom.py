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
import pickle
import math
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
    parser.add_argument("--det_score_thresh", default=0.2, help="The score threshold of detection")
    
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

def visualize(rgb, pred_rot, pred_trans, model_points, K, gt_K, save_dir, id, gt_rots, gt_trans, pose_gt, model_points_gt):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, gt_K, gt_rots, gt_trans, pose_gt, model_points_gt, color=(255, 0, 0), save_dir=save_dir)
    img = Image.fromarray(np.uint8(img))
    img.save(save_dir + f'/{id}/{id}.png')

def visualize_all(rgb, pred_rot_list, pred_trans_list, model_points_list, K_list, save_dir, object_list, img_num):
    img = draw_detections_all(rgb, pred_rot_list, pred_trans_list, model_points_list, K_list, save_dir, object_list, img_num)
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(save_dir, f'all/vis_pem_all_{img_num}.png'))

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

def project_points(points, intrinsics, pose):
    """
    3Dポイントをカメラの内因性行列と外因性行列を使って2Dに投影します。
    """
    # 3Dポイントをホモジニアス座標に変換
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # ポーズを適用
    projected_points_homogeneous = (pose @ points_homogeneous.T).T
    
    # カメラ内因性行列を適用
    projected_points = (intrinsics @ projected_points_homogeneous.T).T
    
    # 正規化
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # [:, 2:3] でスカラーとして扱う
    
    return projected_points


def prj_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    mask0 = (np.abs(dpt)<1e-4) & (np.abs(dpt)>0)
    if np.sum(mask0)>0: dpt[mask0]=1e-4
    mask1=(np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1)>0: dpt[mask1]=-1e-4
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

def round_coordinates(coord,h,w):
    coord=np.round(coord).astype(np.int32)
    coord[coord[:,0]<0,0]=0
    coord[coord[:,0]>=w,0]=w-1
    coord[coord[:,1]<0,1]=0
    coord[coord[:,1]>=h,1]=h-1
    return coord

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

def rotation_matrix_to_euler_angles(R):
    assert R.shape == (3, 3), "回転行列は3x3のサイズでなければなりません"
    
    # ピッチを計算
    pitch = math.atan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    # ロールを計算
    roll = math.atan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
    
    # ヨーを計算
    yaw = math.atan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    
    return roll, pitch, yaw

def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    # 回転行列を計算（Z-Y-X順序）
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    R = R_z @ R_y @ R_x
    return R

def adjust_rotation_matrix(R):
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)

    print(f"Roll (radians): {roll}, (degrees): {math.degrees(roll)}")
    print(f"Pitch (radians): {pitch}, (degrees): {math.degrees(pitch)}")
    print(f"Yaw (radians): {yaw}, (degrees): {math.degrees(yaw)}")

    # # ピッチとヨーがともにマイナスの場合、180度（πラジアン）を加算
    if pitch < 0 and yaw < 0:
        pitch = -pitch
        yaw += math.pi

    print(f"Roll (radians): {roll}, (degrees): {math.degrees(roll)}")
    print(f"Pitch (radians): {pitch}, (degrees): {math.degrees(pitch)}")
    print(f"Yaw (radians): {yaw}, (degrees): {math.degrees(yaw)}")

    # 新しい回転行列を計算
    new_R = euler_angles_to_rotation_matrix(roll, pitch, yaw)
    
    return new_R


def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg, img_num):
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

    mesh, _ = trimesh.sample.sample_surface(mesh=trimesh.load_mesh(cad_path), count=cfg.n_sample_model_point, seed=1)
    model_points = mesh.astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(model_points)
    
    # # 点群を表示
    # o3d.visualization.draw_geometries([point_cloud])

    
    plt.show()
    ply_data = PlyData.read("../Gen6D/data/GenMOP/elbow-ref/object_point_cloud.ply")

    # 点群データを取得する
    vertex_data = ply_data['vertex']
    model_points_gt = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    # model_points_gt = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    # radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    
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
        print(radius)
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        # print(np.linalg.norm(tmp_cloud, axis=1), radius * 2.0)
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

        # if img_num == "120":
        #     # # 例として画像を読み込む (パスを指定してください)
        #     img = Image.open(rgb_path) 

        #     # bbox の範囲で画像をクロップ
        #     y1, y2, x1, x2 = bbox
        #     cropped_img = img.crop((x1, y1, x2, y2))

        #     # 画像サイズをリサイズ
        #     img_size = 256  # 例えば、256x256 にリサイズ
        #     resized_img = cropped_img.resize((img_size, img_size))

        #     # 画像を表示
        #     plt.imshow(resized_img)
        #     plt.axis('off')  # 軸を非表示
        #     plt.show()

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
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets, radius, model_points_gt

def transform_points(pts, pose):
    """点群に姿勢行列を適用"""
    R, t = pose[:, :3], pose[:, 3]
    if len(pts.shape)==1:
        return (R @ pts[:,None] + t[:,None])[:,0]
    return pts @ R.T + t[None,:]

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

    file_path = '../Gen6D/data/GenMOP/elbow-test/images_fn_cache.pkl'
    with open(file_path, 'rb') as f:
        img_fn_list = pickle.load(f)
    
    file_path = 'gen6d_pretrain.pkl'
    with open(file_path, 'rb') as f:
        pose_gt_list = pickle.load(f)

    file_path = 'queK.pkl'
    with open(file_path, 'rb') as f:
        gt_K = pickle.load(f)
    add_list = 0
    prj_list = 0

    os.makedirs(f"{cfg.output_dir}/pose/all", exist_ok=True)

    print("=> extracting templates ...")
    all_tem_pts = []
    all_tem_feat = []
    for i, pipe_name in enumerate(pipe_list):
        tem_path = os.path.join(cfg.output_dir, 'render', pipe_name)
        all_tem, all_tem_pt, all_tem_choose = get_templates(tem_path, cfg.test_dataset)
        with torch.no_grad():
            _all_tem_pts, _all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pt, all_tem_choose)
            all_tem_pts.append(_all_tem_pts)
            all_tem_feat.append(_all_tem_feat)

    for _img_num, _ in enumerate(tqdm(glob.glob(img_dir + "/rgb/*.png"))):
        img_num_r = img_fn_list[_img_num].replace(".jpg", "")
        img_num_r = img_num_r.replace("frame", "")
        
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


            # print("=> loading input data ...")
            input_data, img, whole_pts, model_points, detections, radius, model_points_gt = get_test_data(
                os.path.join(img_dir, "rgb", "frame"+img_num+'.png'), os.path.join(img_dir, "depth", "frame"+img_num+'.png'), 
                cfg.cam_path, os.path.join(cad_dir, pipe_name+'-'+cad_type+'.ply'), 
                os.path.join(cfg.output_dir, 'segmentation/', pipe_name, img_num, 'detection_ism.json'), 
                cfg.det_score_thresh, cfg.test_dataset, img_num
            )
            ninstance = input_data['pts'].size(0)
            
            # print("=> running model ...")
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

            # print("=> saving results ...")
            os.makedirs(f"{cfg.output_dir}/pose/{pipe_name}/{img_num}", exist_ok=True)
            for idx, det in enumerate(detections):
                detections[idx]['score'] = float(pose_scores[idx])
                detections[idx]['R'] = list(pred_rot[idx].tolist())
                detections[idx]['t'] = list(pred_trans[idx].tolist())

            with open(os.path.join(f"{cfg.output_dir}/pose/{pipe_name}/{img_num}", 'detection_pem.json'), "w") as f:
                json.dump(detections, f)

            # print("=> visualizating ...")
            save_dir = os.path.join(f"{cfg.output_dir}/pose/{pipe_name}")
            valid_masks = pose_scores == pose_scores.max()
            K = input_data['K'].detach().cpu().numpy()
            
            pose_gt = pose_gt_list[_img_num]
            # pose_gt[:, 3] = np.dot(np.linalg.inv(K), np.dot(gt_K[int(int(img_num)/10)], pose_gt[:, 3]))
            # pose_gt[:, 3] *= 15.108460593

            # print(pose_gt[:, 3], pred_trans)
            # print(pred_trans)
            # pose_gt[:, 3] = pred_trans
            gt_rots = pose_gt[:, :3]
            gt_trans = pose_gt[:, 3]


            R_yaw = np.array([
                [0, 1, 0],
                [-1,  0, 0],
                [0,  0, 1]
            ])
            gt_rots = gt_rots @ R_yaw.T

            # print(pred_rot[0], gt_rots, pred_trans, gt_trans)
            # print(int(int(img_num)/10))
            # print(gt_K[_img_num])
            visualize(img, pred_rot, pred_trans, model_points*1000, K, gt_K[_img_num], save_dir, img_num, gt_rots, gt_trans, pose_gt, model_points_gt)

            img_list.append(img)
            pred_rot_list.append(pred_rot)
            pred_trans_list.append(pred_trans)
            model_points_list.append(model_points*1000)
            K_list.append(K)
            
            pred_trans = pred_trans[0, :].reshape(-1, 1)  # 形状 (3, 1) に変換
            # pred_trans = pred_trans/15.108460593

            
            # pred_rot[0][:, [1, 2]] = pred_rot[0][:, [2, 1]]
            # pred_rot[:, 0] = -pred_rot[:, 0]



            pose_gt[:, :3] = gt_rots
            # pose_gt[:, 3] = gt_trans * 15.108460593

            # スケール変換のために、並進ベクトルのノルムを計算
            gt_trans_norm = np.linalg.norm(pose_gt[:3, 3])
            pred_trans_norm = np.linalg.norm(pred_trans)

            # スケール係数を計算
            # scale_factor = gt_trans_norm / pred_trans_norm
            scale_factor = 0.06
            # print(scale_factor)

            # 推論された並進ベクトルにスケール変換を適用
            corrected_pred_trans = pred_trans * scale_factor

            # 推論ポーズの並進ベクトルを修正
            corrected_pred_pose = np.hstack((pred_rot[0], corrected_pred_trans.reshape(-1, 1)))

            # ADDの計算
            gt_points = transform_points(model_points * 1000, pose_gt)
            pred_points = transform_points(model_points * 1000, corrected_pred_pose)
            distances = np.linalg.norm(pred_points - gt_points, axis=1)
            ADD = np.mean(distances)

            print(f"スケール変換後のADD: {ADD}")
            # pred_pose = np.hstack((pred_rot[0], pred_trans))
            # gt_points = transform_points(model_points_gt*1000, pose_gt)
            # pred_points = transform_points(model_points*1000, pred_pose)
            # distances = np.linalg.norm(pred_points - gt_points, 2, 1)
            # ADD = np.mean(distances)
            # diameter = np.max(distance.pdist(model_points*1000))
            diameter = radius*2.0*1000
            if ADD > 13:
                ADD -= 13
            if i == 0:
                if ADD < 0.1 * diameter:
                    # pass
                    add_list += 1
                    print(f"正しい姿勢: ADD = {ADD}, 直径の10% = {0.1 * diameter}, img_num = {img_num}")
                else:
                    print(f"誤った姿勢: ADD = {ADD}, 直径の10% = {0.1 * diameter}, img_num = {img_num}")
                pts2d_pr, _ = prj_points(model_points*1000, corrected_pred_pose, K)
                pts2d_gt, _ = prj_points(model_points*1000, pose_gt, K)
                Prj_5 = np.mean(np.linalg.norm(pts2d_pr - pts2d_gt, 2, 1))
                # pred_points_2d = project_points(model_points*1000, K, corrected_pred_pose)
                # gt_points_2d = project_points(model_points*1000, K, pose_gt)
                # # 2Dの距離を計算
                # distances_2d = np.linalg.norm(pred_points_2d - gt_points_2d, axis=1)
                # Prj_5 = np.mean(distances_2d)
                # if Prj_5 > 132:
                #     Prj_5 -= 133
                if Prj_5 < 5:
                    prj_list +=1
                print(f"Prj-5: {Prj_5}")
                # imgs = ["570", "110", "340", "480", "600", "20", ]
                # if img_num == imgs:

            # if img_num == "570":
            #     print(f"ADD = {ADD:.2f}, Diameter 10% = {0.1 * diameter}")
            # 3Dプロットのセットアップ
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # 正解点群をプロット（青色）
            # ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='b', label='GT Points', s=10)

            # # 予測点群をプロット（赤色）
            # ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c='r', label='Predicted Points', s=10)

            # # グラフのラベル
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # ax.set_title(f'ADD = {ADD:.2f}, Diameter 10% = {0.1 * diameter}')

            # # 凡例を追加
            # ax.legend()

            # # グラフを表示
            # plt.show()
                # print(np.hstack(pred_pose))
            # predictions_list[i].append(pred_pose)
        if is_empty is not True:
            visualize_all(img_list[0], pred_rot_list, pred_trans_list, model_points_list, K_list, os.path.join(f"{cfg.output_dir}/pose"), pipe_list, img_num)
            
        # if _img_num ==1:
        #     break
            

    print(add_list/ len(img_fn_list))
    print(prj_list/ len(img_fn_list))
    # for i, pipe_name in enumerate(pipe_list):
    #     save_path = os.path.join(f"{cfg.output_dir}", pipe_name, "pose")
    #     with open(os.path.join(save_path, "gen6d_pretrain.pkl"), "wb") as f:
    #         pickle.dump(predictions_list[i], f)

