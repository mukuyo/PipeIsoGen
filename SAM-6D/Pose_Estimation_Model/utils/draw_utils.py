import numpy as np
import os
import cv2

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def get_bbox_center_coordinate(bbox_coordinates):
    min_coords = np.min(bbox_coordinates, axis=1)
    max_coords = np.max(bbox_coordinates, axis=1)

    center_corrected = (min_coords + max_coords) / 2

    return center_corrected

def draw_3d_bbox(img, imgpts, color, size=2):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img, imgpts

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def project_points(pts,RT,K):
    # print("RT shape:", RT.shape)
    # print("pts shape:", pts.shape)
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    mask0 = (np.abs(dpt)<1e-4) & (np.abs(dpt)>0)
    if np.sum(mask0)>0: dpt[mask0]=1e-4
    mask1=(np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1)>0: dpt[mask1]=-1e-4
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

def pts_range_to_bbox_pts(max_pt, min_pt):
    if not (hasattr(max_pt, '__iter__') and hasattr(min_pt, '__iter__')):
        raise ValueError("max_pt and min_pt should be iterable containing x, y, z coordinates.")
    
    maxx, maxy, maxz = max_pt
    minx, miny, minz = min_pt
    
    pts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz],
        
        [minx, miny, maxz],
        [minx, maxy, maxz],
        [maxx, maxy, maxz],
        [maxx, miny, maxz],
    ]
    return np.asarray(pts, np.float32)

def draw_bbox_3d(img,pts2d,color=(0,255,0)):
    red_colors=np.zeros([8,3],np.uint8)
    red_colors[:,0]=255
    # img = draw_keypoints(img, pts2d, colors=red_colors)

    pts2d = np.round(pts2d).astype(np.int32)
    img = cv2.line(img,tuple(pts2d[0]),tuple(pts2d[1]),color,2)
    img = cv2.line(img,tuple(pts2d[1]),tuple(pts2d[2]),color,2)
    img = cv2.line(img,tuple(pts2d[2]),tuple(pts2d[3]),color,2)
    img = cv2.line(img,tuple(pts2d[3]),tuple(pts2d[0]),color,2)

    img = cv2.line(img,tuple(pts2d[4]),tuple(pts2d[5]),color,2)
    img = cv2.line(img,tuple(pts2d[5]),tuple(pts2d[6]),color,2)
    img = cv2.line(img,tuple(pts2d[6]),tuple(pts2d[7]),color,2)
    img = cv2.line(img,tuple(pts2d[7]),tuple(pts2d[4]),color,2)

    img = cv2.line(img,tuple(pts2d[0]),tuple(pts2d[4]),color,2)
    img = cv2.line(img,tuple(pts2d[1]),tuple(pts2d[5]),color,2)
    img = cv2.line(img,tuple(pts2d[2]),tuple(pts2d[6]),color,2)
    img = cv2.line(img,tuple(pts2d[3]),tuple(pts2d[7]),color,2)
    return img

# def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, gt_intrinsics, gt_rots, gt_trans, pose_gt, model_points_gt, color=(255, 0, 0), save_dir=None):
#     num_pred_instances = len(pred_rots)
#     draw_image_bbox = image.copy()
#     # 3d bbox
#     scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
#     shift = np.mean(model_points, axis=0)
#     bbox_3d = get_3d_bbox(scale, shift)

#     scale_gt = (np.max(model_points_gt, axis=0) - np.min(model_points_gt, axis=0))
#     shift_gt = np.mean(model_points_gt, axis=0)
#     bbox_3d_gt = get_3d_bbox(scale_gt, shift_gt)

#     # 3d point
#     choose = np.random.choice(np.arange(len(model_points)), 512)
#     pts_3d = model_points[choose].T
#     combined_list = []
#     for ind in range(num_pred_instances):
#         color=(0, 0, 255)
#         # draw 3d bounding box
#         transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
#         projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
#         draw_image_bbox, _ = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
#         # draw point cloud
#         # transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
#         # projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
#         # draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

#         color=(0, 255, 0)
#         # draw 3d bounding box
#         # transformed_bbox_3d = gt_rots@bbox_3d + gt_trans[:,np.newaxis]
#         # projected_bbox = calculate_2d_projections(transformed_bbox_3d, gt_intrinsics)
#         # gt_trans *= 15
#         print(pred_trans, gt_trans)
#         # object_bbox_3d = pts_range_to_bbox_pts(np.max(model_points_gt), np.min(model_points_gt))
#         max_pt = np.max(model_points_gt, axis=0)
#         min_pt = np.min(model_points_gt, axis=0)
        
#         # max_pt と min_pt が (x, y, z) の形式であることを確認する
#         transformed_bbox_3d = gt_rots@bbox_3d_gt +gt_trans[:,np.newaxis]
#         projected_bbox = calculate_2d_projections(transformed_bbox_3d, gt_intrinsics)
#         draw_image_bbox, _ = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
#         # object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
#         # projected_bbox,_ = project_points(object_bbox_3d, pose_gt, intrinsics[ind])
#         # draw_image_bbox, _ = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
#         # draw point cloud
#         # transformed_pts_3d = gt_rots@pts_3d + gt_trans[:,np.newaxis]
#         # projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
#         # draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

#         # center_coordinate = get_bbox_center_coordinate(transformed_bbox_3d)
#         # trans_reshaped = center_coordinate.reshape(-1, 1)
#         # combined_arr = np.hstack((pred_rots[ind], trans_reshaped))
#         # combined_list.append(combined_arr)
#     np.save(os.path.join(save_dir, "pose.npy"), np.array(combined_list))

#     return draw_image_bbox


def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, gt_intrinsics, gt_rots, gt_trans, pose_gt, model_points_gt, color=(255, 0, 0), save_dir=None):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    # 3d bbox
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    scale_gt = (np.max(model_points_gt, axis=0) - np.min(model_points_gt, axis=0))
    shift_gt = np.mean(model_points_gt, axis=0)
    bbox_3d_gt = get_3d_bbox(scale_gt, shift_gt)

    # 3d point
    choose = np.random.choice(np.arange(len(model_points)), 512)
    pts_3d = model_points[choose].T
    combined_list = []
    for ind in range(num_pred_instances):
        color=(0, 0, 255)
        # draw 3d bounding box
        transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
        draw_image_bbox, _ = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
        # draw point cloud
        # transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
        # projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
        # draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

        color=(0, 255, 0)
        transformed_bbox_3d = pts_range_to_bbox_pts(np.max(model_points_gt,0), np.min(model_points_gt,0))
        # transformed_bbox_3d = np.array([[    -0.7409,   -0.087899,     -2.6196],
        #                                 [    -0.7409,     0.85776  ,   -2.6196],
        #                                 [    0.52891,     0.85776 ,    -2.6196],
        #                                 [    0.52891,   -0.087899,     -2.6196],
        #                                 [    -0.7409,   -0.087899 ,    -1.3976],
        #                                 [    -0.7409,     0.85776,     -1.3976],
        #                                 [    0.52891,     0.85776 ,    -1.3976],
        #                                 [    0.52891,   -0.087899,     -1.3976]])
        # transformed_bbox_3d = np.array([[   -0.55432,    -0.80312,     -2.7515],
        #                                 [   -0.55432,     0.16901,     -2.7515],
        #                                 [    0.58109,     0.16901,     -2.7515],
        #                                 [    0.58109,    -0.80312,     -2.7515],
        #                                 [   -0.55432,    -0.80312,     -1.4227],
        #                                 [   -0.55432,     0.16901,     -1.4227],
        #                                 [    0.58109,     0.16901,     -1.4227],
        #                                 [    0.58109,    -0.80312,     -1.4227]])
        projected_bbox, _ = project_points(transformed_bbox_3d, pose_gt, gt_intrinsics)
        draw_image_bbox = draw_bbox_3d(draw_image_bbox, projected_bbox, color)

    np.save(os.path.join(save_dir, "pose.npy"), np.array(combined_list))
    return draw_image_bbox

def draw_detections_all(image, pred_rot_list, pred_tran_list, model_point_list, intrinsic_list, save_dir, object_list, img_num):
    draw_image_bbox = image.copy()

    for i, obj_name in enumerate(object_list):
        pred_rots = pred_rot_list[i]
        pred_trans = pred_tran_list[i]
        model_points = model_point_list[i]
        intrinsics = intrinsic_list[i]
        color = (255, 100, 0)
        if i == 1:
            color = (0, 100, 255)

        num_pred_instances = len(pred_rots)
        
        # 3d bbox
        scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
        shift = np.mean(model_points, axis=0)
        bbox_3d = get_3d_bbox(scale, shift)

        # 3d point
        choose = np.random.choice(np.arange(len(model_points)), 512)
        pts_3d = model_points[choose].T
        
        imgpts_list = []
        combined_list = []
        for ind in range(num_pred_instances):
            # draw 3d bounding box
            transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
            draw_image_bbox, imgpts = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
            # draw point cloud
            transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
            projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
            draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

            center_coordinate = get_bbox_center_coordinate(transformed_bbox_3d)
            trans_reshaped = center_coordinate.reshape(-1, 1)
            combined_arr = np.hstack((pred_rots[ind], trans_reshaped))
            combined_list.append(combined_arr)
            
            imgpts_list.append(imgpts)

        np.save(os.path.join(save_dir, obj_name, img_num, "pose.npy"), np.array(combined_list))
        np.save(os.path.join(save_dir, obj_name, img_num, "3d_bbox_projected.npy"), imgpts_list)
        

    return draw_image_bbox
