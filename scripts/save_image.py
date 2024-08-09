import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

directry_name = '08_06/'
parent_folder = 'data/capture/'
subfolder_path = os.path.join(parent_folder, directry_name)
os.makedirs(subfolder_path, exist_ok=True)
os.makedirs(parent_folder + directry_name + 'rgb/', exist_ok=True)
os.makedirs(parent_folder + directry_name + 'depth/', exist_ok=True)

def get_camera_matrix(intrinsics):
    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy

    camera_matrix = np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ])
    
    return camera_matrix

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# フィルターの初期化
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# フィルターパラメータの設定
decimation.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 2)
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)
# temporal.set_option(rs.option.persistence_control, 3)

# カメラパラメータを取得して保存
depth_stream = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

color_stream = profile.get_stream(rs.stream.color)
color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

depth_camera_matrix = get_camera_matrix(depth_intrinsics)
color_camera_matrix = get_camera_matrix(color_intrinsics)

timestamp = int(time.time())
depth_params_filename = f'data/capture/' + directry_name + 'depth_camera_matrix.txt'
color_params_filename = f'data/capture/' + directry_name + 'color_camera_matrix.txt'

np.savetxt(depth_params_filename, depth_camera_matrix, fmt='%f')
np.savetxt(color_params_filename, color_camera_matrix, fmt='%f')

print(f'深度カメラ行列を保存しました: {depth_params_filename}')
print(f'カラーカメラ行列を保存しました: {color_params_filename}')

# ノイズ除去の閾値
NOISE_THRESHOLD = 10000  # ここで閾値を設定
SAVE_INTERVAL = 1  # 画像を自動保存する間隔（秒）

count = 0
last_saved_time = time.time()
file_format = 'png'  # ここで保存形式を設定（'png', 'jpg', 'tiff'など）

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # フィルターの適用
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        color_image = np.asanyarray(color_frame.get_data())

        # ノイズを黒にする処理
        depth_image[depth_image > NOISE_THRESHOLD] = 0

        # 深度画像の反転処理（近いほど白く、遠いほど黒く）
        max_depth = np.max(depth_image)
        min_depth = np.min(depth_image)
        inverted_depth_image = max_depth - depth_image
        inverted_depth_image[depth_image == 0] = 0

        # 深度画像を3次元に変換
        inverted_depth_image_3d = cv2.cvtColor(inverted_depth_image, cv2.COLOR_GRAY2BGR)
        
        # images = np.hstack((color_image, inverted_depth_image_3d))
        
        cv2.imshow('Aligned Images', inverted_depth_image)
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('s'):
            color_filename = f'data/capture/' + directry_name + 'rgb/' + str(count) + '.' + file_format
            depth_filename = f'data/capture/' + directry_name + 'depth/' + str(count) + '.' + file_format
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, inverted_depth_image)
            print(f'保存しました: {color_filename} と {depth_filename}')
            count += 1
        
        # # 自動保存機能
        # if time.time() - last_saved_time > SAVE_INTERVAL:
        #     color_filename = f'data/capture/' + directry_name + 'rgb/' + str(count) + '.' + file_format
        #     depth_filename = f'data/capture/' + directry_name + 'depth/' + str(count) + '.' + file_format
        #     cv2.imwrite(color_filename, color_image)
        #     cv2.imwrite(depth_filename, inverted_depth_image)
        #     print(f'自動保存しました: {color_filename} と {depth_filename}')
        #     last_saved_time = time.time()
        #     count += 1
        
        elif key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()