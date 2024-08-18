import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# ディレクトリ名とパスの設定
directory_name = '08_06/'
parent_folder = 'data/capture/'
subfolder_path = os.path.join(parent_folder, directory_name)
os.makedirs(subfolder_path, exist_ok=True)
os.makedirs(parent_folder + directory_name + 'rgb/', exist_ok=True)
os.makedirs(parent_folder + directory_name + 'depth/', exist_ok=True)

# カメラ行列の取得関数
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

# RealSenseパイプラインの初期化
pipeline = rs.pipeline()
config = rs.config()

# ストリームの有効化
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# パイプラインの開始
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# フィルターの初期化
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

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
color_params_filename = f'data/capture/' + directory_name + 'cam_K.txt'
np.savetxt(color_params_filename, color_camera_matrix, fmt='%f')

# センサーから深度スケールを取得
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print(f"深度スケール（メートル単位）: {depth_scale}")

# 深度スケールをミリメートルに変換
depth_scale_mm = depth_scale * 1000  # メートルからミリメートルに変換

print(f"深度スケール（ミリメートル単位）: {depth_scale_mm}")

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
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 深度をミリメートルに変換
        depth_image_mm = depth_image * depth_scale_mm
        
        cv2.imshow('Aligned Images', depth_image_mm.astype(np.uint16))
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('s'):
            color_filename = f'data/capture/' + directory_name + 'rgb/' + str(count) + '.' + file_format
            depth_filename = f'data/capture/' + directory_name + 'depth/' + str(count) + '.' + file_format
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_image_mm.astype(np.uint16))
            print(f'保存しました: {color_filename} と {depth_filename}')
            count += 1
        
        elif key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
