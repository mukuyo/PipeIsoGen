import os
import numpy as np
import cv2
import json

class Connect:
    """Calculate Pipe Connection"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger
    
    def compute_piping_relationship(self) -> None:
        """compute piping relationship"""
        self.__logger.info("Start compute piping relationship")
        image = cv2.imread(self.__args.rgb_path)
        
        # カメラパラメータをJSONファイルから読み込む
        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)
        
        camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
        depth_scale = cam_params["depth_scale"]
        
        # 矢印の長さをスケーリングするための係数
        arrow_length = 10  # この値を調整して矢印の長さを変更

        # すべての配管の姿勢をロードして辞書に格納
        for obj_name in self.__args.objects_name:
            pose_path = os.path.join(self.__args.pose_dir, obj_name, "pose.npy")
            pose_list = np.load(pose_path, allow_pickle=True)  # リストをロード
            
            direction_list = [1, 2]
            if obj_name == 'tee':
                direction_list = [1, 2, -2]
            
            for i, pose_matrix in enumerate(pose_list):
                for direction in direction_list:
                    if direction == -2:
                        # Z軸方向ベクトルの反対方向を計算
                        z_axis_vector = -pose_matrix[:3, 2]  # -2 means the opposite of the Z-axis vector
                    else:
                        # 通常のZ軸方向ベクトルを計算
                        z_axis_vector = pose_matrix[:3, direction]  # 回転行列の対応する列

                    translation = pose_matrix[:3, 3]  # 並進ベクトル

                    # Z軸の反対方向に矢印を描画
                    z_axis_end_point_3d = translation - z_axis_vector * arrow_length
                    
                    # 3D座標を2D画像座標に変換するために、カメラ座標系に拡張
                    start_point_3d = np.append(translation, 1)  # 補正後の中心点
                    end_point_3d = np.append(z_axis_end_point_3d, 1)  # 矢印の先端
                    
                    # カメラ行列で変換
                    start_point_2d_homogeneous = camera_matrix @ start_point_3d[:3]
                    end_point_2d_homogeneous = camera_matrix @ end_point_3d[:3]

                    # 正規化して2D座標に変換
                    start_point_2d = (start_point_2d_homogeneous / start_point_2d_homogeneous[2])[:2]
                    end_point_2d = (end_point_2d_homogeneous / end_point_2d_homogeneous[2])[:2]
                    
                    # 画像座標系に変換
                    start_point = (int(start_point_2d[0]), int(start_point_2d[1]))
                    end_point = (int(end_point_2d[0]), int(end_point_2d[1]))

                    # デバッグ用にオブジェクトの中心を描画 (赤い点)
                    cv2.circle(image, start_point, 2, (0, 0, 255), -1)  # 赤色の点

                    # 画像上にZ軸方向ベクトルの反対側を描画
                    cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 3)  # 緑色の矢印
        
        # 画像を保存する
        output_path = 'output_image.png'
        cv2.imwrite(output_path, image)
        self.__logger.info(f"Output image saved to {output_path}")
