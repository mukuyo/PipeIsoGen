import os
import cv2
import numpy as np
from PIL import Image

class DrawUtils:
    """Draw Utils class"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger

    # def __draw_pose_info(self, img, imgpts_list) -> None:
    #     """Draw pose information on the image."""
    #     size = 2
    #     for imgpts in imgpts_list:
    #         # 面1（前面）：赤
    #         color = (0, 0, 255)  # Red color
    #         for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
    #             img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

    #         # 面2（後面）：青
    #         color = (255, 0, 0)  # Blue color
    #         for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
    #             img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

    #         # 面3（右面）：緑
    #         color = (0, 255, 0)  # Green color
    #         for i, j in zip([0, 1, 5, 4], [1, 5, 4, 0]):
    #             img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

    #         # 面4（左面）：黄色
    #         color = (0, 255, 255)  # Yellow color
    #         for i, j in zip([2, 3, 7, 6], [3, 7, 6, 2]):
    #             img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

    #         # 面5（上面）：紫
    #         color = (255, 0, 255)  # Purple color
    #         for i, j in zip([1, 3, 7, 5], [3, 7, 5, 1]):
    #             img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

    #         # 面6（下面）：オレンジ
    #         color = (0, 165, 255)  # Orange color
    #         for i, j in zip([0, 2, 6, 4], [2, 6, 4, 0]):
    #             img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

    #     return img

    def __line_intersection(self, p1, p2, p3, p4):
        """計算2つの線分の交点を返す"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        def line_eq(p1, p2):
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = A * p1[0] + B * p1[1]
            return A, B, -C

        A1, B1, C1 = line_eq(p1, p2)
        A2, B2, C2 = line_eq(p3, p4)

        det = A1 * B2 - A2 * B1
        if det == 0:
            return None  # 平行線または同一線分

        x = (B2 * -C1 - B1 * -C2) / det
        y = (A1 * -C2 - A2 * -C1) / det
        return (x, y)

    def __draw_pose_info(self, img, imgpts_list) -> None:
        """Draw pose information on the image."""
        for imgpts in imgpts_list:
            size = 2
            color = (0, 0, 255)
            # upper
            for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
                img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)

            # under
            for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
                img = cv2.line(img, tuple(imgpts[i].astype(int)), tuple(imgpts[j].astype(int)), color, size)
                
            color = (0, 255, 0)
            size = 1
            # 対角線
            intersect_points = []
            for i in range(2):
                p1 = imgpts[0+i*4].astype(int)
                p2 = imgpts[3+i*4].astype(int)
                p3 = imgpts[1+i*4].astype(int)
                p4 = imgpts[2+i*4].astype(int)
                
                img = cv2.line(img, p1, p2, color, size)
                img = cv2.line(img, p3, p4, color, size)
                
                # 交点を計算
                intersect_point = self.__line_intersection(p1, p2, p3, p4)
                if intersect_point:
                    intersect_points.append(intersect_point)
                    # 交点が画像の範囲内に収まっているかチェック
                    if (0 <= intersect_point[0] < img.shape[1]) and (0 <= intersect_point[1] < img.shape[0]):
                        # 交点を描画
                        intersect_point = tuple(map(int, intersect_point))
                        img = cv2.circle(img, intersect_point, 3, (255, 0, 0), -1)  # 赤色の点
            
            # 交点を結ぶ線を描画
            if len(intersect_points) == 2:
                p1 = tuple(map(int, intersect_points[0]))
                p2 = tuple(map(int, intersect_points[1]))
                img = cv2.line(img, p1, p2, (255, 0, 0), 2)
        
        return img



    def init_pose_show(self) -> None:
        """Load an image from a path, show initial pose information, and save the image."""
        # Load the image from the given path
        img = cv2.imread(self.__args.rgb_path)

        if img is None:
            self.__logger.error(f"Failed to load image from {self.__args.rgb_path}")
            return

        for obj_name in self.__args.objects_name:
            # Load the 3D bounding box points
            pts_path = os.path.join(self.__args.pose_dir, obj_name, "3d_bbox.npy")
            imgpts = np.load(pts_path)

            # Draw the pose information
            img = self.__draw_pose_info(img, imgpts)

        # Specify the save path
        save_path = os.path.join(self.__args.output_dir, "isometric/pose_info.png")
        
        # Save the result
        cv2.imwrite(save_path, img)
        self.__logger.info(f"Pose image saved to {save_path}")
