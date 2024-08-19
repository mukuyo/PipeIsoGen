"""Getting distance for between two pipes"""
import numpy as np
from cv2 import imread, IMREAD_UNCHANGED  # pylint: disable = no-name-in-module
from pyrealsense2 import rs2_deproject_pixel_to_point, intrinsics, \
                         distortion  # pylint: disable = no-name-in-module


class Distance:
    """get distance between two pipes"""
    def __init__(self, _cfg) -> None:
        self.cfg = _cfg

    def _compute_3d_distance(self, position1, position2,
                             depth_image, _intrinsics):
        depth1 = depth_image[position1[1], position1[0]]
        depth2 = depth_image[position2[1], position2[0]]

        point1 = rs2_deproject_pixel_to_point(_intrinsics,
                                              [position1[0], position1[1]],
                                              depth1 * 0.22)
        point2 = rs2_deproject_pixel_to_point(_intrinsics,
                                              [position2[0], position2[1]],
                                              depth2 * 0.22)

        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        return distance

    def get_info(self, trans_info):
        """get distance information"""
        depth_image = imread(self.cfg['isometric']['depth_path'] +
                             self.cfg['input_name'], IMREAD_UNCHANGED)
        _intrinsics = intrinsics()
        _intrinsics.width = depth_image.shape[1]
        _intrinsics.height = depth_image.shape[0]
        _intrinsics.ppx = 361
        _intrinsics.ppy = 242
        _intrinsics.fx = 470
        _intrinsics.fy = 470
        _intrinsics.model = distortion.none
        _intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        for info in trans_info:
            distance = self._compute_3d_distance(info.position1,
                                                 info.position2,
                                                 depth_image,
                                                 _intrinsics)
            info.distance_val = int(distance)

        return trans_info
