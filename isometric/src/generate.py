import os
import sys
import json
import argparse
import numpy as np
from logging import getLogger, DEBUG, StreamHandler, Formatter

# Add the correct path for the isometric module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from isometric.src.connect import Connect
from isometric.common.pipe import Pipe
from isometric.utils.draw import DrawUtils

class Iso:
    """Isometric class"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger

        self.__init_pipe()

        self.__draw = DrawUtils(args, logger)
        self.__connect = Connect(args, logger)
        
    def __init_pipe(self) -> None:
        self.__logger.info("Init Pipe Information")
        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
        self.__pipes: list[Pipe] = []
        pipe_count = 0
        for obj_name in self.__args.objects_name:
            pose_list = np.load(os.path.join(self.__args.pose_dir, obj_name, "pose.npy"))
            for pose_matrix in pose_list:
                self.__pipes.append(Pipe(args, logger, obj_name, pipe_count, pose_matrix, camera_matrix))
                pipe_count += 1
    
    def generate_iso(self) -> None:
        """Generate isometric"""
        self.__draw.pipe_direction(self.__pipes)
        self.__connect.compute_piping_relationship(self.__pipes)
        first_pipe = self.__connect.find_first_pipe(self.__pipes)
        trans_pipes = self.__connect.traverse_pipes(self.__pipes, first_pipe)
        for trance in trans_pipes:
            start_pipe_num = trance[0]
            end_pipe_num = trance[1]
            distance = self.__connect.get_distance(self.__pipes[start_pipe_num], self.__pipes[end_pipe_num])
            self.__draw.pipe_line(self.__pipes[start_pipe_num], self.__pipes[end_pipe_num], distance)
        for pipe in self.__pipes:
            if len(pipe.relationship) == pipe.candidate_num:
                continue
            self.__draw.remain_pipe_line(pipe)
        self.__draw.save_dxf()
        
if __name__ == "__main__":
    fmt = Formatter("[%(levelname)s] %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    handler = StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    handler.setLevel(DEBUG)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--pose_dir", nargs="?", help="Path to pose estimation information")
    parser.add_argument("--objects_name", default=['tee', 'elbow'], help="Target object name")
    args = parser.parse_args()

    logger.info('start predict')

    iso = Iso(args, logger)
    iso.generate_iso()
    
    logger.info('end predict')
