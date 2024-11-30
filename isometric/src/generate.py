import os
import sys
import json
import argparse
import numpy as np
import glob
from logging import getLogger, DEBUG, StreamHandler, Formatter

# Add the correct path for the isometric module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from isometric.src.connect import Connect
from isometric.common.pipe import Pipe
from isometric.utils.draw import DrawUtils
from tqdm import tqdm

class Iso:
    """Isometric class"""
    def __init__(self, args, logger) -> None:
        self.__args = args
        self.__logger = logger
        self.__rgb_dir = os.path.join(self.__args.img_dir, "rgb")
        self.__depth_dir = os.path.join(self.__args.img_dir, "depth")

        os.makedirs(os.path.join(self.__args.output_dir, "isometric"), exist_ok=True)

        self.__draw = DrawUtils(args, logger)
        self.__connect = Connect(args, logger)

        with open(self.__args.cam_path, 'r') as f:
            cam_params = json.load(f)        
        self.__camera_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
    
    def generate_iso(self) -> None:
        """Generate isometric"""
        for _img_num, _ in enumerate(glob.glob(self.__rgb_dir + "/*.png")):
            img_num = _img_num * 10
            self.__pipes: list[Pipe] = []
            pipe_count = 0
            for obj_name in self.__args.objects_name:
                save_dir = os.path.join(self.__args.output_dir, "isometric", str(img_num))
                os.makedirs(save_dir, exist_ok=True)
                pose_list = np.load(os.path.join(os.path.join(self.__args.output_dir, "pose", obj_name, str(img_num), "pose.npy")))
                for pose_matrix in pose_list:
                    self.__pipes.append(Pipe(args, logger, obj_name, pipe_count, pose_matrix, self.__camera_matrix))
                    pipe_count += 1

            self.__draw.pipe_direction(self.__pipes, save_dir, img_num)
            self.__connect.compute_piping_relationship(self.__pipes)
            first_pipe = self.__connect.find_first_pipe(self.__pipes)
            trans_pipes = self.__connect.traverse_pipes(self.__pipes, first_pipe)
            for trance in trans_pipes:
                start_pipe_num = trance[0]
                end_pipe_num = trance[1]
                distance = self.__connect.get_distance(self.__pipes[start_pipe_num], self.__pipes[end_pipe_num], os.path.join(self.__depth_dir, f"frame{str(img_num)}.png"))
                self.__draw.pipe_line(self.__pipes[start_pipe_num], self.__pipes[end_pipe_num], distance)
            for pipe in self.__pipes:
                if len(pipe.relationship) == pipe.candidate_num:
                    continue
                self.__draw.remain_pipe_line(pipe)
            self.__draw.save_dxf(img_num)
        
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
    parser.add_argument("--img_dir", nargs="?", help="Path to image")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--pose_dir", nargs="?", help="Path to pose estimation information")
    parser.add_argument("--objects_name", default=['elbow', 'tee'], help="Target object name")
    args = parser.parse_args()

    logger.info('start predict')

    iso = Iso(args, logger)
    iso.generate_iso()
    
    logger.info('end predict')
