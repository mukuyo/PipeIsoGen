"""This is a generate isomeric program"""
import sys
import yaml
import argparse

from logging import getLogger, DEBUG, StreamHandler, Formatter

# from isometric.dxf.generate import GenDxf
from connect import Connect
# from isometric.src.distace import Distance

class Iso():
    """Isometric class"""
    def __init__(self, args, logger) -> None:
        self.__logger = logger
        self.__connect = Connect(args, logger)

    def generate_iso(self) -> None:
        """generate_isometric"""
        self.__connect.compute_piping_relationship()
    #     pare_results = self.__trans.facing_each_other(pose_results)
    #     if pare_results[0]:
    #         all_results = self.__trans.remain_pipes(pare_results, pose_results)
    #         sort_info = self.__trans.sort_results(all_results)
    #         isometric_info = self.__distance.get_info(sort_info)
    #         # self.__draw.line_2d(isometric_info)
    #         # self.__draw.isometric(isometric_info)
    #         self.__dxf.isometric(isometric_info)
    #         self.__logger.info('Complete making piping isometric drawing!!')
    #     else:
    #         self.__logger.info('Connected pipe not found')

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
    args = parser.parse_args()

    logger.info('start predict')

    iso = Iso(args, logger)
    iso.generate_iso()
    
    logger.info('end predict')

