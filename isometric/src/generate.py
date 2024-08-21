import os
import sys
import argparse
from logging import getLogger, DEBUG, StreamHandler, Formatter

# Add the correct path for the isometric module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from connect import Connect
from isometric.utils.draw import DrawUtils

class Iso:
    """Isometric class"""
    def __init__(self, args, logger) -> None:
        self.__draw = DrawUtils(args, logger)
        self.__connect = Connect(args, logger)

    def generate_iso(self) -> None:
        """Generate isometric"""
        # self.__draw.init_pose_show()
        self.__connect.compute_piping_relationship()

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
