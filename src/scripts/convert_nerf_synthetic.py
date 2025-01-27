from io import BytesIO
import subprocess
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
import argparse
from tqdm import tqdm
import json

import os

from PIL import Image
from convert import ConvertDataset, Metadata, Example

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input dtu raw directory")
parser.add_argument("--output_dir", type=str, help="output directory")
args = parser.parse_args()

INPUT_IMAGE_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)



def read_cam_file(filename):
    scale_factor = 1.0 / 200

    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ")
    extrinsic = extrinsic.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ")
    intrinsic = intrinsic.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor
    near_far = [depth_min, depth_max]
    return intrinsic, extrinsic, near_far


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    if stage == "train": return []

    return ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    for cur_id in range(0, 100):
        cur_image_name = f"r_{cur_id}.png"
        img_bin = load_raw(example_path / cur_image_name)
        images_dict[cur_id] = img_bin

    return images_dict


def load_metadata(metadata_path: Path, images: dict[int, Tensor]) -> Metadata:
    
    with open(metadata_path) as f:
        meta = json.load(f)
    
    timestamps = []
    cameras = []
    url = ""
    vid = 0
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    for frame in meta["frames"]:
        
        c2w = np.array(frame['transform_matrix']) @ blender2opencv
        w2c = np.linalg.inv(c2w)
        
        w, h = 800, 800
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        focal *= w / 800  # modify focal length to match size self.img_wh

        # normalized the intr
        fx = focal
        fy = focal
        cx = w / 2
        cy = h / 2
        # w = 2.0 * cx
        # h = 2.0 * cy
        w, h = Image.open(BytesIO(images[vid].numpy().tobytes())).size
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = cx / w
        saved_cy = cy / h
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))
        
        timestamps.append(int(vid))
        vid += 1

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }



class ConvertNeRFSynthetic(ConvertDataset):
    def get_image_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key / "train"
    
    def get_metadeta_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key / f"transforms_train.json"
    
    def get_output_dir(self, stage):
        return OUTPUT_DIR / stage
    
    def get_example_keys(self, stage):
        return get_example_keys(stage)
    
    def load_images(self, image_path):
        return load_images(image_path)
    
    def load_metadata(self, metadata_path, images):
        return load_metadata(metadata_path, images)
    pass

if __name__ == "__main__":
    ConvertNeRFSynthetic().exec()