import glob
from io import BytesIO
import subprocess
from pathlib import Path
from typing import Literal, TypedDict
from torchvision import transforms as T
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

    return ["train", "truck"]


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    
    cur_id = 0
    for image_path in example_path.iterdir():
        cur_image_name = image_path
        img_bin = load_raw(cur_image_name)
        images_dict[int(cur_image_name.stem)] = img_bin
        cur_id += 1
    
    return images_dict


def load_metadata(metadata_path: Path, images: dict[int, Tensor]) -> Metadata:
    
    timestamps = []
    cameras = []
    url = ""
    vid = 0
    
    for _, idx in enumerate(images.keys()):
        
        proj_mat_filename = metadata_path / f'{idx:08d}_cam.txt'
        intrinsics, extrinsics, near_far = read_cam_file(proj_mat_filename)
        
        w, h = Image.open(BytesIO(images[idx].numpy().tobytes())).size
        focal = [intrinsics[0, 0], intrinsics[1, 1]]
        
        w2c = torch.eye(4).float()
        w2c[:3] = torch.FloatTensor(extrinsics[:3]) # the extrinsics is w2c, not c2w.
        
        # normalized the intr
        fx = focal[0]
        fy = focal[1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        # w = 2.0 * cx
        # h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = cx / w
        saved_cy = cy / h
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 1.0, 10.0]

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



class ConvertTanksandTemples(ConvertDataset):
    def get_image_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key / "images"
    
    def get_metadeta_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key / "cams_1"
    
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
    ConvertTanksandTemples().exec()