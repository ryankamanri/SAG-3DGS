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


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    if stage == "train": return []

    return ["scene0101_04", "scene0241_01"]


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
        images_dict[cur_id] = img_bin
        cur_id += 1
    
    return images_dict


def load_metadata(metadata_path: Path, images: dict[int, Tensor]) -> Metadata:
    
    timestamps = []
    cameras = []
    url = ""
    vid = 0
    
    proj_mat_filename = metadata_path / f'{vid:08d}_cam.txt'
    
    intrinsic = np.loadtxt(metadata_path / "intrinsic" / "intrinsic_color.txt").astype(np.float32)[:3,:3]

    for _, idx in enumerate(images.keys()):
        
        w, h = Image.open(BytesIO(images[idx].numpy().tobytes())).size
        focal = [intrinsic[0, 0], intrinsic[1, 1]]
        
        extrinsics = np.loadtxt(metadata_path / "pose" / f"{idx}.txt").astype(np.float32)
        c2w = torch.eye(4).float()
        c2w[:3] = torch.FloatTensor(extrinsics[:3])
        w2c = torch.inverse(c2w)
        
        # normalized the intr
        fx = focal[0]
        fy = focal[1]
        cx = w / 2
        cy = h / 2
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



class ConvertScanNet(ConvertDataset):
    def get_image_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key / "exported" / "color"
    
    def get_metadeta_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key / "exported"
    
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
    ConvertScanNet().exec()