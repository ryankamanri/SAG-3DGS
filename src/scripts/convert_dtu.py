''' Build upon: https://github.com/dcharatan/real_estate_10k_tools
                https://github.com/donydchen/matchnerf/blob/main/datasets/dtu.py 
    DTU Acquired instruction: https://github.com/donydchen/matchnerf?tab=readme-ov-file#dtu-for-both-training-and-testing'''

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
SPLIT_LIST_DIR = Path(os.curdir) / Path("config/dataset/view_sampler/mvsnerf_stored")

def build_camera_info(id_list, root_dir):
    """Return the camera information for the given id_list"""
    intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
    scale_factor = 1.0 / 200
    downSample = 1.0
    for vid in id_list:
        proj_mat_filename = os.path.join(
            root_dir, f"Cameras/train/{vid:08d}_cam.txt")
        intrinsic, extrinsic, near_far = read_cam_file(proj_mat_filename)

        # Note: I think any rescaling operation of intrinsics here is useless because it will be normalized when training
        intrinsic[:2] *= 4
        intrinsic[:2] = intrinsic[:2] * downSample
        intrinsics[vid] = intrinsic

        extrinsic[:3, 3] *= scale_factor
        world2cams[vid] = extrinsic
        cam2worlds[vid] = np.linalg.inv(extrinsic)

        near_fars[vid] = near_far

    return intrinsics, world2cams, cam2worlds, near_fars


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
    """ Extracted from: https://github.com/donydchen/matchnerf/blob/main/configs/dtu_meta/val_all.txt """
    keys = []
    with open(str(SPLIT_LIST_DIR / f"dtu_{stage}.lst"), 'r') as f:
        scan_xx = f.readline().rstrip() # remove '\n'
        while scan_xx != "":
            keys.append(f"{scan_xx}_train")
            scan_xx = f.readline()[:-1]
    print(f"Stage {stage} found {len(keys)} keys.")
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    for cur_id in range(1, 50):
        for light_id in range(0, 7):
            cur_image_name = f"rect_{cur_id:03d}_{light_id}_r5000.png"
            img_bin = load_raw(example_path / cur_image_name)
            images_dict[(cur_id - 1) * 7 + light_id] = img_bin

    return images_dict


def load_metadata(intrinsics, world2cams, images) -> Metadata:
    timestamps = []
    cameras = []
    url = ""

    for vid, intr in intrinsics.items():
        for light_id in range(0, 7):
            timestamps.append(int(vid) * 7 + light_id)

            # normalized the intr
            fx = intr[0, 0]
            fy = intr[1, 1]
            cx = intr[0, 2]
            cy = intr[1, 2]
            # w = 2.0 * cx
            # h = 2.0 * cy
            w, h = Image.open(BytesIO(images[vid * 7 + light_id].numpy().tobytes())).size
            saved_fx = fx / w
            saved_fy = fy / h
            saved_cx = cx / w
            saved_cy = cy / h
            camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

            w2c = world2cams[vid]
            camera.extend(w2c[:3].flatten().tolist())
            cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


class ConvertDTU(ConvertDataset):
    def get_image_dir(self, stage, key):
        return INPUT_IMAGE_DIR / "Rectified" / key
    
    def get_metadeta_dir(self, stage, key):
        return INPUT_IMAGE_DIR / "Cameras" / "train"
    
    def get_output_dir(self, stage):
        return OUTPUT_DIR / stage
    
    def get_example_keys(self, stage):
        return get_example_keys(stage)
    
    def load_images(self, image_path):
        return load_images(image_path)
    
    def load_metadata(self, metadata_path, images):
        intrinsics, world2cams, cam2worlds, near_fars = build_camera_info(
            list(range(49)), INPUT_IMAGE_DIR
        )
        return load_metadata(intrinsics, world2cams, images)
    pass

if __name__ == "__main__":
    ConvertDTU().exec()