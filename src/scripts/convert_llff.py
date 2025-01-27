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

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    # 草，这个地方源代码没有乘这个blender2opencv，做这个操作相当于把相机转换到另一个坐标系了，和一般的nerf坐标系不同
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    print('center in center_poses',poses_centered[:, :3, 3].mean(0))

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv

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

    return ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    cur_id = 0
    img_wh = (960, 640)
    src_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                                ])
    
    for image_path in example_path.iterdir():
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_wh, Image.LANCZOS)
        transform = T.ToTensor()
        img = transform(img)  # (3, h, w)
        images_dict[cur_id] = src_transform(img)
        cur_id += 1
    
    return images_dict


def load_metadata(metadata_path: Path, images: dict[int, Tensor]) -> Metadata:
    
    timestamps = []
    cameras = []
    url = ""
    vid = 0
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    poses_bounds = np.load(os.path.join(metadata_path, 'poses_bounds.npy'))  # (N_images, 17)
    image_paths = sorted(glob.glob(os.path.join(metadata_path, 'images/*')))
    # load full resolution image then resize

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)

    # Step 1: rescale focal length according to training resolution
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    print('original focal', focal)
    
    img_wh = (960, 640)
    focal = [focal * img_wh[0] / W, focal * img_wh[1] / H]
    print('porcessed focal', focal)

    # Step 2: correct poses
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    poses, _ = center_poses(poses, blender2opencv)
    
    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    near_original = bounds.min()
    scale_factor = near_original * 0.75  # 0.75 is the default parameter
    bounds /= scale_factor
    poses[..., 3] /= scale_factor

    

    w, h = img_wh


    for idx, img in enumerate(images):
        c2w = torch.eye(4).float()
        c2w[:3] = torch.FloatTensor(poses[idx])
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
        return INPUT_IMAGE_DIR / key / "images_4"
    
    def get_metadeta_dir(self, stage, key):
        return INPUT_IMAGE_DIR / key
    
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