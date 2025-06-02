import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
import re
from typing import Literal

import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k", "acid", "dtu", "llff", "tandt", "ns", "scannet"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    depth_map_path: str
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        # Revised by kamanri, the bounds are derive from `Metadata`, not config.
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # print(chunk_path)
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.train_times_per_scene if self.stage == "train" else self.cfg.test_times_per_scene
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics, nears, fars = self.convert_poses(example["cameras"])
                
                if self.cfg.name == "re10k":
                    # we need to put extra near & far bounds for re10k
                    # we dont use the near & far from here when training
                    nears = torch.ones_like(nears)
                    fars = torch.ones_like(fars) * 100.0

                scene = f"{self.cfg.name}_{example['key']}_{(run_idx % times_per_scene):02d}"

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        run_idx % times_per_scene, 
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    
                    fine_tune_indices = self.view_sampler.sample_fine_tune(run_idx % times_per_scene, scene, extrinsics, intrinsics)
                        
                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])
                    # print(context_indices)
                except Exception:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images, context_alphas = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images, target_alphas = self.convert_images(target_images)
                
                fine_tune_images, fine_tune_alphas = self.convert_images([
                    example["images"][index.item()] for index in fine_tune_indices
                ]) if fine_tune_indices != None else (None, None)
                
                # load depth from pfm file
                if self.cfg.name == "dtu" and self.stage == "train":
                    pfm_path = Path(self.cfg.depth_map_path) / example['key'][:example['key'].index("_")]
                    context_depth_maps = [read_pfm(str(pfm_path / f"depth_map_{(index // 7).item():04d}.pfm"))[0] for index in context_indices]
                    context_depth_masks = [np.array(depth_map != 0., dtype=np.float32) for depth_map in context_depth_maps]
                    # downsample to 512 * 640
                    context_depth_maps = [cv2.resize(depth_map, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) for depth_map in context_depth_maps]
                    context_depth_maps = [depth_map[44:556, 80:720] for depth_map in context_depth_maps]
                    context_depth_maps = torch.stack([torch.from_numpy(depth_map.copy()) for depth_map in context_depth_maps], dim=0)
                    context_depth_masks = [cv2.resize(depth_mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) for depth_mask in context_depth_masks]
                    context_depth_masks = [depth_mask[44:556, 80:720] for depth_mask in context_depth_masks]
                    context_depth_masks = torch.stack([torch.from_numpy(depth_mask.copy()) for depth_mask in context_depth_masks], dim=0)
                    # downscale to 1 / 200
                    context_depth_maps = context_depth_maps / 200.0
                    # resize & crop
                    resize_crop = tf.Compose([
                        tf.Resize(min(self.cfg.image_shape)), 
                        tf.CenterCrop(tuple(self.cfg.image_shape))
                    ])
                    context_depth_maps = resize_crop(context_depth_maps)
                    context_depth_masks = resize_crop(context_depth_masks)
                
                # load depth from VGGT prediction
                offline_depth = False
                if self.cfg.name == "re10k" and self.stage == "train":
                    if not offline_depth:
                        context_depth_maps, context_depth_confs = torch.tensor(0.), torch.tensor(0.) # real time prediction, set `use_vggt = True` on 'src/model/encoder/mvsnet/cas_mvsnet_module.py'
                    else:
                        # TODO: check its validation
                        scene_depth_path = Path(self.cfg.depth_map_path) / self.stage / f"{example['key']}.pt"
                        scene_depth_dict = torch.load(str(scene_depth_path), map_location="cpu") # load to cpu, NOT original device
                        context_timestamps = [example["timestamps"][i.item()] for i in context_indices]
                        context_depth_maps = torch.stack([scene_depth_dict[str(t.item())]["depth"] for t in context_timestamps]).float()
                        context_depth_confs = torch.ones(context_images.shape[0], self.cfg.image_shape[0], self.cfg.image_shape[1]) # (V, H, W)
                        # context_depth_confs = torch.stack([scene_depth_dict[t]["depth_conf"] for t in context_timestamps])
                        nears = context_depth_maps.reshape(context_images.shape[0], self.cfg.image_shape[0] * self.cfg.image_shape[1]).min(dim=-1).values * 0.8 # (V)
                        nears = torch.clamp(nears, min=0.1) # avoid too small near values (0) may be devided by zero in later calculations
                        fars = context_depth_maps.reshape(context_images.shape[0], self.cfg.image_shape[0] * self.cfg.image_shape[1]).max(dim=-1).values * 1.2 # (V)
                    pass

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                v, _, _, _ = context_images.shape
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "target_extrinsics": extrinsics[target_indices],
                        "target_intrinsics": intrinsics[target_indices],
                        "image": context_images,
                        "alpha": context_alphas, 
                        "depth": context_depth_maps if self.stage == "train" else torch.tensor(0.), 
                        "depth_conf": context_depth_confs if self.cfg.name == "re10k" and self.stage == "train" else torch.tensor(0.), 
                        "depth_mask": context_depth_masks if self.cfg.name == "dtu" and self.stage == "train" else torch.ones(v, self.cfg.image_shape[0], self.cfg.image_shape[1]), 
                        "near": nears[context_indices] / nf_scale,
                        "far": fars[context_indices] / nf_scale,
                        "index": context_indices,
                    },
                    "fine_tune": {
                        "extrinsics": extrinsics[fine_tune_indices], 
                        "intrinsics": intrinsics[fine_tune_indices], 
                        "image": fine_tune_images, 
                        "alpha": fine_tune_alphas, 
                        "near": nears[fine_tune_indices] / nf_scale,
                        "far": fars[fine_tune_indices] / nf_scale,
                        "index": fine_tune_indices,
                    } if fine_tune_indices != None else {}, 
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "alpha": target_alphas, 
                        "near": nears[target_indices] / nf_scale,
                        "far": fars[target_indices] / nf_scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
        Float[Tensor, "batch"],  # nears
        Float[Tensor, "batch"],  # fars
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        
        nears, fars = poses[:, 4], poses[:, 5]
        return w2c.inverse(), intrinsics, nears, fars

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"]]:
        torch_images, torch_alphas = [], []
        for image in images:
            image = self.to_tensor(Image.open(BytesIO(image.numpy().tobytes()))) # remove head binaries
            c, h, w = image.shape
            torch_images.append(image[:3]) # (rgb) without (a)
            torch_alphas.append(image[-1] if c == 4 else torch.ones(h, w, device=image.device)) # a
            
        return torch.stack(torch_images), torch.stack(torch_alphas)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        # Note: we do not need to consider the situation of validation step due to that we've used `ValidationWrapper(dataset, 1)` as the validation dataset.
        return (
            len(self.index.keys()) * self.cfg.test_times_per_scene if self.stage == "test"
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )
