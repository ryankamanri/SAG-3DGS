from abc import ABC, abstractmethod
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm

INPUT_IMAGE_DIR = Path("/data/scene-rep/Real-Estate-10k")
INPUT_METADATA_DIR = Path("/data/scene-rep/Real-Estate-10k/metadata/RealEstate10K")
OUTPUT_DIR = Path("/data/scene-rep/Real-Estate-10k/re10k_pt")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)



class Metadata(TypedDict):
    """
    ### The metadata of dataset images
    item:
        `url`: what the image originally from (a video website), which may be useless if you do not specify it
        `timestamps`: the timestamp when this image is extracted from the video at, you can also fill a unique index.
        `cameras`: camera extrinsics and intrinsics with shape (18).
        
        Note that the structure of cameras is:
            `Tensor([fx/w, fy/h, cx/w, cy/h, 0.0, 0.0, w2c[0, :], w2c[1, :], w2c[2, :]])`
    """
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]

StageType = Literal["test", "train"]

class ConvertDataset(ABC):
    """
    ### A base class used to convert any dataset to pytorch chunks.
    You can inherit this class and rewrite some function of:
        - `get_image_dir`
        - `get_metadata_dir`
        - `get_output_dir`
        - `get_example_keys`
        - `get_size`
        - `load_images`
        - `load_metadata`
    """
    def get_image_dir(self, stage: StageType, key: str) -> Path:
        return INPUT_IMAGE_DIR / stage / key
    
    def get_metadeta_dir(self, stage: StageType, key: str) -> Path:
        return INPUT_METADATA_DIR / stage / f"{key}.txt"
    
    def get_output_dir(self, stage: StageType) -> Path:
        return OUTPUT_DIR / stage
    
    def get_example_keys(self, stage: StageType) -> list[str]:
        image_keys = set(
            example.name
            for example in tqdm((INPUT_IMAGE_DIR / stage).iterdir(), desc="Indexing images")
        )
        metadata_keys = set(
            example.stem
            for example in tqdm(
                (INPUT_METADATA_DIR / stage).iterdir(), desc="Indexing metadata"
            )
        )

        missing_image_keys = metadata_keys - image_keys
        if len(missing_image_keys) > 0:
            print(
                f"Found metadata but no images for {len(missing_image_keys)} examples.",
                file=sys.stderr,
            )
        missing_metadata_keys = image_keys - metadata_keys
        if len(missing_metadata_keys) > 0:
            print(
                f"Found images but no metadata for {len(missing_metadata_keys)} examples.",
                file=sys.stderr,
            )

        keys = image_keys & metadata_keys
        print(f"Found {len(keys)} keys.")
        return keys


    def get_size(self, path: Path) -> int:
        """Get file or folder size in bytes."""
        return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


    def load_images(self, image_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
        """Load JPG images as raw bytes (do not decode)."""
        return {int(path.stem): torch.tensor(np.memmap(path, dtype="uint8", mode="r")) for path in image_path.iterdir()}


    def load_metadata(self, metadata_path: Path, images: dict[int, Tensor]) -> Metadata:
        with metadata_path.open("r") as f:
            lines = f.read().splitlines()

        url = lines[0]

        timestamps = []
        cameras = []

        for line in lines[1:]:
            timestamp, *camera = line.split(" ")
            timestamps.append(int(timestamp))
            cameras.append(np.fromstring(",".join(camera), sep=","))

        timestamps = torch.tensor(timestamps, dtype=torch.int64)
        cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

        return {
            "url": url,
            "timestamps": timestamps,
            "cameras": cameras,
        }
        
        
    def exec(self):
        for stage in ("train", "test"):
            keys = self.get_example_keys(stage)

            chunk_size = 0
            chunk_index: int = 0
            chunk: list[Example] = []

            def save_chunk(chunk_size, chunk_index, chunk):
                chunk_key = f"{chunk_index:0>6}"
                # print(
                #     f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
                # )
                dir = self.get_output_dir(stage)
                dir.mkdir(exist_ok=True, parents=True)
                torch.save(chunk, dir / f"{chunk_key}.torch")

                # Reset the chunk.
                return 0, chunk_index + 1, []

            for key in tqdm(keys, desc="Generate chunks"):
                image_dir = self.get_image_dir(stage, key)
                metadata_dir = self.get_metadeta_dir(stage, key)
                num_bytes = self.get_size(image_dir)

                # Read images and metadata.
                images = self.load_images(image_dir)
                example = self.load_metadata(metadata_dir, images)

                # Merge the images into the example.
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
                assert len(images) == len(example["timestamps"])

                # Add the key to the example.
                example["key"] = key

                # print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
                chunk.append(example)
                chunk_size += num_bytes

                if chunk_size >= TARGET_BYTES_PER_CHUNK:
                    chunk_size, chunk_index, chunk = save_chunk(chunk_size, chunk_index, chunk)

            if chunk_size > 0:
                chunk_size, chunk_index, chunk = save_chunk(chunk_size, chunk_index, chunk)
                
            # generate index
            print("Generate key:torch index...")
            index = {}
            stage_path = self.get_output_dir(stage)
            for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
                if chunk_path.suffix == ".torch":
                    chunk = torch.load(chunk_path)
                    for example in chunk:
                        index[example["key"]] = str(chunk_path.relative_to(stage_path))
            with (stage_path / "index.json").open("w") as f:
                json.dump(index, f)
    pass


