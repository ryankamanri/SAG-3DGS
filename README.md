<p align="center">
  <h1 align="center">DMVA-GS: Depth-guided Generalizable 3D Gaussian Splatting with Multi-scale Voxel Attention Fusion</h1>
</p>

## Abstract
We propose DMVA-GS, a novel multi-scale 3D Gaussian Splatting (3DGS) feed-forward inference network that can accurately and efficiently reconstruct unseen scenes from sparse input views. Existing generalizable 3DGS methods typically rely on self-supervised multi-view stereo for depth estimation, and the number of Gaussians scaling linearly with the number of input views, which leads to performance deteriorates sharply as more views are provided when depth becomes unreliable. To address these limitations, we first introduce external depth supervision and a multi-scale architecture that improves robustness across diverse scene textures. Then, we incorporate geometric consistency constraints via reprojection to strengthen the geometric awareness of our network. In addition, we incorporate a voxelized representation and a novel voxel attention mechanism that aggregates Gaussians within each voxel, enabling strong control over Gaussian density and mitigating the over-growth problem. Experimental results on the RealEstate10K, ACID, DTU, Real Forward-Facing, NeRF Synthetic, and Tanks and Temples datasets demonstrate that DMVA-GS achieves state-of-the-art performance in cross-dataset generalization.

## Installation

test on Ubuntu 22.04.5 LTS.

make sure you installed cuda, conda and git:
``` bash
nvcc --version
conda --version
git --version
```

then run:
``` bash
git clone https://github.com/ryankamanri/SAG-3DGS
cd SAG-3DGS
conda create -p env/dmva python=3.10
conda activate env/dmva
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 # or cu121 if you have cuda 12.1, you must match your cuda version
pip install -r requirements.txt
git submodule update --init --recursive
pip install ./submodules/diff-gaussian-rasterization-modified --no-build-isolation
pip install ./submodules/vggt --no-build-isolation
```

and install pretrained [VGGT model](https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt) as `./pretrained/vggt/model.pt`, [our module](https://drive.google.com/file/d/10EiA6W_BUr1K47jG4Sh8DLd6ou-zUR8Z/view?usp=drive_link) as `./checkpoints/epoch_4-step_300000.ckpt`.


## Acquiring Datasets

### Real Estate10K and ACID

Our DMVA-GS uses the same training datasets as [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat). You can follow their instructions.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

### DTU, Real Forward-facing, NeRF Synthetic and Tanks and Temples

We provide the convert script in `src/scripts`, You could convert the origin dataset to pixelSplat's data chunks for its dataloader：

```bash
python src/scripts/convert_dtu.py --input_dir $DTU_PATH$ --output_dir datasets/dtu 
python src/scripts/convert_llff.py --input_dir $REAL_FORWARD_FACING_PATH$ --output_dir datasets/llff
python src/scripts/convert_nerf_synthetic.py --input_dir $NERF_SYNTHETIC_PATH$ --output_dir datasets/nerf_synthetic
python src/scripts/convert_tandt.py --input_dir $TANKS_AND_TEMPLES_PATH$ --output_dir datasets/tandt
```


## Running the Code

```bash
# train
python -m src.main +experiment=re10k

# test
# RealEstate 10K
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/epoch_4-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.re10k.view_sampler=evaluation \
test.compute_scores=true

# ACID
python -m src.main +experiment=acid \
checkpointing.load=checkpoints/epoch_4-step_300000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.re10k.view_sampler=evaluation \
test.compute_scores=true

# DTU 3-view
nohup python -m src.main +experiment=dtu \
model/encoder=incremental \
checkpointing.load=checkpoints/epoch_4-step_300000.ckpt \
test.num_context_views=3 \
dataset.view_sampler.mvsnerf.num_context_views_test=3 \
mode=test \
wandb.name=dtu/3view \
test.compute_scores=true \

# Real Forward-facing 3-view
nohup python -m src.main +experiment=llff \
model/encoder=incremental \
checkpointing.load=checkpoints/epoch_4-step_300000.ckpt \
test.num_context_views=3 \
dataset.view_sampler.mvsnerf.num_context_views_test=3 \
mode=test \
wandb.name=llff/3view \
test.compute_scores=true \

# NeRF Synthetic 3-view
nohup python -m src.main +experiment=ns \
model/encoder=incremental \
checkpointing.load=checkpoints/epoch_4-step_300000.ckpt \
test.num_context_views=3 \
dataset.view_sampler.mvsnerf.num_context_views_test=3 \
mode=test \
wandb.name=ns/3view \
test.compute_scores=true \

# Tanks and Temples 3-view
nohup python -m src.main +experiment=tandt \
model/encoder=incremental \
checkpointing.load=checkpoints/epoch_4-step_300000.ckpt \
test.num_context_views=3 \
dataset.view_sampler.mvsnerf.num_context_views_test=3 \
mode=test \
wandb.name=tandt/3view \
test.compute_scores=true \
```

* the rendered novel views will be stored under `outputs/test`
