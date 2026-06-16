# $\text{ViT}^3$ for Image Generation

This directory contains the image generation code built on the official [DiT](https://github.com/facebookresearch/DiT) implementation.

## Results and Models

| Model | Dataset | Reso | Checkpoint | FID | IS | Prec. | Rec. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\text{DiT}^3$-B/2 | ImageNet-1K | $256^2$ | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/c68d33055c6b44f39584/?dl=1) | 39.44 | 37.22 | 0.51 | 0.63 |
| $\text{DiT}^3$-B/4 | ImageNet-1K | $256^2$ | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/49a2ea7a60624a8a94f3/?dl=1) | 65.25 | 22.28 | 0.37 | 0.55 |
| $\text{DiT}^3$-B/8 | ImageNet-1K | $256^2$ | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/d09682e5b50d40d48f4e/?dl=1) | 120.82 | 10.87 | 0.20 | 0.26 |

*Note that the original weights were lost during server migration. The checkpoints provided here were retrained, so the results differ slightly from those in the original paper.*

## Usage

### Installation

The environment setup is the same as in the official [DiT](https://github.com/facebookresearch/DiT) codebase.

```shell
conda env create -f environment.yml
conda activate DiTTT
```

### Training

To train a $\text{DiT}^3$-B/2 model on ImageNet 256 $\times$ 256, run:

```shell
torchrun --nnodes=1 --nproc_per_node=<GPU_NUM> train.py --model DiTTT-B/2 --data-path /path/to/imagenet/train --image-size 256 --global-batch-size 256
```

You can change `--model` to other supported DiT variants.

### Inference

To sample images from a trained checkpoint, run:

```shell
python sample.py --model DiTTT-B/2 --image-size 256 --ckpt /path/to/model.pt --cfg-scale 4.0 --num-sampling-steps 250 --seed 0
```

The generated image grid will be saved as `sample.png`.

### Evaluation

Following the official DiT evaluation pipeline, first generate 50,000 samples with `sample_ddp.py`:

```shell
torchrun --nnodes=1 --nproc_per_node=<GPU_NUM> sample_ddp.py --model DiTTT-B/2 --image-size 256 --ckpt /path/to/model.pt --num-fid-samples 50000 --sample-dir samples --cfg-scale 1 --num-sampling-steps 250
```

This will save the generated images and an `.npz` sample file under `samples/`. To compute FID, Inception Score, sFID, Precision, and Recall, use the TensorFlow evaluation suite from [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations):

```shell
python evaluator.py /path/to/VIRTUAL_imagenet256_labeled.npz /path/to/generated_samples.npz
```

Please refer to the guided-diffusion evaluation README for instructions on downloading the ImageNet reference batch.

## Citation

If you find this repo helpful, please consider citing us.

```bibtex
@inproceedings{han2025vit,
  title={ViT$^3$: Unlocking Test-Time Training in Vision},
  author={Han, Dongchen and Li, Yining and Li, Tianyu and Cao, Zixuan and Wang, Ziming and Song, Jun and Cheng, Yu and Zheng, Bo and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
