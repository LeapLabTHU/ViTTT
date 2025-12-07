# $\text{ViT}^3$: Unlocking Test-Time Training in Vision

## Dependencies

- Python 3.9
- PyTorch==1.11.0
- torchvision==0.12.0
- numpy
- timm==0.4.12
- einops
- yacs

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Pretrained Models

| model  | Reso | Params | FLOPs | acc@1 | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\text{ViT}^3\text{-T}$ | $224^2$ | 6M | 1.2G | 76.5 | [config](cfgs/vittt_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/e12d1bd1c20c4e06a2da/?dl=1) |
| $\text{ViT}^3\text{-S}$ | $224^2$ | 24M | 4.8G | 81.6 | [config](cfgs/vittt_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/4ab59ca54b474dff9b70/?dl=1) |
| $\text{ViT}^3\text{-B}$ | $224^2$ | 90M | 18.0G | 82.6 | [config](cfgs/vittt_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/c9e9bb09f85d449ab8cf/?dl=1) |
| $\text{H-ViT}^3\text{-T}^‡$ | $224^2$ | 29M | 4.9G | 84.0 | [config](cfgs/h_vittt_t_mesa.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/f0c7c73ceabe49b7b161/?dl=1) |
| $\text{H-ViT}^3\text{-S}^‡$ | $224^2$ | 54M | 8.8G | 84.9 | [config](cfgs/h_vittt_s_mesa.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3eeab8621a7f43929a87/?dl=1) |
| $\text{H-ViT}^3\text{-B}^‡$ | $224^2$ | 94M | 16.7G | 85.5 | [config](cfgs/h_vittt_b_mesa.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/1d0d7fb840254275b29d/?dl=1) |

Evaluate $\text{ViT}^3$ on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

Evaluate $\text{H-ViT}^3$ on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

## Train Models from Scratch

- To train $\text{ViT}^3$ on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --amp
```

- To train $\text{H-ViT}^3$ on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --amp
```

## Citation

If you find this repo helpful, please consider citing us.

```latex
@article{han2025vit,
  title={ViT$^3$: Unlocking Test-Time Training in Vision},
  author={Han, Dongchen and Li, Yining and Li, Tianyu and Cao, Zixuan and Wang, Ziming and Song, Jun and Cheng, Yu and Zheng, Bo and Huang, Gao},
  journal={arXiv preprint arXiv:2512.01643},
  year={2025}
}
```
