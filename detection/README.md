# $\text{ViT}^3$ for Object Detection

Our experiments are conducted on COCO dataset based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Results and Models

### MaskRCNN

|  Backbone   |  Pretrain   | Lr Schd | box mAP | mask mAP | #params | FLOPs |                       config                        |                            model                             |
| :---------: | :---------: | :-----: | :-----: | :------: | :-----: | :---: | :-------------------------------------------------: | :----------------------------------------------------------: |
| H-ViT$^3$-T | ImageNet-1K |   1x    |  47.3   |   42.8   |  48.4M  | 271G  | [config](./configs/vittt/vittt_t_mrcnn_1x.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/a00315bbfd504edebfb0/?dl=1) |
| H-ViT$^3$-S | ImageNet-1K |   1x    |  49.1   |   44.1   |  74.0M  | 349G  |       [config](./configs/vittt/vittt_s_mrcnn_1x.py)                                              | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/b42d36b21a5e44d0b266/?dl=1) |
| H-ViT$^3$-B | ImageNet-1K |   1x    |  50.0   |   44.6   |  114M   | 510G  |                        [config](./configs/vittt/vittt_b_mrcnn_1x.py)                            | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/4a641d087ca248e8804b/?dl=1) |
| H-ViT$^3$-T | ImageNet-1K |   3x    |  48.9   |   44.0   |  48.4M  | 271G  |                    [config](./configs/vittt/vittt_t_mrcnn_3x.py)                                | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8c39ef3d7d144a81b986/?dl=1) |
| H-ViT$^3$-S | ImageNet-1K |   3x    |  50.5   |   45.0   |  74.0M  | 349G  |                           [config](./configs/vittt/vittt_s_mrcnn_3x.py)                           | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/faa577d9a25140ef94d1/?dl=1) |
| H-ViT$^3$-B | ImageNet-1K |   3x    |  51.0   |   45.3   |  114M   | 510G  |             [config](./configs/vittt/vittt_b_mrcnn_3x.py)                                          | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/e9c1aef780d949489408/?dl=1) |


## Usage

### Installation

1. Clone the official [mmdetection](https://github.com/open-mmlab/mmdetection) repository (version 3.3.0) and follow its [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.
2. Copy and merge our provided modified files into the corresponding directories of the cloned `mmdetection` repository to integrate our $\text{ViT}^3$ backbone. 

### ImageNet-1K Pretrained Model

Please download our ImageNet-1K pretrained $\text{H-ViT}^3$ models and place them under `./data/` folder, e.g. `./data/H_VITTT_T.pth`.

### Inference

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> 

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> 
```

### Training

To train with pre-trained backbone models, run:
```
# multi-gpu training
torchrun --nproc_per_node <GPU_NUM> tools/train.py <CONFIG_FILE> --launcher="pytorch"
```

## Citation

If you find this repo helpful, please consider citing us.

```latex
@inproceedings{han2025vit,
  title={ViT$^3$: Unlocking Test-Time Training in Vision},
  author={Han, Dongchen and Li, Yining and Li, Tianyu and Cao, Zixuan and Wang, Ziming and Song, Jun and Cheng, Yu and Zheng, Bo and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```