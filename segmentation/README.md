# $\text{ViT}^3$ for Semantic Segmentaion

Our experiments are conducted on ADE20K dataset based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/).

## Results and Models

### UperNet

|  Backbone   |  Pretrain   | Lr Schd | mIoU (SS) | #params | FLOPs |                  config                  |                            model                             |
| :---------: | :---------: | :-----: | :-------: | :-----: | :---: | :--------------------------------------: | :----------------------------------------------------------: |
| H-ViT$^3$-T | ImageNet-1K |  160K   |   48.0    |  57.9M  | 946G  | [config](./configs/vittt/vittt_tiny.py)  | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/df39e508ae5b49c59cca/?dl=1) |
| H-ViT$^3$-S | ImageNet-1K |  160K   |   50.2    |  83.6M  | 1026G | [config](./configs/vittt/vittt_small.py) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/71c786dd7e8046b5ab43/?dl=1) |
| H-ViT$^3$-B | ImageNet-1K |  160K   |   51.7    |  124M   | 1195G | [config](./configs/vittt/vittt_base.py)  | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/6bd6db157003462f9fed/?dl=1) |


## Usage

### Installation

1. Clone the official [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/) repository (version 1.2.2) and follow its [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation.
2. Copy and merge our provided modified files into the corresponding directories of the cloned `mmsegmentation` repository to integrate our $\text{ViT}^3$ backbone.

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