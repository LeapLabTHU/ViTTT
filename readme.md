# $\text{ViT}^3$: Unlocking Test-Time Training in Vision

This repo contains the official PyTorch code and pre-trained models for **Vision Test-Time Training ($\text{ViT}^3$)**.

+ $\text{ViT}^3$: [Unlocking Test-Time Training in Vision](https://arxiv.org/abs/2512.01643)

## Abstract

<p align="center">
    <img src="figures/fig1_ttt.png" width= "400">
</p>

Test-Time Training (TTT) has recently emerged as a promising direction for efficient sequence modeling. TTT reformulates attention operation as an online learning problem, constructing a compact inner model from key-value pairs at test time. This reformulation opens a rich and flexible design space while achieving linear computational complexity. However, crafting a powerful visual TTT design remains challenging: fundamental choices for the inner module and inner training lack comprehensive understanding and practical guidelines. To bridge this critical gap, in this paper, we present a systematic empirical study of TTT designs for visual sequence modeling. From a series of experiments and analyses, we distill six practical insights that establish design principles for effective visual TTT and illuminate paths for future improvement. These findings culminate in the Vision Test-Time Training ($\text{ViT}^3$) model, a pure TTT architecture that achieves linear complexity and parallelizable computation. We evaluate $\text{ViT}^3$ across diverse visual tasks, including image classification, image generation, object detection, and semantic segmentation. Results show that $\text{ViT}^3$ consistently matches or outperforms advanced linear-complexity models (e.g., Mamba and linear attention variants) and effectively narrows the gap to highly optimized vision Transformers. We hope this study and the $\text{ViT}^3$ baseline can facilitate future work on visual TTT models.

## Usage

We provide a minimal implementation of $\text{ViT}^3$ block in [ttt_block.py](./ttt_block.py), which can act as a plug-in module in various vasion tasks. 

- Example:

```python
from ttt_block import TTT
block = TTT(dim=512, num_heads=16)
x = torch.rand(1, 256, 512)
x = block(x, h=16, w=16)
```

## Results

Please go to the folder [vittt](./vittt) for specific document.

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [MILA](https://github.com/LeapLabTHU/MLLA). 

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

## Contact

If you have any questions, please feel free to contact the authors.

Dongchen Han: [hdc23@mails.tsinghua.edu.cn](mailto:hdc23@mails.tsinghua.edu.cn)

