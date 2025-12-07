# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Modified by Dongchen Han
# --------------------------------------------------------

from .vittt import vittt_tiny, vittt_small, vittt_base
from .h_vittt import h_vittt_tiny, h_vittt_small, h_vittt_base


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type in ['vittt_tiny', 'vittt_small', 'vittt_base']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE)')

    elif model_type in ['h_vittt_tiny', 'h_vittt_small', 'h_vittt_base']:
        model = eval(model_type + '(drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'use_checkpoint=config.TRAIN.USE_CHECKPOINT)')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
