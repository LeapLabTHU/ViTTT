import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from timm.models.registry import register_model
#from timm.models.vision_transformer import _cfg
from ..builder import BACKBONES
from ...utils import get_root_logger
#from mmengine.runner import _load_checkpoint
#from mmengine.runner import load_checkpoint as _load_checkpoint
import torch.utils.checkpoint as checkpoint

from mmengine.logging import MMLogger
from mmengine.model import (BaseModule, ModuleList, Sequential, constant_init,
                            normal_init, trunc_normal_init)
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmdet.registry import MODELS




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + self.dwc(x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

def rope(x, shape, base=10000):
    channel_dims, feature_dim = shape[:-1], shape[-1]
    k_max = feature_dim // (2 * len(channel_dims))

    assert feature_dim % k_max == 0

    # angles
    theta_ks = 1 / (base ** (torch.arange(k_max, device=x.device) / k_max))
    angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d, device=x.device) for d in channel_dims], indexing='ij')], dim=-1)

    # rotation
    rotations_re = torch.cos(angles).unsqueeze(dim=-1)
    rotations_im = torch.sin(angles).unsqueeze(dim=-1)

    x = x.reshape(*x.shape[:-1], -1, 2)
    x_re = x[..., :1]
    x_im = x[..., 1:]
    pe_x = torch.cat([x_re * rotations_re - x_im * rotations_im, x_im * rotations_re + x_re * rotations_im], dim=-1)
    return pe_x.flatten(-2)

class TTTAttention(nn.Module):

    def __init__(self, dim, input_resolution, **kwargs):

        super().__init__()
        head_dim = 32
        self.dim = dim
        self.num_heads = dim // head_dim

        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)
        self.proj = nn.Linear(dim + head_dim, dim)
    
    def gradient_gate_product_loss(self, k, v, w1, w2):
        # forward
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        # backward
        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        # clip gradient
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return g1, g2

    def gradient_3x3dwc_product_loss(self, k, v, implementation='conv'):
        # backward
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == 'conv':
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        else:
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        # clip gradient
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        return g
    def forward(self, x, hw_shape=None):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h, w = hw_shape
        assert n == h * w, "input feature has wrong size"
        d = c // self.num_heads
        num_heads = self.num_heads
        head_dim = d 
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)

        q1 = rope(q1.reshape(b, h, w, c), (h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k1 = rope(k1.reshape(b, h, w, c), (h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)

        w1, w2 = self.w1, self.w2
        g1, g2 = self.gradient_gate_product_loss(k1, v1, w1, w2)
        w1, w2 = w1 - g1, w2 - g2

        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)

        self.scale = 1 / 3.0
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        w3 = self.w3
        g3 = self.gradient_3x3dwc_product_loss(k2, v2, implementation='prod')
        w3 = w3.repeat(b, 1, 1, 1) - g3

        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = TTTAttention(dim=dim, input_resolution=input_resolution) # if input_resolution[0] > 7 else Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x, hw_shape=None):
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, dim_out, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        in_channels = dim
        out_channels = dim_out
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x, input_size):
        """
        x: B, H*W, C
        """
        H, W = input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.reshape(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = self.conv(x.permute(0, 3, 1, 2))
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).permute(0, 2, 1)

        return x, out_size



#@MODELS.register_module()

class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, dim_out, input_resolution, depth, mlp_ratio=4., drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Block(dim=dim, input_resolution=input_resolution, mlp_ratio=mlp_ratio, drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, dim_out=dim_out)
        else:
            self.downsample = None

    def forward(self, x, hw_shape=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, hw_shape)
            else:
                x = blk(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.patch_size = patch_size
        self.img_size = img_size

        self.conv = nn.Sequential(
            ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.conv(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x, out_size
@MODELS.register_module()
class ViTTT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, num_features=1024,
                 dim=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, out_indices=(0, 1, 2, 3), init_cfg=None,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dim[0]
        self.ape = ape
        self.num_features = [dim[i] for i in range(self.num_layers)]
        self.mlp_ratio = mlp_ratio
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        
        # Add a norm layer for each output
        for i in out_indices:
            stages_norm_layer = norm_layer(self.num_features[i])
            stages_norm_layer_name = f'norm{i}'
            self.add_module(stages_norm_layer_name, stages_norm_layer)

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim[0])
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=dim[i_layer],
                               dim_out=dim[i_layer + 1] if i_layer < self.num_layers - 1 else None,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)


        self.apply(self._init_weights)
    def init_weights(self):
        
        logger = MMLogger.get_current_instance()
        assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                f'specify `Pretrained` in ' \
                                                f'`init_cfg` in ' \
                                                f'{self.__class__.__name__} '
        checkpoint = CheckpointLoader.load_checkpoint(
            self.init_cfg.checkpoint, logger=logger, map_location='cpu')
        logger.warn(f'Load pre-trained model for '
                    f'{self.__class__.__name__} from original repo')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        #if self.convert_weights:
            # Because pvt backbones are not supported by mmpretrain,
            # so we need to convert pre-trained weights to match this
            # implementation.
         #   state_dict = pvt_convert(state_dict)
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x, hw_shape = self.patch_embed(x)

        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape, out, out_hw_shape = layer(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)

        return x


@MODELS.register_module()
def vittt_tiny(**kwargs):
    model = ViTTT(dim=[64, 128, 320, 512], depths=[1, 3, 9, 4], **kwargs)
    return model


@MODELS.register_module()
def vittt_small(**kwargs):
    model = ViTTT(dim=[64, 128, 320, 512], depths=[2, 6, 18, 8], **kwargs)
    return model


@MODELS.register_module()
def vittt_base(**kwargs):
    model = ViTTT(dim=[96, 192, 448, 640], depths=[2, 6, 18, 8], **kwargs)
    return model


