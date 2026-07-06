from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import drop, drop_path, trunc_normal_
from mmseg.registry import MODELS

from mmseg.models.backbones import ResNet
from mmseg.models.backbones import VisionTransformer as MMVisionTransformer

from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck

import math
from timm.models.vision_transformer import VisionTransformer


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        # print("out's shape: ", out.shape)
        # print("identity's shape: ", identity.shape)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

@MODELS.register_module()
class CLIPResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, seg_cfg=False,
                  **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.seg_cfg = seg_cfg
        stride_list = [2, 2, 2, 2]
        dilation_list = [1, 1, 1, 1]
        if self.seg_cfg:
            stride_list = [2, 2, 1, 1]
            dilation_list = [1, 1, 2, 4]

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=stride_list[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=stride_list[1])
        self.layer3 = self._make_layer(width * 4, layers[2], stride=stride_list[2], dilation=dilation_list[2])
        self.layer4 = self._make_layer(width * 8, layers[3], stride=stride_list[3], dilation=dilation_list[3])

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)

        self.if_fpn = False

    def init_weights(self, pretrained=None):
        if_resize_pos = False
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            if_resize_pos = True
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            return if_resize_pos, u, w
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        layers = [Bottleneck(self._inplanes, planes, stride, dilation=dilation)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        outs.append(self.fpn1(x) if self.if_fpn else x)
        x = self.layer2(x)
        outs.append(self.fpn2(x) if self.if_fpn else x)
        x = self.layer3(x)
        outs.append(self.fpn3(x) if self.if_fpn else x)
        x = self.layer4(x)
        outs.append(self.fpn4(x) if self.if_fpn else x)

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])

        return tuple(outs)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


@MODELS.register_module()
class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512,
                  drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None,
                    get_embeddings=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.get_embed_proj = kwargs.get('get_embed_proj',False)

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        self.width = width
        self.patch_size = patch_size

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.embed_dim = width
        
    def init_weights(self, pretrained=None):
        if_resize_pos = False
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            # try:
            if 'RemoteCLIP' in pretrained:
                checkpoint = torch.load(pretrained, map_location='cpu')
            else :
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            # # finally :
            # checkpoint = torch.load(pretrained, map_location='cpu')

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    if_resize_pos = True
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    image_token_size = 224 // self.patch_size
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, image_token_size, image_token_size, self.width).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(self.width, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            return if_resize_pos, u, w

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :]
                if self.get_embed_proj:
                    xp = xp@ self.proj
                xp = xp.permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W

            features.append([global_embedding, visual_embedding])
            # print(global_embedding.shape, visual_embedding.shape)

        return tuple(features)
    
    def forward_pre(self,x:torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x
    
    def forward_post(self,x:torch.Tensor):
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj
        return x # [B,H*W+1,C]


class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=6,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        if_resize_pos = False
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            if 'RemoteCLIP' in pretrained:
                checkpoint = torch.load(pretrained, map_location='cpu')
            else :
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        if_resize_pos = True
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            return if_resize_pos, u, w


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = self.out_proj(x)
        return x


@MODELS.register_module()
class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=13,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 pretrained=None,
                 prompt_pos='pre',
                 if_fix_adapter=False,
                 **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim
        self.width = transformer_width
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.prompt_pos = prompt_pos

        self.if_fix_adapter = if_fix_adapter
        # if if_fix_adapter:
        #     for param in self.token_embedding.parameters():
        #         param.requires_grad = False

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            # try:
            if 'RemoteCLIP' in pretrained:
                checkpoint = torch.load(pretrained, map_location='cpu')
            else :
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            # finally :
            # checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = {}
            if_resize_pos = False

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                elif k == 'positional_embedding':
                    if checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is truncated from 77 to', self.context_length)
                        if_resize_pos = True
                    state_dict[k] = checkpoint[k]

                elif k == 'text_projection':
                    expected_shape = self.text_projection.shape
                    ckpt_shape = checkpoint[k].shape
                    if ckpt_shape != expected_shape:
                        print(f"Adapting text_projection: checkpoint {ckpt_shape} -> model {expected_shape}")
                        # 构造一个临时 Linear 层做映射
                        adaptor = nn.Linear(ckpt_shape[1], expected_shape[1], bias=False)
                        adaptor.weight.data = F.interpolate(
                            checkpoint[k].t().unsqueeze(0).unsqueeze(0),
                            size=(expected_shape[1],expected_shape[0]),
                            mode='bilinear'
                        ).squeeze()
                        adapted_weight = adaptor.weight.data.t()

                        state_dict[k] = adapted_weight
                    else:
                        state_dict[k] = checkpoint[k]

                elif k.startswith('token_embedding') or k.startswith('ln_final'):
                    state_dict[k] = checkpoint[k]

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

        return if_resize_pos, missing_keys, unexpected_keys

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    # text's shape: torch.Size([21, 5])   || context's shape: torch.Size([1, 8, 512])
    def forward(self, text, context=None):
        if context != None:
            
            x_text = self.token_embedding(text)  # n_clas, n_text, C torch.Size([21, 5, 512])

            K, N1, C = x_text.shape # K:1` N1:5 C:512
            B, N2, C = context.shape # B:1 N2:8 C:512

            eos_indx = text.argmax(dim=-1) + N2
            eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1) # shape: 21
            # tensor([10, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 10, 10, 10, 10, 12, 10, 10, 10, 11], device='cuda:0')
            
            x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
            context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

            if self.prompt_pos == 'pre':
                x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)
            elif self.prompt_pos == 'post':
                x = torch.cat([x_text[:, :, 0:-1], context, x_text[:, :, -1]], dim=2).reshape(B*K, N1+N2, C)
            elif self.prompt_pos == 'middle':
                x = torch.cat([x_text[:, :, 0:1], context[:, :, 0:N2/2], x_text[:, :, 1:-1], context[:, :, N2/2:],
                               x_text[:, :, -1]], dim=2).reshape(B*K, N1+N2, C)
                
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
            x = x.reshape(B, K, self.embed_dim)
        else:
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            # x = self.out_proj(x)

        return x

    def forward_pre(self, text, context=None):
        if context != None:
            
            x_text = self.token_embedding(text)  # n_clas, n_text, C torch.Size([21, 5, 512])

            K, N1, C = x_text.shape # K:1` N1:5 C:512
            B, N2, C = context.shape # B:1 N2:8 C:512

            eos_indx = text.argmax(dim=-1) + N2
            eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1) # shape: 21
            # tensor([10, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 10, 10, 10, 10, 12, 10, 10, 10, 11], device='cuda:0')
            
            x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
            context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

            if self.prompt_pos == 'pre':
                x = torch.cat([x_text[:, :, 0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)
            elif self.prompt_pos == 'post':
                x = torch.cat([x_text[:, :, 0:-1], context, x_text[:, :, -1]], dim=2).reshape(B*K, N1+N2, C)
            elif self.prompt_pos == 'middle':
                x = torch.cat([x_text[:, :, 0:1], context[:, :, 0:N2/2], x_text[:, :, 1:-1], context[:, :, N2/2:],
                               x_text[:, :, -1]], dim=2).reshape(B*K, N1+N2, C)
                
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            return x,eos_indx,(B,K)
        else:
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            eos_indx = text.argmax(dim=-1)
            K, N1, C = x.shape
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            return x,eos_indx,(-1,K)
    
    def forward_post(self,x,eos_indx,t_size:Tuple):
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(*t_size, self.embed_dim)
        return x

@MODELS.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 text_dim=512,
                 dropout=0.1,
                 if_decouple=False,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        if if_decouple:
            self.anti_decoder = nn.ModuleList([
                TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
            ])

            self.anti_out_proj = nn.Sequential(
                nn.LayerNorm(transformer_width),
                nn.Linear(transformer_width, visual_dim)
            )
        
        self.if_decouple = if_decouple

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def init_weights(self):
        for m in self.modules():
            self._init_weights(m)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        text = self.text_proj(text)

        for layer in self.decoder:
            text_diff = layer(text, visual)

        if self.if_decouple:
            for layer in self.anti_decoder:
                visual_diff = layer(visual, text)

            return self.out_proj(text_diff), self.anti_out_proj(visual_diff)

        return self.out_proj(text_diff)

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 合并 qkv 权重
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v,return_qk=False):
        B, N, C = q.shape
        assert k.shape == v.shape

        # 合并 qkv 计算
        qkv = self.qkv(q).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # 分离成 q, k, v [B, H, N, C/H]

        # 转换为 (B*H, N, D)
        q = q.transpose(1, 2).reshape(B * self.num_heads, N, C // self.num_heads)
        k = k.transpose(1, 2).reshape(B * self.num_heads, N, C // self.num_heads)
        v = v.transpose(1, 2).reshape(B * self.num_heads, N, C // self.num_heads)

        # 线性注意力公式：K^T·V -> A = Q @ (K^T·V)
        k = k.transpose(-1, -2)  # (B*H, D, N)
        kv = torch.bmm(k, v)     # (B*H, D, D)

        attn = torch.bmm(q, kv)  # (B*H, N, D)
        attn = attn / (C // self.num_heads)  # scale

        attn = attn.view(B, self.num_heads, N, C // self.num_heads)
        x = attn.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qk:
            q_for_qk = q.transpose(1, 2).reshape(B * self.num_heads, N, C // self.num_heads)
            k_for_qk = k.transpose(1, 2).reshape(B * self.num_heads, N, C // self.num_heads)
            qk_attn = (q_for_qk @ k_for_qk.transpose(-2, -1)) * self.scale
            qk_attn = qk_attn.view(B, self.num_heads, N, N)
            return x, qk_attn.sum(dim=1)/self.num_heads
        return x,None
    
class EffiTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        # 使用 EfficientAttention 替换原来的 Attention
        self.self_attn = EfficientAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = EfficientAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem,return_attn=False):
        q = k = v = self.norm1(x)
        tgt1,attn1 = self.self_attn(q, k, v,return_attn)
        x = x + tgt1
        q = self.norm2(x)
        tgt2,attn2 = self.cross_attn(q, mem, mem,return_attn)
        x = x + tgt2
        x = x + self.dropout(self.mlp(self.norm3(x)))
        if return_attn:
            return x,attn1
        return x,None
    
@MODELS.register_module()
class EffiContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    EffiTransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.anti_decoder = nn.ModuleList([
                EffiTransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
            ])

        self.anti_out_proj = nn.Sequential(
                nn.LayerNorm(transformer_width),
                nn.Linear(transformer_width, visual_dim)
            )
        self.transformer_layers = transformer_layers


    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def init_weights(self):
        for m in self.modules():
            self._init_weights(m)

    
    def forward(self, text, visual,return_attn=False):
        B, N, C = visual.shape
        visual_diff = self.memory_proj(visual)
        text_diff = self.text_proj(text)

        i = 0
        for layer in self.decoder:
            # if i == self.transformer_layers -1:
            #     text_diff,attn = layer(text_diff, visual_diff,return_attn)
            # else :
            text_diff,_ = layer(text_diff, visual_diff)
            i+=1
        i = 0
        for layer in self.anti_decoder:
            if i == self.transformer_layers -1:
                visual_diff,attn2 = layer(visual_diff, text_diff,return_attn)
            else :
                visual_diff,_ = layer(visual_diff, text_diff)
            i+=1

        if return_attn:
            return self.out_proj(text_diff), self.anti_out_proj(visual_diff),attn2
        return self.out_proj(text_diff), self.anti_out_proj(visual_diff),None
