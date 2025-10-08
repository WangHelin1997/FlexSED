from einops import rearrange
from torch.cuda.amp import autocast
from functools import partial
from typing import Optional, Tuple
import torchaudio.transforms as audio_transforms
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from .dasheng import AudioPatchEmbed, Block

# if hasattr(nn.functional, 'scaled_dot_product_attention'):
#     ATTENTION_MODE = 'flash'
# else:
#     ATTENTION_MODE = 'math'
# print(f'attention mode is {ATTENTION_MODE}')


class Dasheng_Encoder(nn.Module):
    def __init__(self,
                 patch_size: Tuple[int, int] = (64, 4),
                 patch_stride: Tuple[int, int] = (64, 4),
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1008,
                 pooling='mean',
                 time_patch_out: Optional[float] = None,
                 freq_patch_out: Optional[float] = None,
                 block_type='Block',
                 attention_type='Attention',
                 eval_avg='cat',
                 n_fft: int = 512,
                 n_mels: int = 64,
                 hop_size: int = 160,
                 win_size: int = 512,
                 f_min: int = 0,
                 f_max: int = 8000,
                 center: bool = True,
                 **kwargs):
        super().__init__()
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = n_mels
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=win_size,
                                            center=center,
                                            n_fft=n_fft,
                                            f_max=f_max,
                                            hop_length=hop_size,
                                            n_mels=self.n_mels,
                                            power=1))

        self.to_db = audio_transforms.AmplitudeToDB(stype='magnitude', top_db=kwargs.get('top_db', 120))

        self.init_bn = nn.Sequential(
            Rearrange('b c f t -> b f c t'),
            nn.BatchNorm2d(self.n_mels, momentum=0.01),
            Rearrange('b f c t -> b c f t'))

        self.target_length = target_length
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels,
                                                       target_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)
        self.num_patches = self.patch_embed.num_patches

        if pooling == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(
                torch.randn(1, embed_dim) * .02)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=attention_type,
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self.init_weights)
        if hasattr(self, 'cls_token') and self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        # group_masking = kwargs.get('group_masking', False)
        # if isinstance(group_masking, bool):
        #     if group_masking is True:
        #         self.masking_func = self.random_masking_group
        #     else:
        #         self.masking_func = self.random_masking
        # elif isinstance(group_masking, int):
        #     self.masking_func = partial(self.random_masking_group,
        #                                 group_factor=group_masking)
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {
    #         'time_pos_embed', 'cls_token', 'freq_pos_embed', 'token_pos_embed'
    #     }

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just for sin pos embed
        x = rearrange(x, 'b c f t -> b (f t) c')
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # x, mask, ids_restore = self.masking_func(x, mask_ratio)
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        # x = self.norm(x)
        return x

    def load_state_dict(self, state_dict, **kwargs):
        if 'time_pos_embed' in state_dict and self.time_pos_embed.shape != state_dict[
                'time_pos_embed'].shape:
            print("Positional Embedding shape not the same with model, resizing!")
            self.change_pos_embedding(state_dict)
        # Call the parent class method and capture the missing/unexpected keys
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False, **kwargs)
        # Print missing and unexpected keys
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict['time_pos_embed']
        pretrained_freq_pos_embed = state_dict['freq_pos_embed']

        if target_freq_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length]
        else:
            state_dict['time_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode='bilinear')
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict[
                 'freq_pos_embed'] = pretrained_freq_pos_embed[:, :, :
                                                              target_freq_pos_embed_length, :]
        else:
            state_dict['freq_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode='bilinear')

    def forward_to_spec(self, x):
        # Do not use fp16 for feature extraction, that is likely to get nan
        with autocast(enabled=False):
            X = self.front_end(x)
            # X = rearrange(X, 'b f t -> b 1 f t')
            # X = self.init_bn(X)
        return X

    def forward(self, x):
        # x = self.forward_to_spec(x)
        # print(x.shape)
        with autocast(enabled=False):
            x = self.to_db(x)
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.init_bn(x)
        x = self.forward_features(x)
        return x