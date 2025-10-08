import torch
import torch.nn as nn
import copy
from functools import partial
from .dasheng import LayerScale, Attention, Mlp


class Decoder_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_type='Attention',
        fusion='adaln',
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()

        self.fusion = fusion
        if fusion == 'adaln':
            self.adaln = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, x, c=None):
        B, T, C = x.shape

        if self.fusion == 'adaln':
            ada = self.adaln(c)
            (scale_msa, gate_msa, shift_msa,
             scale_mlp, gate_mlp, shift_mlp) = ada.reshape(B, 6, -1).chunk(6, dim=1)
            # self attention
            x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
            tanh_gate_msa = torch.tanh(1 - gate_msa)
            x = x + tanh_gate_msa * self.ls1(self.attn(x_norm))
            # mlp
            x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
            tanh_gate_mlp = torch.tanh(1 - gate_mlp)
            x = x + tanh_gate_mlp * self.ls2(self.mlp(x_norm))
        else:
            x = x + self.ls1(self.attn(self.norm1(x)))
            x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 2,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        cls_dim: int = 512,
        fusion: str = 'adaln',
        **kwargs
    ):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        init_values = None

        block_function = Decoder_Block
        self.blocks = nn.ModuleList([
            block_function(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type="Attention",
                fusion=fusion,
            ) for _ in range(depth)
        ])

        self.fusion = fusion
        cls_out = embed_dim

        self.cls_embed = nn.Sequential(
                nn.Linear(cls_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, cls_out, bias=True),)

        self.sed_head = nn.Linear(embed_dim, 1, bias=True)
        self.norm = norm_layer(embed_dim)
        self.apply(self.init_weights)
        # self.energy_head = nn.Linear(embed_dim, 1, bias=True)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

        if self.fusion == 'adaln':
            for block in self.blocks:
                nn.init.constant_(block.adaln.weight, 0)
                nn.init.constant_(block.adaln.bias, 0)

    def forward(self, x, cls):
        B, L, C = x.shape
        _, N, D = cls.shape
        # Expand x to shape (B, N, L, C)
        x = x.unsqueeze(1).expand(-1, N, -1, -1)
        # Reshape both tensors to (B*N, L, C) for processing
        x = x.reshape(B * N, L, C)
        cls = cls.reshape(B * N, D)

        cls = self.cls_embed(cls)

        shift = 0
        if self.fusion == 'adaln':
            pass
        elif self.fusion == 'token':
            cls = cls.unsqueeze(1)
            x = torch.cat([cls, x], dim=1)
            shift = 1
        else:
            raise NotImplementedError("unknown fusion")

        for block in self.blocks:
            x = block(x, cls)

        x = x[:, shift:]

        x = self.norm(x)

        strong = self.sed_head(x)
        return strong.transpose(1, 2)


class TSED_Wrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        ft_blocks=[11, 12],
        frozen_encoder=True
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        print("Loading Dasheng weights for decoders...")
        for i, blk_idx in enumerate(ft_blocks):
            decoder_block = self.decoder.blocks[i]
            encoder_block = self.encoder.blocks[blk_idx]
            state_dict = copy.deepcopy(encoder_block.state_dict())
            missing, unexpected = decoder_block.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"Block {blk_idx}:")
                if missing:
                    print(f"✅ Expected missing keys: {missing}")
                if unexpected:
                    print(f"  Unexpected keys: {unexpected}")
        # Copy norm_layer weights
        self.decoder.norm.load_state_dict(copy.deepcopy(self.encoder.norm.state_dict()))

        # Remove the injected blocks and norm_layer from the encoder
        for blk_idx in sorted(ft_blocks, reverse=True):
            # Reverse to avoid index shift issues
            del self.encoder.blocks[blk_idx]
        # Remove encoder norm layer
        del self.encoder.norm

        self.frozen_encoder = frozen_encoder
        if frozen_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward_to_spec(self, x):
        return self.encoder.forward_to_spec(x)

    def forward_encoder(self, x):
        if self.frozen_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        return x

    def forward(self, x, cls):
        x = self.forward_encoder(x)
        pred = self.decoder(x, cls)
        return pred
