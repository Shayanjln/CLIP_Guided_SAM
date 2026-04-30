# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock, Adapter, Adapter_Text, Adapter_Text_Gated


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        args,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        clip_vit=None, 
        sidecar_last_k=1, 
        sidecar_blocks=(11,),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.args = args

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        self.shared_text_proj = nn.Linear(512, embed_dim)  # shared 512→D

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
                #torch.zeros(1, self.img_size // patch_size, self.img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()

        ####
        for i in range(depth):
            if i>= 8 and i<11:
                use_attn_adapter = True
            else:
                use_attn_adapter = False

            block = Block_Text(
                args=self.args,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                # NEW: only allocate clip-fuse params for blocks that will get clip_feat
                use_clip_fuse=(i in sidecar_blocks),
                use_attn_adapter = use_attn_adapter
            )
            self.blocks.append(block)           

        
        self.sidecar_blocks = set(sidecar_blocks)
        self.clip_sidecar = None
        if clip_vit is not None:
            self.clip_sidecar = CLIPSidecarFromSAM(
                clip_vit=clip_vit,
                sam_dim=embed_dim,
                last_k=sidecar_last_k,
                rank_out=32,           # set >0 for low-rank out-proj
                freeze_clip=False
            )

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                          # (B,H,W,D)
        t_proj = self.shared_text_proj(text_embeddings)  # (B,D)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            clip_feat = None
            if (self.clip_sidecar is not None) and (i in self.sidecar_blocks):
                with torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
                    # NOTE: sidecar uses *current* x as input, so it tracks the
                    # features evolving through the stack (empirically better than raw image)
                    clip_feat = self.clip_sidecar(x)     # (B,H,W,D)

            x = blk(x, t_proj, clip_feat)               # ← NEW sig: pass clip_feat
            
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

class Block_Text(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        args,
        dim: int,
        num_heads: int,
        text_dim: int = 512,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_clip_fuse: bool = False,
        use_attn_adapter = False
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        #self.MLP_Adapter = Adapter_Text(dim, mlp_ratio=0.25, skip_connect=False)  # MLP-adapter, no skip connection
        # self.MLP_Adapter = Adapter_Text_Gated(
        #     D_features=dim,
        #     #T_features=512,
        #     mlp_ratio=0.25,          # change to 0.33 to be a touch wider than 0.25
        #     act_layer=act_layer,
        #     skip_connect=True,       # keep residual inside the adapter
        #     use_conv_refiner=True,   # enable DW+PW refiner
        #     gate_on="both",          # pre + post gating
        #     zero_init=True,          # safe start
        # )
        self.use_attn_adapter = use_attn_adapter
        if self.use_attn_adapter:
            self.Space_Adapter = Adapter(dim, mlp_ratio=0.25)  # with skip connection
        self.scale = scale
        #self.scale_mlp = nn.Parameter(torch.tensor(0.5))
        #self.mlp_adapter_proj = nn.Linear(dim*2, dim)
        #self.Depth_Adapter = Adapter(dim, mlp_ratio=0.25, skip_connect=False)  # no skip connection

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        self.has_clip_fuse = bool(use_clip_fuse)
        if self.has_clip_fuse:
            bottleneck = max(1, dim // 4)  # rank r; tune {dim//16, //8, //4}
            self.clip_fuse_down_x    = nn.Linear(dim, bottleneck, bias=False)
            self.clip_fuse_down_clip = nn.Linear(dim, bottleneck, bias=False)
            self.clip_fuse_up        = nn.Linear(bottleneck, dim, bias=False)
            #self.clip_fuse_gate      = nn.Parameter(torch.tensor(0.0))  # safe start
            self.ln_fuse             = norm_layer(dim)

    # def fuse_clip(self, x, clip_feat):
    #     """
    #     x, clip_feat: (B,H,W,D)
    #     Low-rank additive fusion with a gate.
    #     """
    #     z = self.clip_fuse_down_x(x) + self.clip_fuse_down_clip(clip_feat)
    #     z = F.gelu(z)
    #     z = self.clip_fuse_up(z)
    #     #return torch.sigmoid(self.clip_fuse_gate) * z
    #     return z
    def fuse_clip(self, x, clip_feat, t_proj, tau: float = 0.07, stopgrad_weights: bool = False):
        """
        x, clip_feat: (B,H,W,D)  |  t_proj: (B,D)
        Returns a text-aware low-rank fused residual.
        """
        B, H, W, D = x.shape

        # --- cosine sims to build per-pixel weights ---
        x_dir    = F.normalize(x, dim=-1)
        clip_dir = F.normalize(clip_feat, dim=-1)
        t_dir    = F.normalize(t_proj, dim=-1).view(B, 1, 1, D)

        sim_x = (x_dir * t_dir).sum(dim=-1, keepdim=True)       # (B,H,W,1)
        sim_c = (clip_dir * t_dir).sum(dim=-1, keepdim=True)    # (B,H,W,1)

        sims = torch.cat([sim_x, sim_c], dim=-1)                # (B,H,W,2)
        if stopgrad_weights:
            sims = sims.detach()                                # optional: stabilize

        w = torch.softmax(sims / tau, dim=-1)                   # (B,H,W,2)
        w_x, w_c = w[..., :1], w[..., 1:]                       # split

        # --- low-rank fuse with text-aware weights ---
        z = self.clip_fuse_down_x(x) * w_x + self.clip_fuse_down_clip(clip_feat) * w_c
        z = F.gelu(z)
        z = self.clip_fuse_up(z)
        return z

    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor, clip_feat: torch.Tensor | None = None) -> torch.Tensor:
        shortcut = x
        # Window partition (unchanged)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # --- Attn + Space_Adapter (unchanged) ---
        x = self.norm1(x)
        x = self.attn(x)

        if self.use_attn_adapter:
            x = x + self.scale * self.Space_Adapter(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        xn = self.norm2(x)
        x = x + self.mlp(xn)

        #adapter_feats, _sim = self.MLP_Adapter(xn, text_embeddings)
        #x = x + self.scale * adapter_feats

        # --- NEW: fuse CLIP sidecar if provided ---
        # if self.has_clip_fuse and (clip_feat is not None):
        #     #xf = self.ln_fuse(x)
        #     x  = x + self.fuse_clip(x, clip_feat)
        if self.has_clip_fuse and (clip_feat is not None):
            xf = self.ln_fuse(x)              # keep the pre-fuse LN
            x  = x + self.fuse_clip(xf, clip_feat, text_embeddings, tau=1, stopgrad_weights=False)

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def closest_numbers(target):
    a = int(target ** 0.5)
    b = a + 1
    while True:
        if a * b == target:
            return (a, b)
        elif a * b < target:
            b += 1
        else:
            a -= 1


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPSidecarFromSAM(nn.Module):
    """
    Forward SAM's patch grid through K of CLIP's last resblocks (frozen), then
    return a (B,H,W,D_sam) feature map aligned to SAM's grid.

    Param-efficient: train only proj_in/proj_out; keep CLIP blocks frozen.
    """
    def __init__(self, clip_vit, sam_dim: int, last_k: int = 2, rank_out: int = 0, freeze_clip: bool = True):
        super().__init__()
        self.clip_width = clip_vit.embed_dim          # e.g., 768 for ViT-B/16
        #print('CLIP dim: ',clip_vit.embed_dim)
        self.sam_dim = sam_dim                        # e.g., 768 for SAM-B
        #print('SAM dim: ',sam_dim)
        self.last_k = last_k

        # ---- tap CLIP embeddings + norms (frozen) ----
        # (we shallow-copy modules to keep buffers/params tied; then freeze)
        self.class_embedding = clip_vit.class_embedding
        self.positional_embedding = clip_vit.positional_embedding
        self.ln_pre  = clip_vit.ln_pre
        self.ln_post = clip_vit.ln_post

        # ---- select the last K blocks as sidecar ----
        all_blocks = clip_vit.transformer.resblocks
        self.blocks = nn.ModuleList([all_blocks[i] for i in range(len(all_blocks)-last_k, len(all_blocks))])

        if freeze_clip:
            for p in self.blocks.parameters(): p.requires_grad = False
            for p in self.ln_pre.parameters(): p.requires_grad = False
            for p in self.ln_post.parameters(): p.requires_grad = False
            self.class_embedding.requires_grad = False
            self.positional_embedding.requires_grad = False

        # ---- light projections (trainable) ----
        # Project SAM features -> CLIP width (if same width, make this identity-initialized)
        # self.proj_in  = nn.Linear(self.sam_dim, self.clip_width, bias=False)
        # if self.sam_dim == self.clip_width:
        #     with torch.no_grad():
        #         self.proj_in.weight.copy_(torch.eye(self.sam_dim))

        # Low-rank out projection (optional): CLIP -> r -> SAM to save params
        # if rank_out and rank_out < min(self.clip_width, self.sam_dim):
        #     self.proj_mid = nn.Linear(self.clip_width, rank_out, bias=False)
        #     self.proj_out = nn.Linear(rank_out, self.sam_dim, bias=False)
        # else:
        #     self.proj_mid = None
        #     self.proj_out = nn.Linear(self.clip_width, self.sam_dim, bias=False)

        # safe, slow ramp-in
        self.sidecar_scale = nn.Parameter(torch.tensor(0.0))  # will be sigmoided

    @torch.no_grad()
    def _interp_pos(self, tokens: torch.Tensor):
        """
        Interpolate CLIP's absolute pos emb to match current grid side.
        tokens: (B, N+1, D)
        """
        pos = self.positional_embedding
        side = int((pos.shape[0] - 1) ** 0.5)
        new_side = int((tokens.shape[1] - 1) ** 0.5)
        if side == new_side:
            return pos
        new_pos = pos[1:, :].reshape(side, side, -1).permute(2, 0, 1)          # (D, S, S)
        new_pos = F.interpolate(new_pos.unsqueeze(0), (new_side, new_side), mode='bilinear', align_corners=False)
        new_pos = new_pos.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, -1)  # (N, D)
        return torch.cat([pos[:1, :], new_pos], dim=0)  # (N+1, D)

    def forward(self, x_bhwc: torch.Tensor):
        """
        x_bhwc: (B, H, W, D_sam)  -> returns clip-enhanced (B, H, W, D_sam)
        """
        B, H, W, D = x_bhwc.shape
        assert D == self.sam_dim, "SAM feature dim mismatch"

        # (B, H*W, D_sam) -> to CLIP width
        x = x_bhwc.reshape(B, H*W, D)

        #x = self.proj_in(x)  # (B, N, D_clip)

        # prepend CLS and add pos
        cls = self.class_embedding.to(x.dtype)[None, None, :].expand(B, 1, -1)    # (B,1,D_clip)
        x = torch.cat([cls, x], dim=1)                                            # (B, N+1, D_clip)
        pos = self._interp_pos(x)                                                 # (N+1, D_clip)
        x = x + pos.to(x.dtype)

        # CLIP forward expects (L, B, D)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # -> (N+1, B, D_clip)

        for blk in self.blocks:
            #x = blk(x)
            out = blk(x)
            if isinstance(out, (tuple, list)):
                x = out[0]     # take tokens; ignore weights/x_ori
            else:
                x = out

        x = self.ln_post(x)     # (N+1, B, D_clip)

        # drop CLS, back to grid
        x = x.permute(1, 0, 2)[:, 1:, :]                # (B, N, D_clip)
        # if self.proj_mid is not None:
        #     x = self.proj_out(self.proj_mid(x))         # (B, N, D_sam)
        # else:
        #     x = self.proj_out(x)

        x = x.reshape(B, H, W, self.sam_dim)            # (B,H,W,D_sam)
        #gate = torch.sigmoid(self.sidecar_scale)        # learned scalar gate
        #return gate * x
        return x


