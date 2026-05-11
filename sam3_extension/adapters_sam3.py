# sam3_extension/adapters_sam3.py
# -----------------------------------------------------------------------------
# Semantic adapters for SAM 3's ViT backbone.
#
# These mirror the design of SimAda/models/sam/modeling/common.py::Adapter_Text_Vis
# (and the regular vanilla Adapter), but are dimension-agnostic so they can plug
# into SAM 3's ViT, which differs from SAM-B in patch size, embed dim, depth and
# spatial resolution. Nothing in the original paper code is changed.
#
# Design (identical in spirit to the paper, see Sec. "Semantic Adapters"):
#
#   F_l  : (B, H_l, W_l, C_l)              <- SAM 3 ViT features entering MLP
#   V    : (B, 1+N, T_features)            <- CLIP patch+CLS tokens (T_features=512 for B/16, 768 for L/14)
#   t    : (B, T_features)                 <- CLIP text embedding
#   s    : (B, N, 1) (optional)            <- text-patch cosine similarity (broadcast added to V if used)
#
#   T'    = GELU(W_t t)                    -> (B, C_l) broadcast to (B,H_l,W_l,C_l)
#   U     = V (+ s)                        -> (B, 1+N, T_features)
#   U'    = U @ W_v  (drop CLS, reshape)   -> (B, side, side, C_l)
#   U_l   = interp(U', H_l, W_l)           -> (B, H_l, W_l, C_l)
#   F_l_hat = F_l + adapter_MLP(F_l + T' + U_l)   (+ skip conn)
#
# The adapter is placed in parallel to MLP inside each ViT block (see
# vit_adapter_inject.py), exactly as in your SAM-B pipeline.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """
    Vanilla (non-semantic) adapter. Mirrors common.Adapter from SimAda.
    Used in parallel to attention (no semantic input).
    """

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        # Zero-init final projection so adapter starts as a no-op.
        # nn.init.zeros_(self.D_fc2.weight)
        # if self.D_fc2.bias is not None:
        #     nn.init.zeros_(self.D_fc2.bias)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        #print(f"[debug] xs values: mean={xs.mean().item():.4f} std={xs.std().item():.4f} max={xs.max().item():.4f} min={xs.min().item():.4f}")
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Adapter_Text_Vis_SAM3(nn.Module):
    """
    Semantic adapter for SAM 3.

    Identical structure to common.Adapter_Text_Vis, with two changes for SAM 3:
      (1) D_features defaults to 1024 (SAM 3 ViT embed dim) instead of 768.
      (2) The interpolation target is a constructor arg (default 72x72 for
          SAM 3 at img_size=1008, patch_size=14), instead of hardcoded 64x64.

    Args:
        D_features: SAM 3 ViT channel dim (1024).
        T_features: CLIP feature dim (512 for ViT-B/16, 768 for ViT-L/14).
        spatial_target: (H_l, W_l) of SAM 3 ViT's feature grid.
                        For img_size=1008 patch_size=14 -> (72, 72).
        mlp_ratio: bottleneck ratio for the adapter MLP.
        skip_connect: same as paper code (typically False for the MLP_Adapter
                      and True for the Space_Adapter).
        has_cls_token: whether CLIP vision token includes CLS at index 0.
                       Open-CLIP/CLIP-Surgery: True.
    """

    def __init__(
        self,
        D_features: int = 1024,
        T_features: int = 512,
        spatial_target=(72, 72),
        mlp_ratio: float = 0.25,
        act_layer=nn.GELU,
        skip_connect: bool = True,
        has_cls_token: bool = True,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.has_cls_token = has_cls_token
        self.spatial_target = spatial_target
        D_hidden_features = max(1, int(D_features * mlp_ratio))
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.text_projection = nn.Linear(T_features, D_features)
        self.vis_projection = nn.Linear(T_features, D_features)

    def forward(self, x, text_embedding, clip_vis_embeddings):
        """
        x: (B, H_l, W_l, D)         - SAM 3 ViT feature map entering MLP
        text_embedding: (B, T_features)
        clip_vis_embeddings: (B, 1+N, T_features) if has_cls_token else (B, N, T_features)
        """
        # --- Project text and add to spatial grid via broadcast --------------
        projected_text = self.text_projection(text_embedding)                 # (B, D)
        projected_text = self.act(projected_text)
        projected_text = projected_text.unsqueeze(1).unsqueeze(1)             # (B, 1, 1, D)

        # --- Project CLIP visual tokens, drop CLS, reshape to grid -----------
        projected_vis = self.vis_projection(clip_vis_embeddings)              # (B, 1+N, D) or (B, N, D)
        if self.has_cls_token:
            patches = projected_vis[:, 1:, :]
        else:
            patches = projected_vis
        num_patches = patches.shape[1]
        side = int(num_patches ** 0.5)
        if side * side != num_patches:
            raise ValueError(
                f"Adapter_Text_Vis_SAM3 expects a square CLIP grid; got {num_patches}."
            )
        patches = patches.view(patches.shape[0], side, side, patches.shape[-1])    # (B, s, s, D)

        # Interpolate to SAM 3 ViT spatial dims (default 72x72)
        H_l, W_l = self.spatial_target
        if (side, side) != (H_l, W_l):
            patches = (
                patches.permute(0, 3, 1, 2)                                       # (B, D, s, s)
                .float()                                                          # bilinear is float
                .contiguous()
            )
            patches = F.interpolate(patches, size=(H_l, W_l), mode="bilinear", align_corners=False)
            patches = patches.permute(0, 2, 3, 1).to(x.dtype)                     # (B, H_l, W_l, D)
        else:
            patches = patches.to(x.dtype)

        # If x's spatial doesn't match self.spatial_target (e.g., user changed
        # img_size at runtime), align to x's actual spatial dims.
        if patches.shape[1:3] != x.shape[1:3]:
            patches = (
                patches.permute(0, 3, 1, 2).float().contiguous()
            )
            patches = F.interpolate(patches, size=(x.shape[1], x.shape[2]), mode="bilinear", align_corners=False)
            patches = patches.permute(0, 2, 3, 1).to(x.dtype)

        x_plus_t = x + projected_text
        x_plus_t_plus_v = x_plus_t + patches

        xs = self.D_fc1(x_plus_t_plus_v)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs + patches
        else:
            x = xs
        return x