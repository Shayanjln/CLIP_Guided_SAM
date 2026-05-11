# sam3_extension/vit_adapter_inject.py
# -----------------------------------------------------------------------------
# Inject semantic adapters into SAM 3's ViT image encoder, in parallel to MLP,
# WITHOUT changing any file under sam3/. We "monkey-patch" each Block at runtime:
#
#   F_l_after_attn = F_l + drop_path(ls1(attn(norm1(F_l))))
#   F_l_after_mlp  = F_l_after_attn + drop_path(ls2(mlp(norm2(F_l_after_attn))))
#                   + scale * adapter(norm2(F_l_after_attn), text_emb, vis_emb)   <-- NEW
#
# This mirrors your design in
# SimAda/models/sam/modeling/image_encoder_para_text_vis.py::Block_Vis.forward
# (semantic adapter parallel to MLP, regular adapter parallel to attention).
#
# We expose two callables that the wrapper sets on the ViT before forward:
#   model.image_encoder.trunk._sam3_text_emb         (B, T_features)
#   model.image_encoder.trunk._sam3_vis_emb          (B, 1+N, T_features)
# These are read inside each adapted block.
# -----------------------------------------------------------------------------

import math
import types
from typing import List, Optional

import torch
import torch.nn as nn

from .adapters_sam3 import Adapter, Adapter_Text_Vis_SAM3


class _AdapterCarrier(nn.Module):
    """Tiny module that *holds* the adapters so they appear in named_parameters."""

    def __init__(self, blocks_with_adapters: nn.ModuleList):
        super().__init__()
        self.blocks_with_adapters = blocks_with_adapters


def _patched_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Replacement for sam3.model.vitdet.Block.forward that adds:
      - regular adapter parallel to attention (Space_Adapter)   [optional]
      - semantic adapter parallel to MLP      (MLP_Adapter)     [main]

    The semantic inputs are read from attributes set on the parent ViT trunk:
      trunk._sam3_text_emb : (B, T_features) or None
      trunk._sam3_vis_emb  : (B, 1+N, T_features) or None
    If either is None, we treat the block as if there were no semantic
    injection (paper's "no semantic" mode = vanilla/regular only).

    Layout matches sam3/model/vitdet.py::Block.forward exactly EXCEPT for
    the two adapter contributions added in parallel.
    """
    from sam3.model.vitdet import window_partition, window_unpartition

    shortcut = x
    x_norm = self.norm1(x)

    if self.window_size > 0:
        H, W = x_norm.shape[1], x_norm.shape[2]
        x_norm, pad_hw = window_partition(x_norm, self.window_size)

    x_attn = self.ls1(self.attn(x_norm))

    if self.window_size > 0:
        x_attn = window_unpartition(x_attn, self.window_size, pad_hw, (H, W))

    x = shortcut + self.dropout(self.drop_path(x_attn))

    # Optional regular adapter parallel to ATTENTION (Space_Adapter), like in
    # the paper. Skip if not present (we register it conditionally below).
    if getattr(self, "Space_Adapter", None) is not None:
        x = x + self.scale_adp * self.Space_Adapter(x)

    # MLP path
    xn = self.norm2(x)
    x = x + self.dropout(self.drop_path(self.ls2(self.mlp(xn))))

    # Semantic adapter parallel to MLP (the core of the paper's contribution).
    if getattr(self, "MLP_Adapter", None) is not None:
        trunk = self._sam3_trunk_ref()
        text_emb = getattr(trunk, "_sam3_text_emb", None)
        vis_emb = getattr(trunk, "_sam3_vis_emb", None)
        if (text_emb is not None) and (vis_emb is not None):
            x = x + self.scale_adp * self.MLP_Adapter(xn, text_emb, vis_emb)
    return x


def inject_adapters_into_sam3_vit(
    sam3_model,
    adapter_block_indices: Optional[List[int]] = None,
    use_space_adapter: bool = True,
    use_semantic_adapter: bool = True,
    T_features: int = 512,
    spatial_target=(72, 72),
    scale: float = 0.5,
    has_cls_token: bool = True,
    space_skip_connect: bool = True,
    semantic_skip_connect: bool = False,
):
    """
    Walk SAM 3's ViT and add adapters to a chosen subset of blocks.

    Args:
        sam3_model: the model returned by build_sam3_image_model() (or any
                    container that has `.backbone` -> visual neck -> trunk
                    where trunk is the sam3 ViT). Falls back to inspecting
                    common attribute paths.
        adapter_block_indices: which block indices (0-based) get adapters.
                    Default: last 8 of 32 blocks, i.e. [24..31]. This is
                    proportional to your S=4/12 in SAM-B (33%) -> 8/32 (25%),
                    and includes the last global-attn block (31).
        use_space_adapter: add a vanilla Adapter parallel to attention.
        use_semantic_adapter: add Adapter_Text_Vis_SAM3 parallel to MLP.
        T_features: CLIP feature dim (512 for ViT-B/16, 768 for ViT-L/14).
        spatial_target: SAM 3 ViT feature grid (H_l, W_l). Default 72x72.
        scale: adapter contribution scale (paper uses 0.5).
        has_cls_token: whether CLIP vision tokens include CLS.
        space_skip_connect / semantic_skip_connect: same flags as in paper code.

    Returns:
        adapter_carrier: an nn.Module holding the new params (so they can be
                         added to optimizer / saved separately).
    """
    # 1) Locate the ViT trunk.
    trunk = _find_sam3_vit_trunk(sam3_model)
    if trunk is None:
        raise RuntimeError("Could not locate SAM 3 ViT trunk in the model.")

    blocks = trunk.blocks
    depth = len(blocks)
    D = blocks[0].mlp.fc1.in_features  # SAM 3 ViT embed_dim, typically 1024

    # if adapter_block_indices is None:
    #     adapter_block_indices = list(range(max(0, depth - 8), depth))
    #####################################################################
    # --- FOR DEBUGGING: INJECT INTO ALL BLOCKS ---
    if adapter_block_indices is None:
        adapter_block_indices = list(range(depth))  # --- INJECT INTO ALL BLOCKS ---

    print('injecting adapters into blocks:', adapter_block_indices)

    # 2) Make trunk carry the semantic embeddings as attributes.
    trunk._sam3_text_emb = None
    trunk._sam3_vis_emb = None
    # Weak ref so blocks can find the trunk without circular ownership.
    import weakref
    trunk_ref = weakref.ref(trunk)

    # 3) For each chosen block, attach the adapters and replace forward.
    blocks_with_adapters = nn.ModuleList()
    for idx, blk in enumerate(blocks):
        if idx not in adapter_block_indices:
            continue

        if use_space_adapter:
            blk.Space_Adapter = Adapter(
                D_features=D, mlp_ratio=0.25, skip_connect=space_skip_connect
            )
        else:
            blk.Space_Adapter = None

        if use_semantic_adapter:
            blk.MLP_Adapter = Adapter_Text_Vis_SAM3(
                D_features=D,
                T_features=T_features,
                spatial_target=spatial_target,
                mlp_ratio=0.25,
                skip_connect=semantic_skip_connect,
                has_cls_token=has_cls_token,
            )
        else:
            blk.MLP_Adapter = None

        blk.scale_adp = scale
        blk._sam3_trunk_ref = trunk_ref
        blk.forward = types.MethodType(_patched_block_forward, blk)

        blocks_with_adapters.append(blk)

    carrier = _AdapterCarrier(blocks_with_adapters)
    # Stash on the model for convenience (optimizer scheduling, save/load).
    sam3_model._sam3_adapters = carrier
    sam3_model._sam3_adapter_indices = adapter_block_indices
    return carrier


def _find_sam3_vit_trunk(sam3_model):
    """
    SAM 3 model graph (built by build_sam3_image_model()):

      sam3_model.backbone (SAM3VLBackbone)
        .visual                  (Sam3DualViTDetNeck)
            .trunk               (sam3.model.vitdet.ViT)   <-- target
            .other necks ...
        .text                    (VETextEncoder)

    We try several attribute paths to be robust.
    """
    candidates = [
        ("backbone", "visual", "trunk"),
        ("backbone", "vision_backbone", "trunk"),
        ("backbone", "visual", "vit"),
        ("backbone", "trunk"),
    ]
    for path in candidates:
        cur = sam3_model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and hasattr(cur, "blocks"):
            return cur
    return None


def set_semantic_inputs(sam3_model, text_emb: torch.Tensor, vis_emb: torch.Tensor):
    """Set the semantic embeddings to be consumed by the next ViT forward."""
    trunk = _find_sam3_vit_trunk(sam3_model)
    if trunk is None:
        raise RuntimeError("Could not locate SAM 3 ViT trunk.")
    trunk._sam3_text_emb = text_emb
    trunk._sam3_vis_emb = vis_emb


def clear_semantic_inputs(sam3_model):
    """Clear after a forward to avoid stale state across batches."""
    trunk = _find_sam3_vit_trunk(sam3_model)
    if trunk is None:
        return
    trunk._sam3_text_emb = None
    trunk._sam3_vis_emb = None