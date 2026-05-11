# sam3_extension/freeze_sam3.py
# -----------------------------------------------------------------------------
# Freezing utilities for SAM3 + CLIP-guided adapters.
#
# Updated version:
#   - can unfreeze SAM3Interactive prompt encoder
#   - can unfreeze SAM3Interactive mask decoder
#   - default flags are explicit; runner decides what to train per ablation
# -----------------------------------------------------------------------------

import torch.nn as nn

from .vit_adapter_inject import _find_sam3_vit_trunk


_ADAPTER_NAMES = ("Space_Adapter", "MLP_Adapter")


def freeze_sam3_for_clip_guided(
    sam3_model,
    train_adapters: bool = True,
    train_neck: bool = False,
    train_text_encoder_last_n: int = 0,
    train_transformer_encoder: bool = False,
    train_transformer_decoder: bool = False,
    train_segmentation_head: bool = False,
    train_interactive_prompt_encoder: bool = True,
    train_interactive_mask_decoder: bool = True,
):
    """
    Freeze SAM3, then selectively unfreeze the parts used in your CLIP-guided
    interactive/adapted training path.

    For your current setup, the important trainable SAM3 components are usually:
      - ViT adapters
      - SAM3Interactive prompt encoder
      - SAM3Interactive mask decoder

    Args:
        sam3_model:
            SAM3 image model. If using the interactive path, it must be built with
            enable_inst_interactivity=True.

        train_adapters:
            Unfreeze Space_Adapter and MLP_Adapter modules injected into the ViT.

        train_interactive_prompt_encoder:
            Unfreeze sam3_model.inst_interactive_predictor.model.sam_prompt_encoder.

        train_interactive_mask_decoder:
            Unfreeze sam3_model.inst_interactive_predictor.model.sam_mask_decoder.

        Other flags:
            Optional SAM3-native grounding components, mostly for experiments that
            use forward_grounding rather than the SAM2-style interactive path.
    """
    # 1) Freeze everything.
    for p in sam3_model.parameters():
        p.requires_grad = False

    # 2) ViT adapters.
    if train_adapters:
        trunk = _find_sam3_vit_trunk(sam3_model)
        if trunk is not None:
            for blk in trunk.blocks:
                for adp_name in _ADAPTER_NAMES:
                    adp = getattr(blk, adp_name, None)
                    if isinstance(adp, nn.Module):
                        for p in adp.parameters():
                            p.requires_grad = True

    # 3) Optional ViT neck.
    if train_neck:
        if hasattr(sam3_model, "backbone") and hasattr(sam3_model.backbone, "visual"):
            visual = sam3_model.backbone.visual
            trunk = _find_sam3_vit_trunk(sam3_model)
            trunk_param_ids = set(id(p) for p in trunk.parameters()) if trunk is not None else set()
            for _, p in visual.named_parameters():
                if id(p) not in trunk_param_ids:
                    p.requires_grad = True

    # 4) Optional SAM3 internal text encoder layers.
    if train_text_encoder_last_n > 0:
        text_enc = _find_sam3_text_encoder(sam3_model)
        if text_enc is not None:
            text_layers = _find_text_encoder_layers(text_enc)
            if text_layers is not None:
                n_total = len(text_layers)
                start = max(0, n_total - train_text_encoder_last_n)
                for i in range(start, n_total):
                    for p in text_layers[i].parameters():
                        p.requires_grad = True

    # 5) Optional SAM3 native grounding transformer/head.
    if train_transformer_encoder:
        if hasattr(sam3_model, "transformer") and hasattr(sam3_model.transformer, "encoder"):
            for p in sam3_model.transformer.encoder.parameters():
                p.requires_grad = True

    if train_transformer_decoder:
        if hasattr(sam3_model, "transformer") and hasattr(sam3_model.transformer, "decoder"):
            for p in sam3_model.transformer.decoder.parameters():
                p.requires_grad = True

    if train_segmentation_head and hasattr(sam3_model, "segmentation_head"):
        for p in sam3_model.segmentation_head.parameters():
            p.requires_grad = True

    # 6) SAM2/SAM1-style interactive branch used by sam3_clip_guided.py.
    interactive = getattr(sam3_model, "inst_interactive_predictor", None)
    if interactive is not None and hasattr(interactive, "model"):
        inner = interactive.model

        if train_interactive_prompt_encoder and hasattr(inner, "sam_prompt_encoder"):
            for p in inner.sam_prompt_encoder.parameters():
                p.requires_grad = True

        if train_interactive_mask_decoder and hasattr(inner, "sam_mask_decoder"):
            for p in inner.sam_mask_decoder.parameters():
                p.requires_grad = True

    return sam3_model


def freeze_clip_layers_keep_last_n(clip_model, trainable_layers: int = 4):
    """
    Freeze CLIP except the last N layers of the vision and text transformers.

    Set trainable_layers=0 for fully frozen CLIP.
    """
    for p in clip_model.parameters():
        p.requires_grad = False

    if trainable_layers <= 0:
        return clip_model


    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer"):
        rb = clip_model.visual.transformer.resblocks
        n = len(rb)
        for i in range(max(0, n - trainable_layers), n):
            for p in rb[i].parameters():
                p.requires_grad = True

    if hasattr(clip_model, "transformer") and hasattr(clip_model.transformer, "resblocks"):
        rb = clip_model.transformer.resblocks
        n = len(rb)
        for i in range(max(0, n - trainable_layers), n):
            for p in rb[i].parameters():
                p.requires_grad = True

    return clip_model

def freeze_clip_vision_attention_last_k(clip_model, k: int = 12):
    """
    Match original CLIP-Guided SAM:
      - freeze all CLIP parameters
      - unfreeze only attention params in last k visual transformer blocks
      - keep text encoder frozen
    """
    for p in clip_model.parameters():
        p.requires_grad = False

    if k <= 0:
        return clip_model

    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer"):
        rb = clip_model.visual.transformer.resblocks
        n = len(rb)
        for i in range(max(0, n - k), n):
            for name, p in rb[i].named_parameters():
                if "attn" in name:
                    p.requires_grad = True

    return clip_model


def _find_sam3_text_encoder(sam3_model):
    if hasattr(sam3_model, "backbone") and hasattr(sam3_model.backbone, "text"):
        return sam3_model.backbone.text
    if hasattr(sam3_model, "text_encoder"):
        return sam3_model.text_encoder
    return None


def _find_text_encoder_layers(text_enc):
    for path in [
        ("transformer", "resblocks"),
        ("transformer", "layers"),
        ("language_backbone", "transformer", "resblocks"),
        ("language_backbone", "transformer", "layers"),
        ("layers",),
        ("resblocks",),
    ]:
        cur = text_enc
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and hasattr(cur, "__len__"):
            return cur
    return None


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def print_trainable_summary(model, max_lines: int = 80):
    """
    Useful debug helper. Prints trainable parameter/module names.
    """
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"[trainable] tensors={len(names)}")
    for name in names[:max_lines]:
        print(" ", name)
    if len(names) > max_lines:
        print(f" ... ({len(names) - max_lines} more)")
