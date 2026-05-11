#!/usr/bin/env python3
"""
debug_sam3_adapter_smoke_test_v2.py

Run from inside sam3_extension:

    python debug_sam3_adapter_smoke_test_v2.py \
        --device cuda \
        --adapter_blocks last1 \
        --sam3_image_size 1008 \
        --target_size 256

This version explicitly builds SAM3 with enable_inst_interactivity=True.
That is required for the SAM2-style interactive predictor branch.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def setup_import_paths():
    ext_dir = Path(__file__).resolve().parent
    parent = ext_dir.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))


def parse_adapter_blocks(s, depth):
    s = str(s).strip().lower()
    if s.startswith("last"):
        n = int(s.replace("last", ""))
        return list(range(max(0, depth - n), depth))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_dummy_target(batch_size, target_size, device):
    target = torch.zeros(batch_size, 1, target_size, target_size, device=device)
    y0, y1 = target_size // 4, 3 * target_size // 4
    x0, x1 = target_size // 4, 3 * target_size // 4
    target[:, :, y0:y1, x0:x1] = 1.0
    return target


def simple_bce_dice_loss(pred_logits, target):
    if pred_logits.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target.float(), size=pred_logits.shape[-2:], mode="nearest")

    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = pred_logits.sigmoid()
    inter = (pred * target).sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * inter + 1e-6) / (
        pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + 1e-6
    )
    return bce + dice.mean()


def collect_adapter_params(model):
    params = []
    for name, p in model.named_parameters():
        if ("Space_Adapter" in name or "MLP_Adapter" in name) and p.requires_grad:
            params.append((name, p))
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sam3_ckpt", type=str, default="")
    ap.add_argument("--adapter_blocks", type=str, default="last1")
    ap.add_argument("--sam3_image_size", type=int, default=1008)
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--t_features", type=int, default=512)
    ap.add_argument("--clip_grid", type=int, default=14)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    setup_import_paths()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[info] device={device}")

    from sam3.model_builder import build_sam3_image_model
    from sam3_extension.vit_adapter_inject import (
        inject_adapters_into_sam3_vit,
        _find_sam3_vit_trunk,
        set_semantic_inputs,
        clear_semantic_inputs,
    )
    from sam3_extension.freeze_sam3 import freeze_sam3_for_clip_guided, count_trainable_params
    from sam3_extension.sam3_clip_guided import _sam3_forward_with_points

    print("[1/8] Building SAM3 with enable_inst_interactivity=True ...")
    build_kwargs = dict(
        device=device,
        enable_inst_interactivity=True,
    )
    if args.sam3_ckpt:
        build_kwargs["checkpoint_path"] = args.sam3_ckpt

    sam3 = build_sam3_image_model(**build_kwargs)
    sam3 = sam3.to(device)

    print("[2/8] Checking interactive predictor...")
    print("  hasattr(model, 'inst_interactive_predictor'):", hasattr(sam3, "inst_interactive_predictor"))
    print("  model.inst_interactive_predictor is None:", sam3.inst_interactive_predictor is None)
    if sam3.inst_interactive_predictor is None:
        raise RuntimeError(
            "SAM3 was built without inst_interactive_predictor. "
            "Check whether your build_sam3_image_model supports enable_inst_interactivity=True."
        )

    print("[3/8] Locating ViT trunk...")
    trunk = _find_sam3_vit_trunk(sam3)
    if trunk is None:
        raise RuntimeError("Could not locate SAM3 ViT trunk.")
    depth = len(trunk.blocks)
    print(f"[ok] Found ViT trunk with {depth} blocks.")

    adapter_block_indices = parse_adapter_blocks(args.adapter_blocks, depth)
    print(f"[4/8] Injecting adapters into blocks: {adapter_block_indices}")
    carrier = inject_adapters_into_sam3_vit(
        sam3,
        adapter_block_indices=adapter_block_indices,
        use_space_adapter=True,
        use_semantic_adapter=True,
        T_features=args.t_features,
        spatial_target=(args.sam3_image_size // 14, args.sam3_image_size // 14),
        scale=0.5,
        has_cls_token=True,
        semantic_skip_connect=False,
    )
    carrier = carrier.to(device)

    print("[5/8] Freezing everything except adapters...")
    freeze_sam3_for_clip_guided(
        sam3,
        train_adapters=True,
        train_neck=False,
        train_text_encoder_last_n=0,
        train_transformer_encoder=False,
        train_transformer_decoder=False,
        train_segmentation_head=False,
    )

    total, trainable = count_trainable_params(sam3)
    print(f"[params] SAM3 total={total/1e6:.2f}M | trainable={trainable/1e6:.4f}M")

    adapter_params = collect_adapter_params(sam3)
    print(f"[params] trainable adapter tensors: {len(adapter_params)}")
    for name, p in adapter_params[:8]:
        print(f"  {name:80s} {tuple(p.shape)}")
    if not adapter_params:
        raise RuntimeError("No trainable adapter parameters found.")

    first_name, first_param = adapter_params[0]
    before = first_param.detach().clone()

    print("[6/8] Creating synthetic image, semantic embeddings, points, and mask...")
    B = 1
    imgs_sam3 = torch.randn(B, 3, args.sam3_image_size, args.sam3_image_size, device=device)

    num_clip_tokens = args.clip_grid * args.clip_grid + 1
    text_emb = torch.randn(B, args.t_features, device=device)
    vis_emb = torch.randn(B, num_clip_tokens, args.t_features, device=device)

    points = torch.tensor(
        [[[args.sam3_image_size * 0.50, args.sam3_image_size * 0.50],
          [args.sam3_image_size * 0.45, args.sam3_image_size * 0.45],
          [args.sam3_image_size * 0.55, args.sam3_image_size * 0.55]]],
        device=device,
        dtype=torch.float32,
    )
    point_labels = torch.ones(B, points.shape[1], device=device, dtype=torch.int32)

    target = make_dummy_target(B, args.target_size, device)

    print("[7/8] Forward through SAM3 backbone + interactive mask decoder...")
    sam3.train()

    # IMPORTANT:
    # Keep semantic inputs set until AFTER loss.backward().
    # SAM3 uses activation checkpointing, so parts of the forward graph may be
    # recomputed during backward. If we clear these attributes immediately after
    # the forward pass, recomputation sees a different graph and PyTorch raises:
    # "A different number of tensors was saved during the original forward and recomputation."
    set_semantic_inputs(sam3, text_emb, vis_emb)

    pred_logits = _sam3_forward_with_points(
        sam3,
        imgs_sam3,
        points,
        point_labels,
        target_hw=(args.target_size, args.target_size),
    )

    print(f"[ok] pred_logits shape: {tuple(pred_logits.shape)}")
    loss = simple_bce_dice_loss(pred_logits, target)
    print(f"[ok] loss: {loss.item():.6f}")

    print("[8/8] Backward and adapter gradient/update check...")
    optim = torch.optim.AdamW([p for _, p in adapter_params], lr=args.lr)
    optim.zero_grad(set_to_none=True)

    try:
        loss.backward()
    finally:
        clear_semantic_inputs(sam3)

    grads = []
    no_grad = []
    for name, p in adapter_params:
        if p.grad is None:
            no_grad.append(name)
        else:
            grads.append(float(p.grad.detach().abs().mean().item()))

    print(f"[grad] adapter tensors with grad: {len(grads)} / {len(adapter_params)}")
    if grads:
        print(f"[grad] mean(abs(grad)) average: {sum(grads)/len(grads):.8e}")
        print(f"[grad] max mean(abs(grad)): {max(grads):.8e}")

    if no_grad:
        print("[warn] Some adapter tensors had grad=None. First few:")
        for n in no_grad[:10]:
            print(" ", n)

    frozen_with_grad = []
    for name, p in sam3.named_parameters():
        if not p.requires_grad and p.grad is not None:
            frozen_with_grad.append(name)

    if frozen_with_grad:
        print("[warn] Frozen params unexpectedly had gradients. First few:")
        for n in frozen_with_grad[:10]:
            print(" ", n)
    else:
        print("[ok] No frozen SAM3 parameters received gradients.")

    optim.step()
    delta = (first_param.detach() - before).abs().max().item()
    print(f"[step] first adapter tensor changed max_abs_delta={delta:.8e} ({first_name})")

    if len(grads) == 0:
        raise RuntimeError("Smoke test failed: no adapter gradients.")
    if delta == 0:
        raise RuntimeError("Smoke test failed: optimizer did not update adapter weights.")

    print("\nSUCCESS: SAM3 adapter training path works with interactive predictor enabled.")


if __name__ == "__main__":
    main()
