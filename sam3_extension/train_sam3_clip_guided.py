# sam3_extension/train_sam3_clip_guided.py
# -----------------------------------------------------------------------------
# Training/evaluation for SAM3 + CLIP-guided interactive/adapted setup.
#
# Updated version:
#   - per-class mIoU, matching your usual protocol
#   - safe semantic-input clearing after backward
#   - supports zero-shot evaluation
#   - saves all trainable SAM3 parameters, including interactive prompt/mask decoder
# -----------------------------------------------------------------------------

import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .vit_adapter_inject import clear_semantic_inputs
from .sam3_clip_guided import forward_supervised_SAM3_CLIP


def bce_dice_iou_loss(pred_logits, target, bce_w=1.0, dice_w=1.0, iou_w=1.0, eps=1e-6):
    if target.shape[-2:] != pred_logits.shape[-2:]:
        target = F.interpolate(target.float(), size=pred_logits.shape[-2:], mode="nearest")

    bce = F.binary_cross_entropy_with_logits(pred_logits, target.float())

    pred_sig = pred_logits.sigmoid()
    inter = (pred_sig * target).sum(dim=(-2, -1))
    dice = 1.0 - (2 * inter + eps) / (
        pred_sig.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + eps
    )

    union = pred_sig.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - inter
    iou = 1.0 - (inter + eps) / (union + eps)

    return bce_w * bce + dice_w * dice.mean() + iou_w * iou.mean()


def compute_iou_per_image(pred_logits, target, thresh=0.5, eps=1e-6):
    """
    Returns IoU for each image-prompt pair as shape [B].
    """
    if target.shape[-2:] != pred_logits.shape[-2:]:
        target = F.interpolate(target.float(), size=pred_logits.shape[-2:], mode="nearest")

    pred = (pred_logits.sigmoid() > thresh)
    gt = target > 0.5

    inter = torch.logical_and(pred, gt).sum(dim=(-2, -1)).float()
    union = torch.logical_or(pred, gt).sum(dim=(-2, -1)).float()

    # pred/gt are [B,1,H,W], so result is [B,1]. Flatten to [B].
    return ((inter + eps) / (union + eps)).view(-1)


def _batch_class_names(batch):
    if "class_name" in batch:
        names = batch["class_name"]
    elif "text" in batch:
        names = batch["text"]
    elif "img_class" in batch:
        names = batch["img_class"]
    else:
        raise KeyError("Batch must contain class names in 'class_name', 'text', or 'img_class'.")

    if isinstance(names, str):
        return [names]
    if isinstance(names, tuple):
        return list(names)
    if isinstance(names, list):
        return names
    return [str(x) for x in names]


@torch.no_grad()
def evaluate_sam3_clip_guided(
    sam3_model,
    clip_model,
    dataloader_val,
    classes,
    *,
    points_from_gt=False,
    num_points=5,
    use_semantic_injection=True,
    sam3_image_size=1008,
    clip_crop_size=224,
    has_cls_token=True,
    max_batches=0,
    print_fn=print,
):
    """
    Computes your usual prompt-based per-class mIoU:

      1. IoU per image-class sample
      2. average IoUs within each class
      3. average class means
    """
    sam3_model.eval()
    clip_model.eval()

    per_class_ious = defaultdict(list)
    total_loss = 0.0
    total_samples = 0

    total_val_batches = len(dataloader_val)
    if max_batches and max_batches > 0:
        total_val_batches = min(total_val_batches, max_batches)

    val_iter = tqdm(
        enumerate(dataloader_val),
        total=total_val_batches,
        desc="Validation",
        dynamic_ncols=True,
        mininterval=2.0,
        leave=False,
        file=sys.__stderr__
    )

    for bidx, batch in val_iter:
        if max_batches and bidx >= max_batches:
            break

        sam_out, _, _, label, _ = forward_supervised_SAM3_CLIP(
            batch,
            clip_model,
            sam3_model,
            classes,
            clip_crop_size=clip_crop_size,
            points_from_gt=points_from_gt,
            num_points=num_points,
            sam3_image_size=sam3_image_size,
            use_semantic_injection=use_semantic_injection,
            has_cls_token=has_cls_token,
            clear_semantic_after_forward=True,
        )

        loss = bce_dice_iou_loss(sam_out, label)
        ious = compute_iou_per_image(sam_out, label).detach().cpu().tolist()
        class_names = _batch_class_names(batch)

        if len(class_names) != len(ious):
            raise ValueError(
                f"Batch class-name count ({len(class_names)}) != IoU count ({len(ious)}). "
                "Check your dataloader output."
            )

        for cls, iou in zip(class_names, ious):
            per_class_ious[str(cls)].append(float(iou))

        B = label.shape[0]
        total_loss += loss.item() * B
        total_samples += B

        val_iter.set_postfix({
            "loss": f"{loss.item():.4f}",
            "classes": len(per_class_ious),
        })

    class_means = {
        cls: float(np.mean(vals))
        for cls, vals in per_class_ious.items()
        if len(vals) > 0
    }

    miou = float(np.mean(list(class_means.values()))) if class_means else 0.0
    avg_loss = total_loss / max(1, total_samples)

    return miou, avg_loss, class_means


def evaluate_zero_shot_sam3_interactive(
    sam3_model,
    clip_model,
    dataloader_val,
    classes,
    *,
    num_points=5,
    sam3_image_size=1008,
    clip_crop_size=224,
    has_cls_token=True,
    max_batches=0,
):
    """
    Zero-shot CLIP-points + SAM3Interactive baseline.
    This uses CLIP for point sampling but no semantic adapter injection and no training.
    """
    return evaluate_sam3_clip_guided(
        sam3_model,
        clip_model,
        dataloader_val,
        classes,
        points_from_gt=False,
        num_points=num_points,
        use_semantic_injection=False,
        sam3_image_size=sam3_image_size,
        clip_crop_size=clip_crop_size,
        has_cls_token=has_cls_token,
        max_batches=max_batches,
    )


def _split_trainable_params(sam3_model, clip_model):
    adapter_params = []
    interactive_prompt_params = []
    interactive_mask_params = []
    other_sam3_params = []
    clip_vis_params = []
    clip_text_params = []

    for name, p in sam3_model.named_parameters():
        if not p.requires_grad:
            continue
        if "Space_Adapter" in name or "MLP_Adapter" in name:
            adapter_params.append(p)
        elif "inst_interactive_predictor.model.sam_prompt_encoder" in name:
            interactive_prompt_params.append(p)
        elif "inst_interactive_predictor.model.sam_mask_decoder" in name:
            interactive_mask_params.append(p)
        else:
            other_sam3_params.append(p)

    for name, p in clip_model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("visual"):
            clip_vis_params.append(p)
        else:
            clip_text_params.append(p)

    return {
        "adapters": adapter_params,
        "interactive_prompt": interactive_prompt_params,
        "interactive_mask": interactive_mask_params,
        "other_sam3": other_sam3_params,
        "clip_vision": clip_vis_params,
        "clip_text": clip_text_params,
    }


def train_sam3_clip_guided(
    sam3_model,
    clip_model,
    dataloader_train,
    dataloader_val,
    classes,
    *,
    epochs: int = 30,
    lr_adapters: float = 1e-4,
    lr_interactive_prompt: float = 1e-5,
    lr_interactive_mask: float = 1e-5,
    lr_clip_vision: float = 1e-6,
    lr_text_encoder: float = 5e-6,
    lr_other_sam3: float = 1e-5,
    weight_decay: float = 0.0,
    points_from_gt_train: bool = True,
    points_from_gt_eval: bool = False,
    num_points: int = 5,
    use_semantic_injection: bool = True,
    sam3_image_size: int = 1008,
    clip_crop_size: int = 224,
    has_cls_token: bool = True,
    save_dir: str = "./checkpoints_sam3_clip_guided",
    save_name: str = "sam3_adapters",
    log_every: int = 10,
    bce_w: float = 1.0,
    dice_w: float = 1.0,
    iou_w: float = 1.0,
    use_amp: bool = True,
    val_max_batches: int = 0,
    print_fn=print,
    eval_every=1,
):
    os.makedirs(save_dir, exist_ok=True)
    device = next(sam3_model.parameters()).device

    eval_every = max(1, int(eval_every))

    groups = _split_trainable_params(sam3_model, clip_model)

    param_groups = []
    if groups["adapters"]:
        param_groups.append({"params": groups["adapters"], "lr": lr_adapters, "name": "adapters"})
    if groups["interactive_prompt"]:
        param_groups.append({"params": groups["interactive_prompt"], "lr": lr_interactive_prompt, "name": "interactive_prompt"})
    if groups["interactive_mask"]:
        param_groups.append({"params": groups["interactive_mask"], "lr": lr_interactive_mask, "name": "interactive_mask"})
    if groups["other_sam3"]:
        param_groups.append({"params": groups["other_sam3"], "lr": lr_other_sam3, "name": "other_sam3"})
    if groups["clip_vision"]:
        param_groups.append({"params": groups["clip_vision"], "lr": lr_clip_vision, "name": "clip_vision"})
    if groups["clip_text"]:
        param_groups.append({"params": groups["clip_text"], "lr": lr_text_encoder, "name": "clip_text"})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and torch.cuda.is_available())

    print_fn("=" * 70)
    print_fn("Param groups:")
    for g in param_groups:
        n_tensors = len(g["params"])
        n_params = sum(p.numel() for p in g["params"])
        print_fn(f"  {g['name']:20s}: {n_tensors:4d} tensors | {n_params/1e6:8.3f}M params | lr={g['lr']:.2e}")
    print_fn("=" * 70)

    init_miou, init_loss, _ = evaluate_sam3_clip_guided(
        sam3_model,
        clip_model,
        dataloader_val,
        classes,
        points_from_gt=points_from_gt_eval,
        num_points=num_points,
        use_semantic_injection=use_semantic_injection,
        sam3_image_size=sam3_image_size,
        clip_crop_size=clip_crop_size,
        has_cls_token=has_cls_token,
        max_batches=val_max_batches,
    )
    print_fn(f"[Init eval] mIoU={init_miou:.4f} loss={init_loss:.4f}")

    best_iou = init_miou
    best_path = None

    for epoch in range(epochs):
        sam3_model.train()
        clip_model.train()

        total_loss = 0.0
        n_seen = 0
        t0 = time.time()

        train_iter = tqdm(
            enumerate(dataloader_train),
            total=len(dataloader_train),
            desc=f"Train epoch {epoch+1}/{epochs}",
            mininterval=2.0,
            dynamic_ncols=True,
            leave=False,
            file=sys.__stderr__
        )

        for step, batch in train_iter:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available()):
                sam_out, _, _, label, _ = forward_supervised_SAM3_CLIP(
                    batch,
                    clip_model,
                    sam3_model,
                    classes,
                    clip_crop_size=clip_crop_size,
                    points_from_gt=points_from_gt_train,
                    num_points=num_points,
                    sam3_image_size=sam3_image_size,
                    use_semantic_injection=use_semantic_injection,
                    has_cls_token=has_cls_token,
                    # CRITICAL: keep semantic state alive until backward finishes.
                    clear_semantic_after_forward=False,
                )

                loss = bce_dice_iou_loss(
                    sam_out,
                    label,
                    bce_w=bce_w,
                    dice_w=dice_w,
                    iou_w=iou_w,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print_fn(f"[warn] non-finite loss at epoch {epoch+1} step {step+1}; skipping")
                clear_semantic_inputs(sam3_model)
                continue

            try:
                scaler.scale(loss).backward()
            finally:
                clear_semantic_inputs(sam3_model)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            B = label.shape[0]
            total_loss += loss.item() * B
            n_seen += B

            train_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / max(1, n_seen):.4f}",
            })

            if (step + 1) % log_every == 0:
                print_fn(
                    f"  [ep {epoch+1}/{epochs}] step {step+1} "
                    f"avg_loss={total_loss/max(1,n_seen):.4f}"
                )

        scheduler.step()

        do_eval = ((epoch + 1) % eval_every == 0) or ((epoch + 1) == epochs)

        elapsed = time.time() - t0
        train_loss = total_loss / max(1, n_seen)

        val_miou = None
        val_loss = None
        val_class_means = None

        if do_eval:
            val_miou, val_loss, val_class_means = evaluate_sam3_clip_guided(
                sam3_model,
                clip_model,
                dataloader_val,
                classes,
                points_from_gt=points_from_gt_eval,
                num_points=num_points,
                use_semantic_injection=use_semantic_injection,
                sam3_image_size=sam3_image_size,
                clip_crop_size=clip_crop_size,
                has_cls_token=has_cls_token,
                max_batches=val_max_batches,
            )

            print_fn(
                f"[Epoch {epoch+1:03d}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} ({elapsed:.1f}s)"
            )

            if val_miou > best_iou or best_path is None:
                best_iou = val_miou
                best_path = os.path.join(save_dir, f"{save_name}_best.pth")

                # torch.save(
                #     {
                #         "epoch": epoch,
                #         "sam3_trainable_state_dict": _extract_trainable_state_dict(sam3_model),
                #         "clip_state_dict": clip_model.state_dict(),
                #         "best_iou": best_iou,
                #         "val_class_means": val_class_means,
                #     },
                #     best_path,
                # )
                # print_fn(f"  --> new best saved: {best_path}")
                print_fn(f"  --> new best found! mIoU={best_iou:.4f}")

        else:
            print_fn(
                f"[Epoch {epoch+1:03d}] train_loss={train_loss:.4f} "
                f"val skipped eval_every={eval_every} ({elapsed:.1f}s)"
            )

        # Optional last checkpoint saving, if you uncomment later:
        # last_path = os.path.join(save_dir, f"{save_name}_last.pth")
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "sam3_trainable_state_dict": _extract_trainable_state_dict(sam3_model),
        #         "clip_state_dict": clip_model.state_dict(),
        #         "val_miou": val_miou,
        #         "val_loss": val_loss,
        #         "val_class_means": val_class_means,
        #     },
        #     last_path,
        # )

    print_fn(f"DONE. Best val mIoU={best_iou:.4f} ({best_path})")
    return best_iou, best_path


def _extract_trainable_state_dict(sam3_model):
    """
    Save all trainable SAM3 params:
      - adapters
      - interactive prompt encoder
      - interactive mask decoder
      - any optional extra unfrozen SAM3 modules
    """
    return {
        name: p.detach().cpu()
        for name, p in sam3_model.named_parameters()
        if p.requires_grad
    }
