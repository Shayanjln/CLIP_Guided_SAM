# sam3_extension/scripts/run_pascalvoc_1_16.py
# -----------------------------------------------------------------------------
# Runner for SAM3 + CLIP-guided interactive/adapted Pascal VOC ablations.
#
# This version uses your real CLIP_Guided_SAM dataset class:
#   PascalVOCBinaryMaskDatasetUnified(samples_json, img_size=1024, out_size=256, ...)
#
# Run from inside sam3_extension:
#
#   PYTHONPATH=..:. python -m scripts.run_pascalvoc_1_16 \
#       --ablation zs_interactive \
#       --train_json ../SAM_CLIP_Script_Training/Pascal_samples_1_16_labeled.json \
#       --val_json ../SAM_CLIP_Script_Training/Pascal_samples_val.json \
#       --batch_size 1 \
#       --num_workers 0 \
#       --val_max_batches 10
#
# Or, if the JSONs are relative to CLIP_Guided_SAM root:
#
#   PYTHONPATH=..:. python -m scripts.run_pascalvoc_1_16 \
#       --ablation zs_interactive \
#       --train_json ../../SAM_CLIP_Script_Training/Pascal_samples_1_16_labeled.json \
#       --val_json ../../SAM_CLIP_Script_Training/Pascal_samples_val.json \
#       --batch_size 1 \
#       --num_workers 0 \
#       --val_max_batches 10
# -----------------------------------------------------------------------------

import os
import sys
import argparse
from pathlib import Path

import random
import numpy as np
import torch

import json
import time
import datetime
import platform
import subprocess

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class Tee:
    """Write stdout/stderr to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_name(s: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return "".join(c if c in allowed else "_" for c in str(s))


def make_run_name(args) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    val = "fullval" if args.val_max_batches == 0 else f"val{args.val_max_batches}"
    if args.run_name:
        return safe_name(args.run_name)
    return safe_name(
        f"{ts}_{args.ablation}_ep{args.epochs}_bs{args.batch_size}_"
        f"pts{args.num_points}_{val}_lrA{args.lr_adapters:g}_lrI{args.lr_interactive_mask:g}"
    )


def setup_run_logging(args):
    run_name = make_run_name(args)
    run_dir = Path(args.save_dir).expanduser().resolve() / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    args_path = run_dir / "args.json"
    cmd_path = run_dir / "command.txt"

    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    with open(cmd_path, "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    log_f = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    print(f"[cwd] Changed working directory to repo root: {_REPO_ROOT}")
    print(f"[run] run_name: {run_name}")
    print(f"[run] run_dir:  {run_dir}")
    print(f"[run] log_path: {log_path}")
    print("=" * 80)
    print("[command]", " ".join(sys.argv))
    print("[python]", sys.version.replace("\n", " "))
    print("[platform]", platform.platform())
    print("[torch]", torch.__version__)
    print("[cuda_available]", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[cuda_device_count]", torch.cuda.device_count())
        print("[cuda_device_0]", torch.cuda.get_device_name(0))
    print("=" * 80)
    print("[args]")
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=" * 80)

    return run_name, run_dir, log_path


def save_result_json(run_dir: Path, result: dict):
    path = run_dir / "result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"[result] saved: {path}")


def notify_user(title: str, message: str, enabled: bool = False):
    if not enabled:
        return

    # Terminal bell. This usually works even over SSH.
    print("\a", end="", flush=True)

    # Linux notification if available.
    try:
        subprocess.run(
            ["notify-send", title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    # macOS notification if running locally.
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

# Make imports robust when running from sam3_extension.
_THIS_FILE = Path(__file__).resolve()
_SCRIPT_DIR = _THIS_FILE.parent
_EXT_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _EXT_DIR.parent

os.chdir(_REPO_ROOT)
print(f"[cwd] Changed working directory to repo root: {_REPO_ROOT}")

for p in [str(_REPO_ROOT), str(_EXT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv",
]


def _resolve_path(p: str) -> str:
    """
    Resolve a path robustly.

    Search order:
      1. absolute path as-is
      2. relative to current working directory
      3. relative to sam3_extension
      4. relative to CLIP_Guided_SAM repo root
    """
    if p is None or str(p).strip() == "":
        return p

    p = os.path.expanduser(str(p))
    if os.path.isabs(p):
        return p

    candidates = [
        Path.cwd() / p,
        _EXT_DIR / p,
        _REPO_ROOT / p,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    # Return cwd-relative for a useful file-not-found message downstream.
    return str((Path.cwd() / p).resolve())


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ablation",
        type=str,
        default=os.environ.get("ABLATION_MODE", "row4"),
        choices=["zs_interactive", "zs_interactive_gt", "ft_interactive", "row2", "row3", "row4", "row5"],
    )

    # Your real JSON workflow.
    p.add_argument(
        "--train_json",
        type=str,
        default="../../SAM_CLIP_Script_Training/Pascal_samples_1_16_labeled.json",
        help="Path to Pascal training prompt-pair JSON.",
    )
    p.add_argument(
        "--val_json",
        type=str,
        default="../../SAM_CLIP_Script_Training/Pascal_samples_val.json",
        help="Path to Pascal validation prompt-pair JSON.",
    )

    # Kept for compatibility with older commands, but not used by this Pascal JSON runner.
    p.add_argument("--data_root", type=str, default="Datasets")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--save_dir",
        type=str,
        default=str(_EXT_DIR / "checkpoints_sam3_rebuttal"),
        help="Directory where run subdirectories will be created to store logs and checkpoints.",
    )
    p.add_argument("--sam3_ckpt", type=str, default=None)
    p.add_argument("--clip_ckpt", type=str, default=None)

    p.add_argument("--lr_adapters", type=float, default=5e-5)
    p.add_argument("--lr_interactive_prompt", type=float, default=5e-5)
    p.add_argument("--lr_interactive_mask", type=float, default=5e-5)
    p.add_argument("--lr_clip", type=float, default=1e-7)

    p.add_argument("--num_points", type=int, default=5)
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--out_size", type=int, default=256)
    p.add_argument("--sam3_image_size", type=int, default=1008)
    p.add_argument("--clip_crop_size", type=int, default=512)
    p.add_argument("--val_max_batches", type=int, default=0, help="0 = full validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default="", help="Optional custom run name.")
    p.add_argument("--notify", action="store_true", help="Ring terminal bell / try system notification when done.")

    p.add_argument(
        "--train_with_clip_points_for_ft_interactive",
        action="store_true",
        help="For ft_interactive, train using CLIP-sampled points instead of GT points.",
    )
    p.add_argument("--eval_every", type=int, default=1, help="Validate every N epochs.")

    return p.parse_args()


def build_pascal_loaders(args):
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    from datasets import PascalVOCBinaryMaskDatasetUnified

    train_json = _resolve_path(args.train_json)
    val_json = _resolve_path(args.val_json)

    if not os.path.exists(train_json):
        raise FileNotFoundError(f"train_json not found: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"val_json not found: {val_json}")

    print(f"[data] train_json: {train_json}")
    print(f"[data] val_json:   {val_json}")

    train_ds = PascalVOCBinaryMaskDatasetUnified(
        train_json,
        img_size=args.img_size,
        out_size=args.out_size,
        use_aug=True,
        use_syn=False,      # IMPORTANT: keep text as one string per prompt-pair for now
    )

    val_ds = PascalVOCBinaryMaskDatasetUnified(
        val_json,
        img_size=args.img_size,
        out_size=args.out_size,
        use_aug=False,
        use_syn=False,      # IMPORTANT: keep text as one string per prompt-pair for now
    )

    def custom_collate(batch):
        # Keep text as list[str]. Default collate would often handle strings okay,
        # but this is explicit and matches your original training code style.
        collated = default_collate([
            {k: v for k, v in sample.items() if k != "text"}
            for sample in batch
        ])
        collated["text"] = [sample["text"] for sample in batch]
        return collated

    # Avoid passing prefetch_factor when num_workers=0.
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"[data] train samples: {len(train_ds)}")
    print(f"[data] val samples:   {len(val_ds)}")

    # Sanity check one batch.
    batch = next(iter(train_loader))
    print("[data] batch keys:", list(batch.keys()))
    print("[data] image:", tuple(batch["image"].shape), batch["image"].dtype)
    print("[data] label:", tuple(batch["label"].shape), batch["label"].dtype)
    print("[data] text example:", batch["text"][0])

    return train_loader, val_loader, PASCAL_CLASSES

def disable_activation_checkpointing(module):
    """
    Disable common activation-checkpointing flags inside SAM3.

    This is needed for row4/row5 when CLIP is trainable and CLIP-derived
    semantic features are injected into SAM3 adapters. Otherwise SAM3's
    checkpoint recomputation can see different hidden semantic state/metadata.
    """
    disabled = []

    for name, m in module.named_modules():
        for attr in [
            "use_act_checkpoint",
            "use_act_ckpt",
            "act_ckpt",
            "grad_checkpointing",
            "act_ckpt_whole_vision_backbone",
            "act_ckpt_whole_language_backbone",
            "use_act_checkpoint_seg_head",
        ]:
            if hasattr(m, attr):
                old = getattr(m, attr)
                if isinstance(old, bool) and old:
                    setattr(m, attr, False)
                    disabled.append((name, attr))

    print(f"[act_ckpt] disabled {len(disabled)} activation-checkpoint flags")
    for name, attr in disabled[:40]:
        prefix = f"{name}." if name else ""
        print(f"  {prefix}{attr}: True -> False")
    if len(disabled) > 40:
        print(f"  ... {len(disabled) - 40} more")


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name, run_dir, log_path = setup_run_logging(args)
    start_time = time.perf_counter()
    result = {
        "run_name": run_name,
        "ablation": args.ablation,
    }

    from sam3.model_builder import build_sam3_image_model

    from utils import initialize_clip, load_clip_model

    # Import package submodules explicitly.
    # Do NOT import vit_adapter_inject as a bare local module, because it uses
    # relative imports such as `from .adapters_sam3 import ...`.
    from sam3_extension.vit_adapter_inject import inject_adapters_into_sam3_vit
    from sam3_extension.freeze_sam3 import (
        freeze_sam3_for_clip_guided,
        freeze_clip_layers_keep_last_n,
        freeze_clip_vision_attention_last_k,
        count_trainable_params,
    )
    from sam3_extension.train_sam3_clip_guided import (
        train_sam3_clip_guided,
        evaluate_sam3_clip_guided,
        evaluate_zero_shot_sam3_interactive,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/5] Building SAM3 with instance interactivity enabled...")
    build_kwargs = dict(enable_inst_interactivity=True)
    if args.sam3_ckpt:
        build_kwargs["checkpoint_path"] = _resolve_path(args.sam3_ckpt)

    sam3 = build_sam3_image_model(**build_kwargs)
    sam3 = sam3.to(device)

    # Needed for CLIP co-adaptation rows. SAM3 uses activation checkpointing
    # internally, and row4/row5 inject trainable CLIP-derived features into
    # checkpointed SAM3 ViT blocks.
    if args.ablation in ("row4", "row5"):
        disable_activation_checkpointing(sam3)
        
        if getattr(sam3, "inst_interactive_predictor", None) is None:
            raise RuntimeError(
                "SAM3 has no inst_interactive_predictor. "
                "Build with enable_inst_interactivity=True."
            )

    use_adapters = args.ablation in ("row2", "row3", "row4", "row5")
    use_semantic_injection = args.ablation in ("row3", "row4", "row5")
    # use_semantic_injection = False
    train_clip_last_n = 12 if args.ablation in ("row4", "row5") else 0

    if use_adapters:
        print(f"[2/5] Injecting adapters (semantic={use_semantic_injection})...")
        carrier = inject_adapters_into_sam3_vit(
            sam3,
            adapter_block_indices=None,
            use_space_adapter=True,
            use_semantic_adapter=use_semantic_injection,
            T_features=512,
            spatial_target=(args.sam3_image_size // 14, args.sam3_image_size // 14),
            scale=0.1,
            has_cls_token=True,
            space_skip_connect=True,
            semantic_skip_connect=False,
        )
        carrier = carrier.to(device)
        print(f"scale set to 0 for debugging. Remember to set a positive scale for real runs!")
    else:
        print("[2/5] No adapters for this ablation.")

    print("[3/5] Loading CLIPSurgery...")
    clip_model = initialize_clip(device, size=args.clip_crop_size, type="CS-ViT-B/16")
    if args.clip_ckpt:
        clip_model = load_clip_model(
            clip_model,
            _resolve_path(args.clip_ckpt),
            device,
            model_type="CS-ViT-B/16",
            freeze=False,
            frz_layers=0,
            input_shape=(args.clip_crop_size, args.clip_crop_size),
        )

    print("[4/5] Applying freezing schedule...")
    if args.ablation in ("zs_interactive", "zs_interactive_gt"):
        freeze_sam3_for_clip_guided(
            sam3,
            train_adapters=False,
            train_interactive_prompt_encoder=False,
            train_interactive_mask_decoder=False,
        )
        freeze_clip_layers_keep_last_n(clip_model, trainable_layers=0)

    elif args.ablation == "ft_interactive":
        freeze_sam3_for_clip_guided(
            sam3,
            train_adapters=False,
            train_interactive_prompt_encoder=True,
            train_interactive_mask_decoder=True,
        )
        freeze_clip_layers_keep_last_n(clip_model, trainable_layers=0)

    else:
        freeze_sam3_for_clip_guided(
            sam3,
            train_adapters=True,
            train_interactive_prompt_encoder=True,
            train_interactive_mask_decoder=True,
        )
        #freeze_clip_layers_keep_last_n(clip_model, trainable_layers=train_clip_last_n)
        freeze_clip_vision_attention_last_k(clip_model, k=train_clip_last_n)

    sam_total, sam_train = count_trainable_params(sam3)
    clip_total, clip_train = count_trainable_params(clip_model)
    print(f"  SAM3: {sam_train/1e6:.3f}M trainable / {sam_total/1e6:.3f}M total")
    print(f"  CLIP: {clip_train/1e6:.3f}M trainable / {clip_total/1e6:.3f}M total")

    result.update({
    "sam_trainable_params": int(sam_train),
    "sam_total_params": int(sam_total),
    "clip_trainable_params": int(clip_train),
    "clip_total_params": int(clip_total),
    })

    print("[5/5] Building Pascal VOC JSON dataloaders...")
    train_loader, val_loader, classes = build_pascal_loaders(args)

    if args.ablation in ("zs_interactive", "zs_interactive_gt"):
        # zs_interactive:    CLIP-sampled points, frozen SAM3Interactive
        # zs_interactive_gt: GT-sampled positive points, frozen SAM3Interactive
        points_from_gt_eval = args.ablation == "zs_interactive_gt"

        if points_from_gt_eval:
            print("[zero-shot] Evaluating GT points + frozen SAM3Interactive...")
        else:
            print("[zero-shot] Evaluating CLIP points + frozen SAM3Interactive...")

        miou, loss, class_means = evaluate_sam3_clip_guided(
            sam3,
            clip_model,
            val_loader,
            classes,
            points_from_gt=points_from_gt_eval,
            num_points=args.num_points,
            use_semantic_injection=False,
            sam3_image_size=args.sam3_image_size,
            clip_crop_size=args.clip_crop_size,
            has_cls_token=True,
            max_batches=args.val_max_batches,
        )

        print("=" * 70)
        label = "GT points" if points_from_gt_eval else "CLIP points"
        print(f"ZERO-SHOT {label} + SAM3Interactive: mIoU={miou:.4f} loss={loss:.4f}")
        print("Per-class mIoU:")
        for c in sorted(class_means.keys()):
            print(f"  {c:15s}: {class_means[c]:.4f}")
        print("=" * 70)

        elapsed = time.perf_counter() - start_time

        result.update({
            "mode": "zero_shot",
            "points": "gt" if points_from_gt_eval else "clip",
            "miou": float(miou),
            "loss": float(loss),
            "class_means": {str(k): float(v) for k, v in class_means.items()},
            "elapsed_seconds": float(elapsed),
            "elapsed_hhmmss": format_seconds(elapsed),
        })

        save_result_json(run_dir, result)

        print("=" * 80)
        print(f"[done] run_name={run_name}")
        print(f"[done] elapsed={format_seconds(elapsed)} ({elapsed:.1f}s)")
        print(f"[done] log={log_path}")
        print("=" * 80)

        notify_user(
            "SAM3 run finished",
            f"{args.ablation} done in {format_seconds(elapsed)}",
            enabled=args.notify,
        )

        return
    

    # Default semi-automatic training/evaluation:
    #   train with CLIP-sampled points
    #   eval with CLIP-sampled points
    #
    # Special ablation:
    #   row5 trains with GT points but still evaluates with CLIP points.
    #   This tests whether clean/manual training prompts transfer better than
    #   train-test aligned CLIP prompts.
    if args.ablation == "row5":
        points_from_gt_train = True
    else:
        points_from_gt_train = False

    points_from_gt_eval = False

    print(f"[points] train source: {'GT' if points_from_gt_train else 'CLIP'}")
    print(f"[points] eval source:  {'GT' if points_from_gt_eval else 'CLIP'}")

    save_name = f"{run_name}_ckpt"
    best_iou, best_path = train_sam3_clip_guided(
        sam3,
        clip_model,
        train_loader,
        val_loader,
        classes,
        epochs=args.epochs,
        lr_adapters=args.lr_adapters,
        lr_interactive_prompt=args.lr_interactive_prompt,
        lr_interactive_mask=args.lr_interactive_mask,
        lr_clip_vision=args.lr_clip,
        lr_text_encoder=5e-6,
        lr_other_sam3=1e-5,
        points_from_gt_train=points_from_gt_train,
        points_from_gt_eval=points_from_gt_eval,
        num_points=args.num_points,
        use_semantic_injection=use_semantic_injection,
        sam3_image_size=args.sam3_image_size,
        clip_crop_size=args.clip_crop_size,
        has_cls_token=True,
        save_dir=str(run_dir),
        save_name=save_name,
        bce_w=1.0,
        dice_w=1.0,
        iou_w=1.0,
        val_max_batches=args.val_max_batches,
        eval_every=args.eval_every,
    )

    print("=" * 70)
    print(f"FINISHED ablation={args.ablation} best mIoU={best_iou:.4f}")
    print(f"Checkpoint: {best_path}")
    print("=" * 70)

    elapsed = time.perf_counter() - start_time

    result.update({
        "mode": "training",
        "best_miou": float(best_iou),
        "best_path": str(best_path),
        "points_from_gt_train": bool(points_from_gt_train),
        "points_from_gt_eval": bool(points_from_gt_eval),
        "elapsed_seconds": float(elapsed),
        "elapsed_hhmmss": format_seconds(elapsed),
    })

    save_result_json(run_dir, result)

    print("=" * 80)
    print(f"[done] run_name={run_name}")
    print(f"[done] elapsed={format_seconds(elapsed)} ({elapsed:.1f}s)")
    print(f"[done] log={log_path}")
    print("=" * 80)

    notify_user(
        "SAM3 run finished",
        f"{args.ablation} done in {format_seconds(elapsed)}",
        enabled=args.notify,
    )


if __name__ == "__main__":
    main()
