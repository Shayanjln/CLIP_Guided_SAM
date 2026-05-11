# sam3_extension/sam3_clip_guided.py
# -----------------------------------------------------------------------------
# Forward function for SAM3 + CLIP-guided semantic injection.
#
# Key training fix:
#   clear_semantic_after_forward=False during training, then clear after backward.
#   This is required because SAM3 uses activation checkpointing; clearing semantic
#   inputs before backward can make recomputation differ from the original forward.
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from .vit_adapter_inject import set_semantic_inputs, clear_semantic_inputs

def unnormalize_imagenet(x):
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def normalize_sam3(x):
    # Matches Sam3Processor preprocessing before backbone.forward_image.
    return (x - 0.5) / 0.5


def normalize_clip(x):
    # Standard OpenAI/CLIP normalization.
    mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (x - mean) / std


def _l2_norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def forward_supervised_SAM3_CLIP(
    batch,
    clip_model,
    sam3_model,
    classes,
    *,
    clip_crop_size: int = 224,
    points_from_gt: bool = False,
    num_points: int = 5,
    pos_thresh: float = 0.6,
    sam3_image_size: int = 1008,
    use_semantic_injection: bool = True,
    has_cls_token: bool = True,
    norm_dim: int = -1,
    clear_semantic_after_forward: bool = True,
):
    """
    Args:
        batch:
            dict with:
              image: [B,3,H,W]
              text or class_name: list[str]
              label: [B,1,H,W]

        clear_semantic_after_forward:
            Evaluation: True is fine.
            Training: must be False; clear semantic inputs after loss.backward().
    """
    device = next(sam3_model.parameters()).device

    #imgs = batch["image"].to(device, dtype=torch.float32)
    imgs_dataset = batch["image"].to(device, dtype=torch.float32)
    imgs_raw = unnormalize_imagenet(imgs_dataset)
    

    if "text" in batch:
        img_class = batch["text"]
    elif "class_name" in batch:
        img_class = batch["class_name"]
    else:
        raise KeyError("Batch must contain 'text' or 'class_name'.")

    label_masks = batch["label"].to(device, dtype=torch.float32)
    B, _, H_in, W_in = imgs_raw.shape

    # ---- CLIP image/text forward ------------------------------------------
    imgs_clip = F.interpolate(
        imgs_raw,
        size=(clip_crop_size, clip_crop_size),
        mode="bilinear",
        align_corners=False,
        )
    imgs_clip = normalize_clip(imgs_clip)

    clip_image_embeddings = clip_model(imgs_clip, "image")  # [B, 1+N, C] or [B,N,C]
    image_features_clip = clip_image_embeddings / (
        clip_image_embeddings.norm(dim=norm_dim, keepdim=True) + 1e-8
    )

    text_features = _get_text_features_simple(clip_model, img_class)  # [B,C]

    # ---- Similarity map ----------------------------------------------------
    patch_feats = image_features_clip[:, 1:, :] if has_cls_token else image_features_clip
    text_norm = _l2_norm(text_features, dim=-1)
    patch_norm = _l2_norm(patch_feats, dim=-1)

    sim_per_patch = (patch_norm * text_norm.unsqueeze(1)).sum(dim=-1, keepdim=True)  # [B,N,1]

    if has_cls_token:
        cls_zero = torch.zeros(B, 1, 1, device=device, dtype=sim_per_patch.dtype)
        sim_full = torch.cat([cls_zero, sim_per_patch], dim=1)
    else:
        sim_full = sim_per_patch

    # Add scalar similarity to each CLIP token channel via broadcast.
    vis_plus_sim = image_features_clip + sim_full

    side = int(patch_feats.shape[1] ** 0.5)
    if side * side != patch_feats.shape[1]:
        raise ValueError(f"Expected square CLIP patch grid, got {patch_feats.shape[1]} patches.")

    sim_map = sim_per_patch.view(B, side, side, 1).permute(0, 3, 1, 2)
    sim_map = F.interpolate(sim_map.float(), size=(H_in, W_in), mode="bilinear", align_corners=False)
    sim_min = sim_map.amin(dim=(1, 2, 3), keepdim=True)
    sim_max = sim_map.amax(dim=(1, 2, 3), keepdim=True)
    sim_map_norm = (sim_map - sim_min) / (sim_max - sim_min + 1e-8)

    # ---- Sample prompts ----------------------------------------------------
    if points_from_gt:
        # label_masks may be lower resolution than the image, e.g. 256x256.
        # So sampled points are in label-mask coordinates.
        binary_for_sampling = (label_masks > 0.5).float() * 255.0
    else:
        # sim_map_norm was already resized to image resolution, e.g. 1024x1024.
        # So sampled points are in image coordinates.
        binary_for_sampling = (sim_map_norm > pos_thresh).float() * 255.0

    points, point_labels = _sample_random_points(binary_for_sampling, num_points=num_points)
    points = points.to(device)
    point_labels = point_labels.to(device)

    # Scale from the coordinate system actually used for sampling.
    Hs, Ws = binary_for_sampling.shape[-2:]
    scale_x = sam3_image_size / float(Ws)
    scale_y = sam3_image_size / float(Hs)

    points_sam3 = points.clone()
    points_sam3[..., 0] *= scale_x
    points_sam3[..., 1] *= scale_y

    # ---- SAM3 image preprocessing -----------------------------------------
    imgs_sam3 = F.interpolate(
        imgs_raw,
        size=(sam3_image_size, sam3_image_size),
        mode="bilinear",
        align_corners=False,
    )
    imgs_sam3 = normalize_sam3(imgs_sam3)

    if use_semantic_injection:
        set_semantic_inputs(sam3_model, text_features, vis_plus_sim)
    else:
        clear_semantic_inputs(sam3_model)

    target_h, target_w = label_masks.shape[-2:]

    try:
        sam_output = _sam3_forward_with_points(
            sam3_model,
            imgs_sam3,
            points_sam3,
            point_labels,
            target_hw=(target_h, target_w),
        )
    finally:
        if clear_semantic_after_forward:
            clear_semantic_inputs(sam3_model)

    return sam_output, clip_image_embeddings, text_features, label_masks, sim_map_norm


def _get_text_features_simple(clip_model, img_class):
    from CLIP_Surgery.clip import clip as _clip

    device = next(clip_model.parameters()).device

    if isinstance(img_class, str):
        img_class = [img_class]

    tokens = []
    for cls in img_class:
        tokens.append(_clip.tokenize(str(cls)))
    text_tokens = torch.cat(tokens, dim=0).to(device)

    text_features = clip_model(text_tokens, "text")
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
    return text_features


def _sample_random_points(binary_mask: torch.Tensor, num_points: int = 5):
    """
    binary_mask: [B,1,H,W], values {0,255} or {0,1}
    returns:
      points: [B,num_points,2] in original image pixel coords (x,y)
      labels: [B,num_points], all positive
    """
    B, _, H, W = binary_mask.shape
    pts = torch.zeros(B, num_points, 2, dtype=torch.float32)
    lbs = torch.ones(B, num_points, dtype=torch.int32)

    bm = (binary_mask > 0).int().view(B, H, W)
    for i in range(B):
        idx = bm[i].nonzero(as_tuple=False)  # [K,2] as y,x
        if idx.numel() == 0:
            cx, cy = W // 2, H // 2
            pts[i] = torch.tensor([[cx, cy]] * num_points, dtype=torch.float32)
            continue
        sel = idx[torch.randint(0, idx.shape[0], (num_points,))]
        pts[i, :, 0] = sel[:, 1].float()
        pts[i, :, 1] = sel[:, 0].float()
    return pts, lbs


def _sam3_forward_with_points(sam3_model, imgs_sam3, points, point_labels, target_hw):
    """
    Differentiable SAM3 image-backbone + SAM2-style interactive mask decoder path.
    """
    backbone_out = sam3_model.backbone.forward_image(imgs_sam3)

    interactive = getattr(sam3_model, "inst_interactive_predictor", None)
    if interactive is None:
        raise RuntimeError(
            "SAM3 model has no inst_interactive_predictor. Build with "
            "build_sam3_image_model(enable_inst_interactivity=True)."
        )

    if "sam2_backbone_out" not in backbone_out:
        raise RuntimeError(
            "Expected 'sam2_backbone_out'. Build SAM3 with enable_inst_interactivity=True."
        )

    sam2_backbone_out = backbone_out["sam2_backbone_out"]

    # Match Sam3Processor.set_image preprocessing for SAM2 branch.
    if "backbone_fpn" in sam2_backbone_out and len(sam2_backbone_out["backbone_fpn"]) >= 2:
        try:
            sam2_backbone_out["backbone_fpn"][0] = interactive.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
            sam2_backbone_out["backbone_fpn"][1] = interactive.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        except AttributeError:
            pass

    masks_logits = _predict_masks_sam2_style(interactive, sam2_backbone_out, points, point_labels)

    if masks_logits.ndim == 3:
        masks_logits = masks_logits.unsqueeze(1)

    if masks_logits.shape[-2:] != target_hw:
        masks_logits = F.interpolate(
            masks_logits.float(),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    return masks_logits


def _predict_masks_sam2_style(interactive, sam2_backbone_out, points, point_labels):
    """
    Differentiable batched version of SAM3InteractiveImagePredictor._predict.
    """
    inner = interactive.model

    _, vision_feats, vision_pos, _ = inner._prepare_backbone_features(sam2_backbone_out)

    if getattr(inner, "no_mem_embed", None) is not None:
        vision_feats[-1] = vision_feats[-1] + inner.no_mem_embed

    bb_feat_sizes = getattr(interactive, "_bb_feat_sizes", None)
    if bb_feat_sizes is None:
        bb_feat_sizes = [tuple(p.shape[-2:]) for p in vision_pos]

    feats = []
    for feat, fs in zip(vision_feats[::-1], bb_feat_sizes[::-1]):
        B = feat.shape[1]
        H, W = fs
        f = feat.permute(1, 2, 0).contiguous().view(B, -1, H, W)
        feats.append(f)
    feats = feats[::-1]

    image_embed = feats[-1]
    high_res_features = feats[:-1]

    sparse_embeddings, dense_embeddings = inner.sam_prompt_encoder(
        points=(points, point_labels),
        boxes=None,
        masks=None,
    )

    low_res_masks, iou_predictions, _, _ = inner.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=inner.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    return low_res_masks
