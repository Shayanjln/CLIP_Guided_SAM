# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        args,
        image_encoder,#: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.args = args
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    # ------------------------------------------------------------------
    # Helper ------------------------------------------------------------
    # ------------------------------------------------------------------
    def _parse_inputs(self, inputs: Tuple[Any, ...]):
        """
        Returns a dict with keys:
            imgs, points, point_labels, masks,
            text_embeddings, vision_embeddings
        Any missing field is set to None.
        """
        result = dict.fromkeys(
            ["imgs", "points", "point_labels", "masks", "text_embeddings", "vision_embeddings"],
            None,
        )

        L = len(inputs)
        if L == 6:   # imgs, points, point_labels, masks, text, vision
            keys = ["imgs", "points", "point_labels", "masks", "text_embeddings", "vision_embeddings"]
        elif L == 5: # imgs, points, point_labels, masks, text   (or imgs, points, point_labels, text, vision)
            # Decide which slot is which by tensor shape
            if inputs[3].ndim == 4:   # masks are BCHW
                keys = ["imgs", "points", "point_labels", "masks", "text_embeddings"]  
            else:                     # vision embeddings are BxC
                keys = ["imgs", "points", "point_labels", "text_embeddings", "vision_embeddings"]
        elif L == 4: # imgs, points, point_labels, X   (X = masks **or** text)
            if inputs[1].shape[-1] == 2:         # points really are (x,y)
                keys = ["imgs", "points", "point_labels", "text_embeddings"]
            else:                                # points actually holds masks
                keys = ["imgs", "masks", "point_labels", "text_embeddings"]
        elif L == 3: # imgs, points, point_labels   (no text/masks)
            keys = ["imgs", "points", "point_labels"]
        else:
            raise ValueError(f"Unsupported input tuple length: {L}")

        for k, v in zip(keys, inputs):
            result[k] = v

        # post‑process: if ‘points’ really contained masks (detected above)
        if result.get("points") is not None and result["points"].ndim == 4:
            result["masks"]  = result["points"]
            result["points"] = None   # no point prompts in this case

        return result



    @property
    def device(self) -> Any:
        return self.pixel_mean.device


        # ------------------------------------------------------------------
    # Forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, inputs):
        # -------- 1) unpack -------------------------------------------------
        data = self._parse_inputs(inputs)
        imgs              = data["imgs"]
        points            = data["points"]
        point_labels      = data["point_labels"]
        masks             = data["masks"]
        text_embeddings   = data["text_embeddings"]
        vision_embeddings = data["vision_embeddings"]

        has_points = points is not None and point_labels is not None
        has_text   = text_embeddings is not None
        has_masks  = masks is not None

        # -------- 2) prepare point tuple ------------------------------------
        points_input = None
        if has_points:            # allow (B,2) → (B,1,2)
            if points.ndim == 2:
                points_input = (points.unsqueeze(1), point_labels.unsqueeze(1))
            else:
                points_input = (points, point_labels)

        # -------- 3) prompt‑encoder (sparse + dense) ------------------------
        #with torch.no_grad():
        # se, de = self.prompt_encoder(                
        #     points=points_input,
        #     boxes=None,
        #     masks=masks,
        #     text_embeds=text_embeddings if self.args.PE_type == "PE_Text" else None  # ### NEW ###
        # )
        pe_kwargs = dict(points=points_input,
                 boxes=None,
                 masks=masks)

        # add text only if this run uses the text‑aware variant
        if getattr(self.args, "PE_type", "Vanilla") == "PE_Text":
            pe_kwargs["text_embeds"] = text_embeddings       # ← add key
        
        se, de = self.prompt_encoder(**pe_kwargs)

        # -------- 4) image‑encoder ------------------------------------------
        if has_text and self.args.IE_type in ["Parallel_Text", "Parallel_Conv_Text",
                                              "Parallel_Text_Cross", "Parallel_Sim", 'Parallel_Text_SideCLIP']:
            encoded_img = self.image_encoder(imgs, text_embeddings)
        elif has_text and self.args.IE_type in ["Parallel_Text_Vis", "Parallel_Text_Vis_Cross", 'NoAdapter_Text_Vis']:
            encoded_img = self.image_encoder(imgs, text_embeddings, vision_embeddings)
        else:  # no text for image‑encoder branch
            encoded_img = self.image_encoder(imgs)
        # -------- 5) mask‑decoder -------------------------------------------
        md_kwargs = dict(
            image_embeddings      = encoded_img,
            image_pe              = self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = se,
            dense_prompt_embeddings  = de,
            multimask_output      = False,
        )

        if has_text and self.args.MD_type in ["MD_Text", "MD_Mod"]:
            md_kwargs["text_embedding"] = text_embeddings   # ### MOD ###
        pred, _ = self.mask_decoder(**md_kwargs)

        return pred

    @torch.no_grad()
    def forward_orig(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
