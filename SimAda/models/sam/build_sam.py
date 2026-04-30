# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    IE_Vanilla, #as ImageEncoderViT,
    IE_Lora,
    IE_Mix,
    IE_Parallel,
    IE_Parallel_Text,
    IE_Parallel_Text_SideCLIP,
    IE_Parallel_Text_Vis,
    IE_Parallel_Text_Cross,
    IE_Parallel_Text_Vis_Cross,
    IE_NoAdptr_Text_Vis,
    IE_Parallel_Sim,
    IE_Parallel_Conv,
    IE_Parallel_Conv_Text,
    IE_Series,
    IE_Convside,
    IE_Convside_Scaled,
    IE_Convside_All_Scaled,
    #ImageEncoderViT,
    MD_Vanilla,
    MD_Text,
    MD_Mod,
    PE_Vanilla,
    PE_Text,
    Sam,
    TwoWayTransformer,
)

def choose_image_encoder_vit(IE_type: str):
    default_selection = False
    if IE_type == 'Lora':
        IE = IE_Lora
    elif IE_type == 'Mix':
        IE = IE_Mix
    elif IE_type == 'Parallel':
        IE = IE_Parallel
    elif IE_type == 'Parallel_Text':
        IE = IE_Parallel_Text
    elif IE_type == 'Parallel_Text_SideCLIP':
        IE = IE_Parallel_Text_SideCLIP
    elif IE_type == 'Parallel_Text_Vis':
        IE = IE_Parallel_Text_Vis
    elif IE_type == 'Parallel_Text_Cross':
        IE = IE_Parallel_Text_Cross
    elif IE_type == 'Parallel_Text_Vis_Cross':
        IE = IE_Parallel_Text_Vis_Cross
    elif IE_type == 'NoAdapter_Text_Vis':
        IE = IE_NoAdptr_Text_Vis
    elif IE_type == 'Parallel_Sim':
        IE = IE_Parallel_Sim
    elif IE_type == 'Parallel_Conv':
        IE = IE_Parallel_Conv
    elif IE_type == 'Parallel_Conv_Text':
        IE = IE_Parallel_Conv_Text
    elif IE_type == 'Series':
        IE = IE_Series
    elif IE_type == 'Convside':
        IE = IE_Convside
    elif IE_type == 'Convside_Scaled':
        IE = IE_Convside_Scaled
    elif IE_type == 'Convside_All_Scaled':
        IE = IE_Convside_All_Scaled
    elif IE_type == 'MD_Text':
        #IE = IE_Parallel_Conv
        IE = IE_Parallel
    else:
        IE = IE_Vanilla
        default_selection = 'Vanilla'

    if default_selection:
        print(f'Default Image Encoder: {default_selection}')
    else:
        print(f'Image Encoder type: {IE_type}')
    return IE

def choose_mask_decoder(MD_type: str):
    if MD_type == 'MD_Text':
        MD = MD_Text
        print('Mask Decoder with text input')
    elif MD_type == 'MD_Mod':
        MD = MD_Mod
        print('Mask Decoder modified')
    else:
        MD = MD_Vanilla
        print('Mask Decoder without text input')
    return MD

def choose_prompt_encoder(PE_type: str):
    if PE_type == 'PE_Text':
        PE = PE_Text
        print('Prompt Encoder with Text Prompt')
    else:
        PE = PE_Vanilla
        print('Prompt Encoder without Text Prompt')
    return PE
    

def build_sam_vit_h(args = None, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


#build_sam = build_sam_vit_h


def build_sam_vit_l(args, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(args, checkpoint=None, clip_vit=None):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        clip_vit=clip_vit,  # NEW
    )


sam_model_registry = {
    "default": build_sam_vit_b,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    clip_vit=None,   # NEW
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    ImageEncoderViT = choose_image_encoder_vit(args.IE_type)
    MaskDecoder = choose_mask_decoder(args.MD_type)
    PromptEncoder = choose_prompt_encoder(args.PE_type)

        # Build kwargs for IE; add clip_vit only for the IE that needs it
    ie_kwargs = dict(
        args=args,
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    # Whitelist IEs that expect a CLIP sidecar
    IE_NEEDS_CLIP = {"Parallel_Text_SideCLIP"}
    if getattr(args, "IE_type", None) in IE_NEEDS_CLIP:
        ie_kwargs["clip_vit"] = clip_vit  # pass it only here

    sam = Sam(
        args,
        image_encoder=ImageEncoderViT(**ie_kwargs),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

        
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict = False)
    return sam

