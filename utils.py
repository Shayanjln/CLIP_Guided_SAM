import torch
import torch.nn as nn
import torch.optim as optim
from CLIP_Surgery import clip
#from SimAda.models.sam import SamPredictor
#from common_utils import get_network, Arguments
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
import math





def initialize_sam(device,sam_path=False, sam_ckpt = False,IE_type='Vanilla', MD_type='Vanilla', PE_type='Vanilla' ,freeze=False, clip_model=None, vit_type = 'vit_b'):
    # Load SAM model
    if sam_ckpt:
        args = Arguments(IE_type = IE_type, MD_type = MD_type, PE_type = PE_type, vit_type = vit_type, sam_ckpt=sam_ckpt, distributed='0')
    else: 
        args = Arguments(IE_type = IE_type, MD_type = MD_type, PE_type = PE_type, vit_type = vit_type, sam_ckpt="sam_vit_b_01ec64.pth",
                         distributed='0')
    use_gpu  = True
    distribution = 'none'

    sam=get_network(args,use_gpu=use_gpu,gpu_device=device,distribution=distribution, clip_model=clip_model)
 
    if freeze:
        sam = freeze_SAM(sam)    

    return sam

def initialize(sam_path,clip_path,IE_type='Parallel',MD_type='Vanilla', PE_type='Vanilla', device='cuda:0'):
    # initialization 
    
    device_trainable = torch.device(device)
    device_frozen = torch.device(device)

    print(f'Using {IE_type} as IE_type and {MD_type} as MD_type')
    
    # Initialize frozen and trainable SAM models
    #frozen_sam = initialize_sam(device_frozen, sam_type='Parallel',freeze=True)
    trainable_sam = initialize_sam(device_trainable, IE_type=IE_type, MD_type=MD_type, PE_type=PE_type, freeze=True)
    trainable_sam = load_sam(trainable_sam, sam_path)
    
    # Initialize CLIP model and class embeddings
    clip_model, clip_processor = clip.load("CS-ViT-B/16", device=device_trainable)
    clip_model = load_clip_model(clip_model,clip_path, 
                                 device_trainable, input_shape=(224,224))
    #clip_model_frozen, _ = clip.load("CS-ViT-B/16", device=device_frozen)
    #clip_model.load_state_dict(torch.load('fine_tuned_sam_joint_bestwith_similarity_1.pth'))
    
    classes = ['cat','dog']
    return trainable_sam, clip_model, classes, device_trainable

def load_sam(sam,file_path):
    sam.load_state_dict(torch.load(file_path))
    return sam


def initialize_clip(device, size=512, type='CS-ViT-B/16'):
    clip_model, _ = clip.load(type, device=device)
    with torch.no_grad():
        input_shape=(size,size)
        x,y = input_shape
        dummy_input = torch.randn(1, 3, x, y).to(device)
        clip_model(dummy_input,mode='image')
    return clip_model


def load_clip_model(clip_model,path,device,model_type="CS-ViT-B/16", freeze=False, frz_layers=2, input_shape=(224,224)):
    if model_type == "CS-ViT-B/16":
        
        print("Clip model: ",model_type," Clip path provided: ",path)
        x,y = input_shape
        dummy_input = torch.randn(1, 3, x, y).to(device)
        clip_model(dummy_input,mode='image')
        clip_model.load_state_dict(torch.load(path))
        print("load successful")
        if freeze:
            clip_model = freeze_clip_layers(clip_model, trainable_layers=frz_layers)
            print(f"Froze last {frz_layers} layers")
            
    return clip_model


def get_network(args, use_gpu=True, gpu_device = 0, distribution = 'none', clip_model=None):
    """ return given network
    """


    from SimAda.models.sam import SamPredictor, sam_model_registry

    builder = sam_model_registry[args.vit_type]
    # Only pass clip_vit if provided (keeps backward compatibility)
    if clip_model is None:
        net = builder(args, checkpoint=args.sam_ckpt)
    else:
        net = builder(args, checkpoint=args.sam_ckpt, clip_vit=clip_model)

    if use_gpu:
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


# net types: 'Vanilla', 'Lora', 'Mix', 'Parallel', 'Series'

class Arguments():
  def __init__(self, sam_ckpt="sam_vit_b_01ec64.pth", IE_type: str = 'Vanilla', MD_type: str = 'Vanilla', PE_type: str = 'Vanilla', vit_type = 'vit_b', thd = False, distributed='0') -> None:
    self.sam_ckpt = sam_ckpt
    self.IE_type = IE_type
    self.MD_type = MD_type
    self.PE_type = PE_type
    self.vit_type = vit_type
    self.thd = False
    self.distributed = distributed


def read_batch(path):
    with open(path, 'rb') as f:
        b = pickle.loads(f.read())
    return b


def freeze_clip_layers(clip_model, trainable_layers: int):
    """
    Freezes all layers except the last `trainable_layers` in the CLIP model.

    Args:
        clip_model: The CLIP model with a specific architecture (CLIPSurgery).
        trainable_layers: Number of trainable layers from the end.

    """
    # Freeze all parameters first
    for param in clip_model.parameters():
        param.requires_grad = False

    # Make the last 'trainable_layers' in both vision and text transformer trainable
    total_visual_layers = len(clip_model.visual.transformer.resblocks)
    for i in range(total_visual_layers - trainable_layers, total_visual_layers):
        for param in clip_model.visual.transformer.resblocks[i].parameters():
            param.requires_grad = True

    total_text_layers = len(clip_model.transformer.resblocks)
    for i in range(total_text_layers - trainable_layers, total_text_layers):
        for param in clip_model.transformer.resblocks[i].parameters():
            param.requires_grad = True

    return clip_model

def freeze_clip_full(clip_model):
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model


def freeze_SAM(sam):
    list_names = ['Space_Adapter','MLP_Adapter','Depth_Adapter','linear_a_q','linear_b_q','linear_a_v','linear_b_v']

    for name,param in sam.image_encoder.named_parameters():
        param.requires_grad = False
        
        if len(name.split('.'))>3:
            if name.split('.')[2] in list_names:
                param.requires_grad = True
        if len(name.split('.'))>=5:
            if name.split('.')[4] in list_names:
                param.requires_grad = True
        
        if name.split('.')[0] == 'convblocks':
            param.requires_grad = True

        if name.split('.')[0] == 'clip_sidecar':
            if 'proj' in name.split('.')[1]:
                param.requires_grad = True

        if name.split('.')[0] == 'shared_text_proj':
            param.requires_grad = True

        if len(name.split('.'))>=3:
            if name.split('.')[2] in ['clip_fuse_down_x','clip_fuse_down_clip','clip_fuse_up','ln_fuse']:
                param.requires_grad = True
        
    
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    return sam

#### Parameter count utilities

import torch
from typing import Optional

def _unwrap_ddp(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m

def count_params(module: torch.nn.Module):
    module = _unwrap_ddp(module)
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def _find_block_list_sam_image_encoder(image_encoder: torch.nn.Module):
    """
    Tries common attribute paths for SAM-style ViT blocks.
    Returns a ModuleList/Sequential-like container or None.
    """
    ie = _unwrap_ddp(image_encoder)

    # Most common in SAM:
    if hasattr(ie, "blocks"):
        return ie.blocks

    # Fallbacks (in case you wrapped/renamed things)
    for path in [
        ("transformer", "blocks"),
        ("vit", "blocks"),
        ("trunk", "blocks"),
        ("encoder", "blocks"),
    ]:
        cur = ie
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return cur

    return None

def count_trainable_blocks_sam_image_encoder(sam: torch.nn.Module) -> Optional[int]:
    """
    Returns number of trainable ViT blocks in SAM image encoder (0..N), or None if not found.
    """
    sam = _unwrap_ddp(sam)
    if not hasattr(sam, "image_encoder"):
        return None

    blocks = _find_block_list_sam_image_encoder(sam.image_encoder)
    if blocks is None:
        return None

    # A "trainable block" = block has at least one param with requires_grad=True
    n_trainable = 0
    for blk in blocks:
        if any(p.requires_grad for p in blk.parameters()):
            n_trainable += 1
    return n_trainable

def count_trainable_blocks_clip_vision(clip: torch.nn.Module) -> Optional[int]:
    """
    Returns number of trainable CLIP ViT resblocks (0..N), or None if not found.
    """
    clip = _unwrap_ddp(clip)
    if not (hasattr(clip, "visual") and hasattr(clip.visual, "transformer") and hasattr(clip.visual.transformer, "resblocks")):
        return None

    resblocks = clip.visual.transformer.resblocks
    n_trainable = 0
    for blk in resblocks:
        if any(p.requires_grad for p in blk.parameters()):
            n_trainable += 1
    return n_trainable

def format_param_pct(trainable: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(trainable) / float(total)

#### end of parameter count utilities

def get_binary_output(pred: torch.Tensor, out_tensor=False):
    """
    Converts model logits into binary predictions.
    Args:
        pred: Model output logits.
        out_tensor: If True, returns a tensor; otherwise, returns a NumPy array.
    Returns:
        Binary output tensor or NumPy array.
    """
    # pred_logits = pred.squeeze()
    pred_probs = torch.sigmoid(pred)
    bin_output = (pred_probs > 0.5).float()
    return bin_output if out_tensor else bin_output.detach().cpu().numpy()



def clip_contrastive_loss(image_features, text_features, temperature=0.05):
    # image_features and text_features shape: (B,C)
    
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity matrix (batch_size x batch_size)
    logits = (image_features @ text_features.T) / temperature  # Cosine similarity divided by temperature

    # Create labels (identity matrix)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size).to(image_features.device)

    # Compute cross-entropy loss
    loss_img = F.cross_entropy(logits, labels)
    loss_txt = F.cross_entropy(logits.T, labels)

    return (loss_img + loss_txt) / 2  # Average loss for image and text

def get_sim(clip_model, device, batch, classes, thresh=0.8, upblock=None):
    clip_model.eval()
    
    with torch.no_grad():
    
        imgs = batch['image'].to(dtype = torch.float32, device = device)
        img_class = batch['text']
        label_masks = batch['label'].to(dtype = torch.float32, device = device)
    
        imgs_clip = torch.nn.functional.interpolate(imgs, size=(224, 224), 
                                                        mode="bilinear", align_corners=False)
        clip_image_embeddings = clip_model(imgs_clip,mode='image')
        image_features_clip = clip_image_embeddings / clip_image_embeddings.norm(dim=1, keepdim=True)
        text_features = get_text_features(clip_model,device,classes,img_class,
                      simple_prompt=True,prompt_ensemble=False,custom_prompt=False)
        #features = image_features_clip @ batch_text_features.t()
        features = get_sim_features(image_features_clip,text_features, device)
        # similarity_map = clip.get_similarity_map(features[:, 1:, :], imgs.shape[2:])
        # similarity_prompt = similarity_map.permute(0,3,1,2)
        # similarity_prompt =  F.interpolate(similarity_prompt, size=(256, 256), 
        #                                            mode='bilinear', align_corners=False)

        if upblock != None:
            
            similarity_map = get_similarity_map(features[:, 1:, :], (256,256), upblock)
            similarity_prompt = similarity_map.permute(0,3,1,2)
        else:
            similarity_map = clip.get_similarity_map(features[:, 1:, :], imgs.shape[2:])
            similarity_prompt = similarity_map.permute(0,3,1,2)
            similarity_prompt =  F.interpolate(similarity_prompt, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                
        prompt_mask = (similarity_prompt > thresh)*255.0
        
        
        return prompt_mask


def sample_random_points_from_similarity_0(mask, num_points=5, label=1, both=False):
    # mask shape: (B, 1, W, H)
    sim_mask = (mask//255).cpu().numpy()
    points = []
    point_labels = []
    for i in range(sim_mask.shape[0]):
        indices = np.argwhere(sim_mask[i,0] == label)
        p = indices[np.random.randint(len(indices),size=num_points)].copy()
        points.append(p[:,::-1])
        point_labels.append([1 for i in range(num_points)])
    points = np.array(points)
    
    return points, point_labels


def sample_random_points_from_similarity(mask, num_points=5, label=1, both=False):
    """
    Args:
        mask (Tensor): shape (B, 1, H, W), values in {0, 255}
        num_points (int): number of positive (and optionally negative) points to sample per image
        label (int): label to treat as "positive" (default 1)
        both (bool): if True, also sample `num_points` negative (label=0) points

    Returns:
        points (Tensor): shape (B, N, 2), where N = num_points or 2*num_points
        point_labels (Tensor): shape (B, N), int64
    """
    sim_mask = (mask // 255)  # Shape: (B, 1, H, W)
    B, _, H, W = sim_mask.shape
    device = sim_mask.device

    all_points = []
    all_labels = []

    for i in range(B):
        pos_mask = (sim_mask[i, 0] == label)
        pos_indices = pos_mask.nonzero(as_tuple=False)
        if pos_indices.numel() == 0:
            raise ValueError(f"No positive points found in sample {i}.")

        pos_sample_idx = torch.randint(0, pos_indices.shape[0], (num_points,), device=device)
        pos_points = pos_indices[pos_sample_idx][:, [1, 0]]  # (x, y)

        points_list = [pos_points]
        labels_list = [torch.ones(num_points, dtype=torch.int64, device=device)]

        if both:
            neg_label = 0 if label == 1 else 1
            neg_mask = (sim_mask[i, 0] == neg_label)
            neg_indices = neg_mask.nonzero(as_tuple=False)
            if neg_indices.numel() == 0:
                raise ValueError(f"No negative points found in sample {i}.")

            neg_sample_idx = torch.randint(0, neg_indices.shape[0], (num_points,), device=device)
            neg_points = neg_indices[neg_sample_idx][:, [1, 0]]  # (x, y)

            points_list.append(neg_points)
            labels_list.append(torch.zeros(num_points, dtype=torch.int64, device=device))

        img_points = torch.cat(points_list, dim=0).unsqueeze(0)  # (1, N, 2)
        img_labels = torch.cat(labels_list, dim=0)               # (N,)

        all_points.append(img_points)
        all_labels.append(img_labels)

    points = torch.cat(all_points, dim=0)          # (B, N, 2)
    point_labels = torch.stack(all_labels, dim=0)  # (B, N)

    return points, point_labels

def sample_random_points_vectorized(mask, num_points=5, label=1, both=False):
    """
    Vectorized version of point sampling.
    Assumes each mask has enough positive and (if both=True) negative points.
    """
    sim_mask = mask // 255  # (B, 1, H, W)
    B, _, H, W = sim_mask.shape
    device = sim_mask.device

    sim_mask = sim_mask.squeeze(1)  # (B, H, W)
    flat_mask = sim_mask.reshape(B, -1)  # (B, H*W)

    # For positive label
    pos_indices = (flat_mask == label)
    pos_idx_list = []

    for b in range(B):
        valid_indices = pos_indices[b].nonzero(as_tuple=False).view(-1)  # <-- safer than .squeeze()
        chosen = valid_indices[torch.randint(0, valid_indices.shape[0], (num_points,), device=device)]
        pos_idx_list.append(chosen)

    pos_idx_tensor = torch.stack(pos_idx_list, dim=0)  # (B, num_points)
    pos_y = pos_idx_tensor // W
    pos_x = pos_idx_tensor % W
    pos_points = torch.stack([pos_x, pos_y], dim=-1)  # (B, num_points, 2)
    pos_labels = torch.ones(B, num_points, dtype=torch.int64, device=device)

    if not both:
        return pos_points, pos_labels

    # For negative label (0)
    neg_indices = (flat_mask == 0)
    neg_idx_list = []

    for b in range(B):
        valid_indices = neg_indices[b].nonzero(as_tuple=False).view(-1)  # <-- again, use .view(-1)
        chosen = valid_indices[torch.randint(0, valid_indices.shape[0], (num_points,), device=device)]
        neg_idx_list.append(chosen)

    neg_idx_tensor = torch.stack(neg_idx_list, dim=0)  # (B, num_points)
    neg_y = neg_idx_tensor // W
    neg_x = neg_idx_tensor % W
    neg_points = torch.stack([neg_x, neg_y], dim=-1)  # (B, num_points, 2)
    neg_labels = torch.zeros(B, num_points, dtype=torch.int64, device=device)

    # Combine pos and neg
    all_points = torch.cat([pos_points, neg_points], dim=1)       # (B, 2*num_points, 2)
    all_labels = torch.cat([pos_labels, neg_labels], dim=1)       # (B, 2*num_points)

    return all_points, all_labels


def sample_random_points_with_probabilities(mask, num_points=10, label=1, both=False, threshold=0.8, sample_from_probabilities=False):
    """
    Samples random points from a given mask, with optional sampling from highest and lowest probabilities (logits or probabilities).
    If `sample_from_probabilities=True`, it samples based on the point's probability, considering the given threshold.
    
    Arguments:
        mask (torch.Tensor): Mask containing either binary values or probabilities/logits of shape (B, 1, H, W).
        num_points (int): Number of points to sample.
        label (int): The label to sample (typically 1 for positive or 0 for negative).
        both (bool): If True, also samples negative points (label 0).
        threshold (float): Minimum threshold for a point to be considered in the sampling pool (used if `sample_from_probabilities=True`).
        sample_from_probabilities (bool): Whether to sample from probabilities/logits instead of binary values.
        
    Returns:
        (torch.Tensor, torch.Tensor): Sampled points and their corresponding labels.
    """
    B, _, H, W = mask.shape
    device = mask.device

    mask = mask.squeeze(1)  # (B, H, W)
    flat_mask = mask.reshape(B, -1)  # (B, H*W)

    if sample_from_probabilities:
        # Use thresholding to filter points for positive and negative sampling
        pos_indices = flat_mask >= threshold  # (B, H*W), points with probability >= threshold
        neg_indices = flat_mask < threshold  # (B, H*W), points with probability < threshold
    else:
        pos_indices = (flat_mask == label)  # (B, H*W), binary mask for positive points
        neg_indices = (flat_mask == 0)  # (B, H*W), binary mask for negative points

    pos_idx_list = []
    neg_idx_list = []

    for b in range(B):
        # For positive points: sample based on probabilities/logits or binary mask
        if sample_from_probabilities:
            valid_indices = pos_indices[b].nonzero(as_tuple=False).view(-1)
            probs = flat_mask[b][valid_indices]  # probabilities for selected points
            chosen = valid_indices[torch.multinomial(probs, num_points, replacement=False)]
        else:
            valid_indices = pos_indices[b].nonzero(as_tuple=False).view(-1)
            chosen = valid_indices[torch.randint(0, valid_indices.shape[0], (num_points,), device=device)]
        pos_idx_list.append(chosen)

        # For negative points
        if sample_from_probabilities:
            valid_indices = neg_indices[b].nonzero(as_tuple=False).view(-1)
            probs = flat_mask[b][valid_indices]
            inv_probs = 1.0 - probs  # Invert: low prob -> high weight
            chosen = valid_indices[torch.multinomial(inv_probs, num_points, replacement=False)]
        else:
            valid_indices = neg_indices[b].nonzero(as_tuple=False).view(-1)
            chosen = valid_indices[torch.randint(0, valid_indices.shape[0], (num_points,), device=device)]

        neg_idx_list.append(chosen)

    pos_idx_tensor = torch.stack(pos_idx_list, dim=0)  # (B, num_points)
    neg_idx_tensor = torch.stack(neg_idx_list, dim=0)  # (B, num_points)

    pos_y = pos_idx_tensor // W
    pos_x = pos_idx_tensor % W
    pos_points = torch.stack([pos_x, pos_y], dim=-1)  # (B, num_points, 2)
    pos_labels = torch.ones(B, num_points, dtype=torch.int64, device=device)

    neg_y = neg_idx_tensor // W
    neg_x = neg_idx_tensor % W
    neg_points = torch.stack([neg_x, neg_y], dim=-1)  # (B, num_points, 2)
    neg_labels = torch.zeros(B, num_points, dtype=torch.int64, device=device)

    if not both:
        return pos_points, pos_labels

    # Combine pos and neg
    all_points = torch.cat([pos_points, neg_points], dim=1)       # (B, 2*num_points, 2)
    all_labels = torch.cat([pos_labels, neg_labels], dim=1)       # (B, 2*num_points)

    return all_points, all_labels

import math
import torch

import torch

import torch

@torch.no_grad()
def sample_uniform_points(mask, num_points=5, mode="random",
                          grid_shape=None, replace=False,
                          force_label=None):
    """
    Uniformly sample points from the whole image (not class-restricted).

    Args:
        mask: (B,1,H,W) or (B,H,W) tensor with values {0,1} or {0,255}.
        num_points: used ONLY in mode="random".
        mode: "random" or "grid".
        grid_shape: (ny, nx) grid resolution for grid mode (num_points ignored).
        replace: for random mode — sample with replacement if True.
        force_label: 
            - None → use true mask labels (default)
            - 0 → set all labels to 0
            - 1 → set all labels to 1

    Returns:
        points: (B, K, 2) int64, where (x,y)
        labels: (B, K) int64
    """
    # Normalize mask
    if mask.dim() == 4 and mask.size(1) == 1:
        sim_mask = (mask[:, 0] > 0).long()
    elif mask.dim() == 3:
        sim_mask = (mask > 0).long()
    else:
        raise ValueError("mask must be (B,1,H,W) or (B,H,W)")

    B, H, W = sim_mask.shape
    device = sim_mask.device

    if mode not in ("random", "grid"):
        raise ValueError("mode must be 'random' or 'grid'")

    # --- Point sampling ---
    if mode == "random":
        total = H * W
        if not replace and num_points > total:
            raise ValueError(f"num_points={num_points} exceeds H*W={total} with replace=False")

        if replace:
            flat_idx = torch.randint(0, total, (B, num_points), device=device)
        else:
            flat_idx = torch.stack(
                [torch.randperm(total, device=device)[:num_points] for _ in range(B)],
                dim=0
            )
        y = flat_idx // W
        x = flat_idx % W
        points = torch.stack([x, y], dim=-1).to(torch.int64)

    else:  # "grid" mode
        if grid_shape is None or not (isinstance(grid_shape, (tuple, list)) and len(grid_shape) == 2):
            raise ValueError("For mode='grid', provide grid_shape=(ny, nx).")
        ny, nx = int(grid_shape[0]), int(grid_shape[1])
        if ny < 1 or nx < 1:
            raise ValueError("grid_shape values must be >= 1")

        ys = torch.linspace(0, H - 1, steps=ny, device=device).round().long()
        xs = torch.linspace(0, W - 1, steps=nx, device=device).round().long()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        gx = grid_x.reshape(-1)
        gy = grid_y.reshape(-1)
        base_points = torch.stack([gx, gy], dim=-1).to(torch.int64)
        points = base_points.unsqueeze(0).expand(B, -1, -1).contiguous()

    # --- Labels ---
    if force_label is not None:
        # Force all to 0 or 1
        labels = torch.full((B, points.shape[1]), int(force_label),
                            dtype=torch.int64, device=device)
    else:
        x = points[..., 0].clamp_(0, W - 1)
        y = points[..., 1].clamp_(0, H - 1)
        b_idx = torch.arange(B, device=device).view(B, 1).expand_as(x)
        labels = sim_mask[b_idx, y, x].to(torch.int64)

    return points, labels

    
def get_sim_features_0(img_embeddings, text_embeddings):
    # text_embeddings shape: [B, 512]
    # img_embedding shape: [B, 197, 512]

    device = img_embeddings.device
    batch_size = img_embeddings.shape[0]
    features = []
    for i in range(batch_size):
        img_emb = img_embeddings[i]
        text_emb = text_embeddings[i].unsqueeze(0)
        feat = text_emb @ img_emb.t()
        features.append(feat)
    features = torch.cat(features, dim=0).to(device)
    return features.unsqueeze(-1)


def get_sim_features(img_embeddings, text_embeddings):
    # Get device automatically from img_embeddings
    device = img_embeddings.device

    # Efficiently compute similarities for the whole batch
    features = torch.bmm(
        text_embeddings.unsqueeze(1),          # [B, 1, 512]
        img_embeddings.transpose(1, 2)         # [B, 512, 197]
    ).squeeze(1).unsqueeze(-1)                 # [B, 197, 1]
    return features

def get_sim_features_surgery(img_embeddings, text_embeddings, redundant_features=None):

    features = clip.clip_feature_surgery(img_embeddings, text_embeddings, redundant_feats=redundant_features)
    batch_indices = torch.arange(features.size(0))  # [0, 1, 2, 3]
    selected = features[batch_indices, :, batch_indices].unsqueeze(-1)

    return selected



def get_text_features_0(clip_model,classes,img_class,
                      simple_prompt=False,prompt_ensemble=False,custom_prompt=False, prompt_templates=None):
    device = next(clip_model.parameters()).device

    if simple_prompt:
        tokens = []
        for cls in img_class:
            text_tkn = clip.tokenize(cls)
            tokens.append(text_tkn)
        text_tokens = torch.cat(tokens, dim=0).to(device)
        text_features = clip_model(text_tokens,'text')
        return text_features
    elif prompt_ensemble:
        #### change encode_text_with_prompt_ensemble function to use clip(txt, 'text') instead of encode_text()
        class_embeddings = clip.encode_text_with_prompt_ensemble(clip_model, classes, device)
        list_text = []
        for cls in img_class:
            class_idx = classes.index(cls)
            class_text_feature = class_embeddings[class_idx].unsqueeze(0)
            list_text.append(class_text_feature)
            
        text_features = torch.cat(list_text, dim=0).to(device)
        return text_features
    elif custom_prompt:
        text_embeddings_ensemble = clip.encode_text_with_prompt_ensemble(clip_model, classes, device, 
                                                                     prompt_templates=prompt_templates)
        list_text = []
        for cls in img_class:
            class_idx = classes.index(cls)
            class_text_feature = text_embeddings_ensemble[class_idx].unsqueeze(0)
            list_text.append(class_text_feature)
            
        text_features = torch.cat(list_text, dim=0).to(device)
        return text_features    
    
    
def get_text_features(clip_model, classes, img_class,
                      simple_prompt=False, prompt_ensemble=False, custom_prompt=False, prompt_templates=None):
    device = next(clip_model.parameters()).device  # Automatically detect device

    if simple_prompt:
        # Batch tokenize text in one call
        tokens = clip.tokenize(img_class).to(device)  # Tokenize all at once
        text_features = clip_model(tokens, 'text')
        return text_features
    
    elif prompt_ensemble:
        #class_embeddings = clip.encode_text_with_prompt_ensemble(clip_model, classes, device)
        #list_text = [class_embeddings[classes.index(cls)].unsqueeze(0) for cls in img_class]
        #list_text = [class_embeddings[classes.index(cls)].unsqueeze(0) for cls in img_class]
        #text_features = torch.cat(list_text, dim=0)
        text_features = clip.encode_text_with_prompt_ensemble(clip_model, img_class, device)
        return text_features

    elif custom_prompt:
        text_embeddings_ensemble = clip.encode_text_with_prompt_ensemble(clip_model, classes, device, prompt_templates=prompt_templates)
        list_text = [text_embeddings_ensemble[classes.index(cls)].unsqueeze(0) for cls in img_class]
        text_features = torch.cat(list_text, dim=0)
        return text_features

def get_text_features_v2(clip_model,img_class, simple_prompt=True):
    """
    img_class: List[List[str]], e.g., [['building', 'edifice'], ['cat']]
    """
    device = next(clip_model.parameters()).device
    if simple_prompt:
        all_prompts = [p for synonyms in img_class for p in synonyms]  # flatten
        tokens = clip.tokenize(all_prompts).to(device)                 # [N, 77]
        features = clip_model(tokens, 'text')                          # [N, D]
        features = features / features.norm(dim=-1, keepdim=True)

        # Now unflatten and average features per sample
        per_sample_features = []
        idx = 0
        for synonyms in img_class:
            count = len(synonyms)
            avg_feat = features[idx:idx+count].mean(dim=0)
            per_sample_features.append(avg_feat)
            idx += count

        return torch.stack(per_sample_features)  # [B, D]
    

def get_text_features_v3(
    clip_model,
    classes,                     # kept for API compatibility (unused here)
    img_class,                   # list of items; each item is a str or list[str]
    simple_prompt=False,
    prompt_ensemble=False,
    custom_prompt=False,
    prompt_templates=None,
):
    """
    Returns a [batch_size, dim] tensor. Each batch item can be a single class string
    or a list of synonym strings; when synonyms are provided, their features are averaged.
    """
    device = next(clip_model.parameters()).device

    # Normalize input into list[list[str]] where each inner list are synonyms
    if not isinstance(img_class, (list, tuple)):
        img_class = [img_class]
    per_item_texts = []
    for item in img_class:
        if isinstance(item, (list, tuple)):
            texts = [str(t).strip() for t in item if str(t).strip()]
            if not texts:
                raise ValueError("One of the batch items has an empty synonyms list.")
            per_item_texts.append(texts)
        else:
            s = str(item).strip()
            if not s:
                raise ValueError("One of the batch items is an empty string.")
            per_item_texts.append([s])

    # Flatten all texts so we can encode in a single batched call
    all_texts = [t for texts in per_item_texts for t in texts]

    with torch.no_grad():
        # Encode text(s) according to the chosen mode
        if custom_prompt:
            if prompt_templates is None:
                raise ValueError("custom_prompt=True requires prompt_templates.")
            # Expectation: clip.encode_text_with_prompt_ensemble(model, texts, device, prompt_templates=...)
            feats_flat = clip.encode_text_with_prompt_ensemble(
                clip_model, all_texts, device, prompt_templates=prompt_templates
            )
        elif prompt_ensemble:
            # Expectation: uses built-in templates inside the helper
            feats_flat = clip.encode_text_with_prompt_ensemble(
                clip_model, all_texts, device
            )
        else:
            # Simple prompt: direct tokenize/encode
            tokens = clip.tokenize(all_texts).to(device)
            feats_flat = clip_model(tokens, "text")  # shape: [N_texts, dim]

        # Now aggregate: mean features for synonyms per batch item
        out_features = []
        idx = 0
        for texts in per_item_texts:
            n = len(texts)
            # Average synonyms for this item
            item_feat = feats_flat[idx:idx + n].mean(dim=0, keepdim=True)
            out_features.append(item_feat)
            idx += n

        text_features = torch.cat(out_features, dim=0)  # [batch_size, dim]

    return text_features


def SAM_finetunable(sam):
    list_names = ['Space_Adapter','MLP_Adapter','Depth_Adapter']

    for name,param in sam.image_encoder.named_parameters():
        param.requires_grad = False
        
        if len(name.split('.'))>3:
            if name.split('.')[2] in list_names:
                param.requires_grad = True
    
    
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    return sam

def get_sim_feat_input(batch, clip_model, device, classes):
    imgs = batch['image'].to(dtype = torch.float32, device = device)
    img_class = batch['text']
    #label_masks = batch['label'].to(dtype = torch.float32, device = device)
    
    imgs_clip = torch.nn.functional.interpolate(imgs, size=(224, 224), 
                                                    mode="bilinear", align_corners=False)
    clip_image_embeddings = clip_model(imgs_clip,'image')  
    image_features_clip = clip_image_embeddings / clip_image_embeddings.norm(dim=1, keepdim=True)
    text_features = get_text_features(clip_model,device,classes,img_class,
                  simple_prompt=True,prompt_ensemble=False,custom_prompt=False) 
    #features = image_features_clip @ batch_text_features.t()
    features = get_sim_features(image_features_clip,text_features, device)

    features_input = features.squeeze(-1)[:,1:]
    return features_input





def calculate_IoU_per_mask(mask,gt):
    flt_pred = mask.flatten().cpu().bool()
    flt_target = gt.flatten().bool()
    intersection = (flt_pred & flt_target).sum().item()
    union = (flt_pred | flt_target).sum().item()
    return intersection/(union + 1e-10)


def get_similarity_map(sm, shape, upsample_block):
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
    side = int(sm.shape[1] ** 0.5)  # Expected square structure
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)  # [B, 1, 14, 14]

    # Trainable upsampling
    sm = upsample_block(sm)  

    sm = sm.permute(0, 2, 3, 1)  # Convert back to [B, H, W, C]
    return sm


def update_ema_frozen(frozen_model,student_model,decay):
    with torch.no_grad():
        for ema_param, param in zip(frozen_model.parameters(), student_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=(1 - decay))  # EMA update

    return frozen_model


def get_dynamic_unsup_weight(epoch,total_epochs,a):
    return a*(np.exp(-5*(1-epoch/total_epochs)**2))


def generate_pseudo_labels(logits, threshold=0.7):
    """
    Convert logits to pseudo-labels based on a confidence threshold.
    
    Args:
        logits (torch.Tensor): Model output of shape [B, 1, H, W] (logits).
        threshold (float): Confidence threshold for selecting pseudo-labels.
    
    Returns:
        pseudo_labels (torch.Tensor): Hard pseudo-labels with shape [B, 1, H, W].
        mask (torch.Tensor): Mask of confident pixels (1 = keep, 0 = ignore).
        confidence_ratio (float): Percentage of confident pixels in each image.
    """
    probs = torch.sigmoid(logits)
    mask = (probs > threshold) | (probs < (1 - threshold))  # Boolean mask

    pseudo_labels = (probs > 0.5).float()

    # Compute percentage of confident pixels per image
    B, _, H, W = logits.shape
    total_pixels = H * W
    confidence_ratio = mask.float().sum(dim=[2, 3]) / total_pixels  # Shape: [B, 1]

    return pseudo_labels, mask.float(), confidence_ratio

    
def masked_loss(preds, pseudo_labels, mask):
    #loss = F.binary_cross_entropy_with_logits(preds, pseudo_labels, reduction='none')
    loss = nn.BCEWithLogitsLoss(reduction = 'none')(preds,pseudo_labels)
    loss = loss * mask  # Only keep high-confidence pixels
    return loss.mean()

def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=[2,3])
    union = p.sum(dim=[2,3]) + target.sum(dim=[2,3]) + eps
    return 1 - (2*inter/union).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
    
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # Apply sigmoid if logits are provided
        preds = torch.sigmoid(preds)

        # Flatten tensors
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Intersection and Union
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        return 1 - iou.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, iou_weight=1.0, eps=1e-6):
        super(CombinedLoss,self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou = IoULoss(eps=eps)
        self.iou_weight = iou_weight
    
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = dice_loss(logits, targets)
        iou = self.iou(logits,targets)
        return self.bce_weight * bce + self.dice_weight * dice + self.iou_weight * iou
    
    
class StructureLoss(nn.Module):
    def __init__(self, weight_edge=5.0):
        super(StructureLoss, self).__init__()
        self.weight_edge = weight_edge

    def forward(self, pred, mask):
        """
        pred: predicted probability map (B x 1 x H x W)
        mask: ground truth binary mask (B x 1 x H x W)
        """
        # Ensure pred is between 0 and 1
        pred = torch.sigmoid(pred)

        # ---------- Weight Map (focus on edges) ----------
        # Use average pooling to compute soft edges from ground truth
        kernel = torch.ones((1, 1, 31, 31), device=mask.device) / (31 * 31)
        avg_mask = F.conv2d(mask, kernel, padding=15)
        weight_map = 1 + self.weight_edge * torch.abs(avg_mask - mask)

        # ---------- Weighted BCE Loss ----------
        bce = F.binary_cross_entropy(pred, mask, reduction='none')
        wbce = (weight_map * bce).sum(dim=(2, 3)) / weight_map.sum(dim=(2, 3))

        # ---------- Weighted IoU Loss ----------
        intersection = (pred * mask * weight_map).sum(dim=(2, 3))
        union = (pred + mask) * weight_map
        union = union.sum(dim=(2, 3)) - intersection
        wiou = 1 - (intersection + 1) / (union + 1)  # smooth to avoid divide-by-zero

        # Combine
        loss = wbce + wiou
        return loss.mean()

class EnhancedStructureLoss(nn.Module):
    def __init__(self, 
                 bce_weight=1.0, 
                 iou_weight=1.0, 
                 dice_weight=1.0, 
                 edge_weight=5.0, 
                 eps=1e-6):
        super(EnhancedStructureLoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.eps = eps

    def forward(self, pred, mask):
        """
        pred: raw logits (B x 1 x H x W)
        mask: binary ground truth (B x 1 x H x W)
        """
        pred = torch.sigmoid(pred)

        # ---------- Weight Map ----------
        kernel_size = 31
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device) / (kernel_size ** 2)
        avg_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)
        weight_map = 1 + self.edge_weight * torch.abs(avg_mask - mask)

        # ---------- Weighted BCE ----------
        bce = F.binary_cross_entropy(pred, mask, reduction='none')
        wbce = (weight_map * bce).sum(dim=(2, 3)) / (weight_map.sum(dim=(2, 3)) + self.eps)

        # ---------- Weighted IoU ----------
        intersection = (pred * mask * weight_map).sum(dim=(2, 3))
        total = (pred + mask) * weight_map
        union = total.sum(dim=(2, 3)) - intersection
        wiou = 1 - (intersection + self.eps) / (union + self.eps)

        # ---------- Weighted Dice ----------
        inter = (pred * mask * weight_map).sum(dim=(2, 3))
        pred_sum = (pred * weight_map).sum(dim=(2, 3))
        mask_sum = (mask * weight_map).sum(dim=(2, 3))
        wdice = 1 - (2 * inter + self.eps) / (pred_sum + mask_sum + self.eps)

        # ---------- Combine ----------
        loss = (self.bce_weight * wbce + 
                self.iou_weight * wiou + 
                self.dice_weight * wdice)

        return loss.mean()


def cosine_rampup(current_epoch, rampup_epochs, max_value=0.4):
    if current_epoch >= rampup_epochs:
        return max_value
    else:
        return max_value * 0.5 * (1 - math.cos(math.pi * current_epoch / rampup_epochs))

def cosine_threshold_scheduler(current_epoch, warmup_epochs, start=0.7, end=0.9):
    if current_epoch >= warmup_epochs:
        return end
    else:
        return end - (end - start) * 0.5 * (1 + math.cos(math.pi * current_epoch / warmup_epochs))

def ema_decay_schedule(current_epoch, total_epochs, start=0.95, end=0.9999):
    if current_epoch >= total_epochs:
        return end
    else:
        return end - (end - start) * 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))

def clip_contrastive_loss(image_features, text_features, temperature=0.05):
    # image_features and text_features shape: (B,C)
    
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute similarity matrix (batch_size x batch_size)
    logits = (image_features @ text_features.T) / temperature  # Cosine similarity divided by temperature

    # Create labels (identity matrix)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size).to(image_features.device)

    # Compute cross-entropy loss
    loss_img = F.cross_entropy(logits, labels)
    loss_txt = F.cross_entropy(logits.T, labels)

    return (loss_img + loss_txt) / 2  # Average loss for image and text

import torch.nn as nn

def select_loss_function(
    simple_loss=True,
    iou_loss=False,
    bce_dice_loss=False,
    structure_loss=False,
    enhanced_structure_loss=False,
    logger=None,
    rank=0
):
    """
    Returns the appropriate loss function based on flags.
    """
    if simple_loss:
        if rank == 0 and logger:
            logger.info('Using simple BCEWithLogitsLoss')
        return nn.BCEWithLogitsLoss()
    
    if bce_dice_loss:
        if rank == 0 and logger:
            logger.info('Using BCE with Dice loss')
        return CombinedLoss(bce_weight=0.5, dice_weight=1, iou_weight=0.0)
    
    if iou_loss:
        # Placeholder for CombinedLoss (You should implement or import this)
        if rank == 0 and logger:
            logger.info('Using Combined Loss (BCE + Dice + IoU)')
        return CombinedLoss(bce_weight=0.2, dice_weight=1.5, iou_weight=0.8)
    
    if structure_loss:
        # Placeholder for StructureLoss (You should implement or import this)
        if rank == 0 and logger:
            logger.info('Using Structure Loss')
        return StructureLoss()
    
    if enhanced_structure_loss:
        # Placeholder for EnhancedStructureLoss (You should implement or import this)
        if rank == 0 and logger:
            logger.info('Using Enhanced Structure Loss')
        return EnhancedStructureLoss()
    
    # Default fallback
    if rank == 0 and logger:
        logger.info('Fallback to BCEWithLogitsLoss')
    return nn.BCEWithLogitsLoss()

import os
import torch
import gc

def load_checkpoint_clip_only(
    clip_model, clip_decoder, optimizer, scheduler, resume_path,
    finetune_clip, clip_decoder_head, use_scheduler, logger, rank, logit_scale=None
):
    """
    Loads checkpoint and restores states for CLIP model training.
    Returns start_epoch, best_val, best_IoU.
    """
    start_epoch = 0
    best_val = 10000
    best_IoU = 0.01

    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
            if rank == 0 and logger:
                logger.info(f"Resuming from checkpoint at {resume_path}")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to load checkpoint: {e}")
            return start_epoch, best_val, best_IoU

        # Load CLIP model
        clip_model.module.load_state_dict(checkpoint['clip_model'])

        # Load CLIP decoder head if exists
        if clip_decoder_head and 'clip_decoder' in checkpoint and checkpoint['clip_decoder']:
            clip_decoder.module.load_state_dict(checkpoint['clip_decoder'])

        if logit_scale is not None and 'logit_scale' in checkpoint:
            logit_scale.data = checkpoint['logit_scale']
            if rank == 0 and logger:
                logger.info(f"Loaded logit_scale: {logit_scale.exp().item():.4f}")


        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0 and logger:
                logger.info('Loaded optimizer state.')

        # Load scheduler state if exists
        if use_scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if rank == 0 and logger:
                logger.info('Loaded scheduler state.')

        # Restore tracking variables
        start_epoch = checkpoint.get('epoch', start_epoch) + 1
        best_val = checkpoint.get('best_val', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)

        gc.collect()
    else:
        if rank == 0 and logger:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")

    return start_epoch, best_val, best_IoU

import torch
import os

def save_checkpoint_clip_only(
    epoch, clip_model, clip_decoder, optimizer, scheduler,
    val_loss, best_val, best_IoU,
    save_path, save_ext,
    finetune_clip, clip_decoder_head, use_scheduler,
    logger, best=False, final=False, logit_scale=None
):
    """
    Saves checkpoint for CLIP model training.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if best:
        filename = f"supervised_best_clip_IoU_{save_ext}.pth"
    elif final:
        filename = f"supervised_clip_epoch_{epoch+1}{save_ext}.pth"
    else:
        filename = f"supervised_clip_saved_{save_ext}.pth"

    ckpt = {
        'epoch': epoch,
        'clip_model': clip_model.module.state_dict() if finetune_clip else None,
        'logit_scale': logit_scale.data if logit_scale is not None else None,
        'clip_decoder': clip_decoder.module.state_dict() if clip_decoder_head else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
        'val_loss': val_loss,
        'best_val': best_val,
        'best_IoU': best_IoU
    }

    ckpt_path = os.path.join(save_path, filename)
    torch.save(ckpt, ckpt_path)

    if logger:
        logger.info(f"Saved checkpoint to {ckpt_path}")


