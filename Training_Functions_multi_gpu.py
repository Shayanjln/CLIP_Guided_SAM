from CLIP_SAM_Utils_Final_MultiGPU import *
import wandb
import traceback
import torch.distributed as dist
from tqdm import trange
import gc



def return_mean_IoU(dataloader_val,sam,clip_model,classes,has_mask_prompt,is_vanilla_sam,SAM_text,
                   points_from_gt=False):
    total_IoU = 0
    for batch in dataloader_val:
        with torch.no_grad():
            output, *_ = forward_supervised_SAM_CLIP_multiGPU(batch,clip_model,sam,classes,
                               has_mask_prompt = has_mask_prompt, include_backgrounds = False, 
                                                                SAM_text = SAM_text,is_vanilla_sam = False, 
                                                                features_input = False, points_from_gt=False)
        
            pred = get_binary_output(output, out_tensor=True)
            target = batch['label']
            flt_pred = pred.flatten().cpu().bool()
            flt_target = target.flatten().bool()
            intersection = (flt_pred & flt_target).sum().item()
            union = (flt_pred | flt_target).sum().item()
            total_IoU = total_IoU + intersection/(union + 1e-10)
    
    total_IoU_mean = total_IoU/len(dataloader_val)
    return total_IoU_mean

def forward_supervised_SAM_CLIP_multiGPU(batch, clip_model, sam, classes, clip_crop_size=224,
                               has_mask_prompt=False, include_backgrounds=False, SAM_text=True, prompt_ensemble=False,
                               is_vanilla_sam=False, features_input=False, points_from_gt=False, thresh=0.8, sim_func_surgery=False, 
                               norm_dim=1, redundant_feat=False, parallel_sim=False, text_vis=False,
                               clip_decoder_head=False, clip_decoder=None, logit_scale=None,
                               use_soft_prompts=False, soft_prompt_sigmoid=False):    
    #get the main device
    device = next(sam.parameters()).device
    # Get inputs from batch and infer device from image tensor
    imgs = batch['image'].to(device ,dtype=torch.float32)
    #device = imgs.device  # Use this to move any new tensors later
    img_class = batch['text']
    label_masks = batch['label'].to(device,dtype=torch.float32)
    
    # Resize for CLIP input and normalize
    imgs_clip = torch.nn.functional.interpolate(imgs, size=(clip_crop_size, clip_crop_size), mode="bilinear", align_corners=False)
    clip_image_embeddings = clip_model(imgs_clip, 'image')
    image_features_clip = clip_image_embeddings / clip_image_embeddings.norm(dim=norm_dim, keepdim=True)

    #print('image features clip shape:',image_features_clip.shape)
    
    # Get text features
    # text_features = get_text_features(
    #     clip_model, classes, img_class,
    #     simple_prompt= not prompt_ensemble, prompt_ensemble=prompt_ensemble, custom_prompt=False
    # )
    text_features = get_text_features_v3(
        clip_model, classes, img_class,
        simple_prompt= not prompt_ensemble, prompt_ensemble=prompt_ensemble, custom_prompt=False
    )
    #text_features = get_text_features_v2(clip_model,img_class, simple_prompt=True)
    redundant_features = clip.encode_text_with_prompt_ensemble(clip_model, [""], device)
    text_features = text_features - redundant_features
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    #print('text feature shape:',text_features.shape)

    # Compute similarity and similarity map
    #features = get_sim_features(image_features_clip, text_features)
    if sim_func_surgery:
        redundant_features = clip.encode_text_with_prompt_ensemble(clip_model, [""], device)
        features = get_sim_features_surgery(image_features_clip, text_features,redundant_features=redundant_features)
    else:    
        features = get_sim_features(image_features_clip, text_features)
    # features shape: [1,1025,1]
    similarity_map = clip.get_similarity_map(features[:, 1:, :], imgs.shape[2:])

    if logit_scale is not None:
        similarity_map = similarity_map * logit_scale.exp()

    similarity_prompt_orig_size = similarity_map.permute(0, 3, 1, 2).to(device)

    if clip_decoder_head:
        similarity_prompt_orig_size = clip_decoder(features,text_features)
    
    num_points = 5
    # Get point prompts
    if points_from_gt:
        #mask_resized = torch.nn.functional.interpolate(label_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        # mask_resized = torch.nn.functional.interpolate(label_masks, size=(1024, 1024), mode="nearest")
        #points, point_labels = sample_random_points_from_similarity(mask_resized * 255, num_points=10)
        # points, point_labels = sample_random_points_vectorized(mask_resized * 255, num_points=10)
        try:
            mask_resized = torch.nn.functional.interpolate(label_masks, size=(1024, 1024), mode="nearest")
            #points, point_labels = sample_random_points_from_similarity(mask_resized * 255, num_points=10)
            points, point_labels = sample_random_points_vectorized(mask_resized * 255, num_points=num_points, both=include_backgrounds)
        except Exception as e:
            print('Sampling failed with label mask.')
            # print(f"Sampling failed with label mask. Falling back to CLIP mask. Reason: {e}")
        #     binary_sim = (similarity_prompt_orig_size > thresh) * 255
        #     #points, point_labels = sample_random_points_from_similarity(binary_sim, num_points=10)
        #     points, point_labels = sample_random_points_vectorized(binary_sim, num_points=num_points)
        #     if include_backgrounds:
        #         binary_sim_neg = (similarity_prompt_orig_size < 1-thresh) * 255
        #         points_neg, point_labels_neg = sample_random_points_vectorized(binary_sim_neg, num_points=num_points, label=1)
        #         point_labels_neg = point_labels_neg*0
        #         points = torch.cat((points,points_neg),dim=1)
        #         point_labels = torch.cat((point_labels,point_labels_neg),dim=1)
    else:
        binary_sim = (similarity_prompt_orig_size > thresh) * 255
        #points, point_labels = sample_random_points_from_similarity(binary_sim, num_points=10)
        points, point_labels = sample_random_points_vectorized(binary_sim, num_points=num_points, both=include_backgrounds)
        #points, point_labels = sample_random_points_with_probabilities(similarity_prompt_orig_size, num_points=num_points, label=1, both=include_backgrounds, threshold=thresh, sample_from_probabilities=True)
        
        if include_backgrounds:
            binary_sim_neg = (similarity_prompt_orig_size < 1-thresh) * 255
            points_neg, point_labels_neg = sample_random_points_vectorized(binary_sim_neg, num_points=num_points, label=1)
            point_labels_neg = point_labels_neg*0
            points = torch.cat((points,points_neg),dim=1)
            point_labels = torch.cat((point_labels,point_labels_neg),dim=1)
            
    similarity_prompt = torch.nn.functional.interpolate(similarity_prompt_orig_size, size=(256, 256), mode='bilinear', align_corners=False)

    # # Convert point tensors
    # points = torch.tensor(np.array(points), dtype=torch.float32, device=device)
    # point_labels = torch.tensor(np.array(point_labels), dtype=torch.int32, device=device)

    # Compose inputs based on prompt type
    if has_mask_prompt:
        # use soft prompts:
        if use_soft_prompts:
            # use of soft prompts is to ensure the mask prompt generation has a gradient path, so CLIP can also focus on generating better prompts
            prompt_mask = similarity_prompt * 255
            if soft_prompt_sigmoid:
                tau = 0.1  # temperature for sigmoid scaling, adjust as needed
                prompt_mask = torch.sigmoid((prompt_mask - thresh) / tau) # tau ~ 0.05–0.2
        else: # use hard prompts:
            #print('in first else of has_mask_prompt, using hard prompts, not soft prompts')
            prompt_mask = (similarity_prompt > thresh) * 255.0

        if parallel_sim:
            inputs = [imgs, points, point_labels, prompt_mask, features[:,1:,0]]
        elif text_vis:
            # features shape: [1,1025,1], image_features_clip shape: [1,1024,512]
            #print('shapes: features:', features.shape, 'image_features_clip:', image_features_clip.shape)
            image_features_input = image_features_clip + features
            #image_features_input = features.expand(-1, -1, 512)
            #inputs = [imgs, points, point_labels, prompt_mask, text_features, image_features_clip]
            #inputs = [imgs, points, point_labels, prompt_mask, text_features, image_features_input]
            #print('in text_vis branch of has_mask_prompt, using image features + sim features as input to SAM, along with text features and mask prompt')
            inputs = [imgs, points, point_labels, prompt_mask, text_features, image_features_input]
        else:
            #print('in second else of has_mask_prompt, not using parallel sim or text_vis, just using text features, and using mask prompts')
            inputs = [imgs, points, point_labels, prompt_mask, text_features]
    elif SAM_text:
        if text_vis:
            image_features_input = image_features_clip + features
            #inputs = [imgs, points, point_labels, None, text_features, image_features_input]
        else:
            inputs = [imgs, points, point_labels, text_features]
    elif features_input:
        features_input_tensor = features.squeeze(-1)[:, 1:]
        inputs = [imgs, points, point_labels, features_input_tensor]
    elif parallel_sim:
        inputs = [imgs, points, point_labels, features[:,1:,0]]
    else:
        #print('only image, points, and point labels')
        inputs = [imgs, points, point_labels]
    # Forward pass through SAM
    #print('inputs length:', len(inputs))
    sam_output = sam(inputs)
 
    return sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, inputs

    

def forward_supervised_SAM_CLIP_multiGPU_2(imgs,img_class,label_masks, clip_model, sam, classes, clip_crop_size=224,
                               has_mask_prompt=False, include_backgrounds=False, SAM_text=True, prompt_ensemble=False,
                               is_vanilla_sam=False, features_input=False, points_from_gt=False, thresh=0.8, sim_func_surgery=False, 
                               norm_dim=1, redundant_feat=False):
    #get the main device
    device = next(sam.parameters()).device
    # Get inputs from batch and infer device from image tensor
    imgs = imgs.to(device ,dtype=torch.float32)
    #device = imgs.device  # Use this to move any new tensors later
    #img_class = batch['text']
    label_masks = label_masks.to(device,dtype=torch.float32)
    
    # Resize for CLIP input and normalize
    imgs_clip = torch.nn.functional.interpolate(imgs, size=(clip_crop_size, clip_crop_size), mode="bilinear", align_corners=False)
    clip_image_embeddings = clip_model(imgs_clip, 'image')
    image_features_clip = clip_image_embeddings / clip_image_embeddings.norm(dim=norm_dim, keepdim=True)

    # Get text features
    text_features = get_text_features(
        clip_model, classes, img_class,
        simple_prompt= not prompt_ensemble, prompt_ensemble=prompt_ensemble, custom_prompt=False
    )
    redundant_features = clip.encode_text_with_prompt_ensemble(clip_model, [""], device)
    text_features = text_features - redundant_features
    

    # Compute similarity and similarity map
    #features = get_sim_features(image_features_clip, text_features)
    if sim_func_surgery:
        redundant_features = clip.encode_text_with_prompt_ensemble(clip_model, [""], device)
        features = get_sim_features_surgery(image_features_clip, text_features,redundant_features=redundant_features)
    else:    
        features = get_sim_features(image_features_clip, text_features)
    similarity_map = clip.get_similarity_map(features[:, 1:, :], imgs.shape[2:])
    similarity_prompt_orig_size = similarity_map.permute(0, 3, 1, 2).to(device)
    
    num_points = 5
    # Get point prompts
    if points_from_gt:
        #mask_resized = torch.nn.functional.interpolate(label_masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        # mask_resized = torch.nn.functional.interpolate(label_masks, size=(1024, 1024), mode="nearest")
        #points, point_labels = sample_random_points_from_similarity(mask_resized * 255, num_points=10)
        # points, point_labels = sample_random_points_vectorized(mask_resized * 255, num_points=10)
        try:
            mask_resized = torch.nn.functional.interpolate(label_masks, size=(1024, 1024), mode="nearest")
            #points, point_labels = sample_random_points_from_similarity(mask_resized * 255, num_points=10)
            points, point_labels = sample_random_points_vectorized(mask_resized * 255, num_points=num_points, both=include_backgrounds)
        except Exception as e:
            print(f"Sampling failed with label mask. Falling back to CLIP mask. Reason: {e}")
            binary_sim = (similarity_prompt_orig_size > thresh) * 255
            #points, point_labels = sample_random_points_from_similarity(binary_sim, num_points=10)
            points, point_labels = sample_random_points_vectorized(binary_sim, num_points=num_points)
            if include_backgrounds:
                binary_sim_neg = (similarity_prompt_orig_size < 1-thresh) * 255
                points_neg, point_labels_neg = sample_random_points_vectorized(binary_sim_neg, num_points=num_points, label=1)
                point_labels_neg = point_labels_neg*0
                points = torch.cat((points,points_neg),dim=1)
                point_labels = torch.cat((point_labels,point_labels_neg),dim=1)
    else:
        binary_sim = (similarity_prompt_orig_size > thresh) * 255
        #points, point_labels = sample_random_points_from_similarity(binary_sim, num_points=10)
        points, point_labels = sample_random_points_vectorized(binary_sim, num_points=num_points, both=include_backgrounds)
        #points, point_labels = sample_random_points_with_probabilities(similarity_prompt_orig_size, num_points=num_points, label=1, both=include_backgrounds, threshold=thresh, sample_from_probabilities=True)

        if include_backgrounds:
            binary_sim_neg = (similarity_prompt_orig_size < 1-thresh) * 255
            points_neg, point_labels_neg = sample_random_points_vectorized(binary_sim_neg, num_points=num_points, label=1)
            point_labels_neg = point_labels_neg*0
            points = torch.cat((points,points_neg),dim=1)
            point_labels = torch.cat((point_labels,point_labels_neg),dim=1)
            
    similarity_prompt = torch.nn.functional.interpolate(similarity_prompt_orig_size, size=(256, 256), mode='bilinear', align_corners=False)

    # # Convert point tensors
    # points = torch.tensor(np.array(points), dtype=torch.float32, device=device)
    # point_labels = torch.tensor(np.array(point_labels), dtype=torch.int32, device=device)

    # Compose inputs based on prompt type
    if has_mask_prompt:
        prompt_mask = (similarity_prompt > thresh) * 255.0
        inputs = [imgs, points, point_labels, prompt_mask, text_features]
    elif SAM_text:
        inputs = [imgs, points, point_labels, text_features]
    elif features_input:
        features_input_tensor = features.squeeze(-1)[:, 1:]
        inputs = [imgs, points, point_labels, features_input_tensor]
    else:
        inputs = [imgs, points, point_labels]

    # Forward pass through SAM
    sam_output = sam(inputs)

    return sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, inputs

def forward_supervised_clip(batch, clip_model, clip_crop_size=224, norm_dim=1, prompt_ensemble=False, 
                            clip_decoder_head=False, clip_decoder=None, logit_scale=None):
    """
    Forward function for supervised CLIP-only segmentation training.

    Args:
        batch: A dict with keys 'image', 'text', 'label'.
        clip_model: CLIP model instance.
        clip_crop_size: Crop size for CLIP input.
        norm_dim: Normalization dimension.
        prompt_ensemble: Whether to use prompt ensemble.
        clip_decoder_head: Whether to use CLIP decoder head.
        clip_decoder: CLIP decoder model.

    Returns:
        similarity_map_resized: The predicted segmentation map (batch_size, 1, H, W).
        label_masks: Ground truth masks (batch_size, 1, H, W).
    """
    device = next(clip_model.parameters()).device

    # Get inputs from batch
    imgs = batch['image'].to(device, dtype=torch.float32)
    img_class = batch['text']
    label_masks = batch['label'].to(device, dtype=torch.float32)

    # Resize and normalize image for CLIP
    imgs_clip = torch.nn.functional.interpolate(imgs, size=(clip_crop_size, clip_crop_size), mode="bilinear", align_corners=False)
    clip_image_embeddings = clip_model(imgs_clip, 'image')
    image_features_clip = clip_image_embeddings / clip_image_embeddings.norm(dim=norm_dim, keepdim=True)

    # Get text features
    text_features = get_text_features(
        clip_model, classes=None, img_class=img_class,
        simple_prompt=not prompt_ensemble, prompt_ensemble=prompt_ensemble, custom_prompt=False
    )

    # Compute similarity features
    features = get_sim_features(image_features_clip, text_features)

    
    

    # Optional CLIP Decoder Head
    if clip_decoder_head and clip_decoder is not None:
        similarity_map = clip_decoder(features, text_features)
    else:
        # Compute similarity map (spatial)
        similarity_map = clip.get_similarity_map(features[:, 1:, :], imgs.shape[2:])
        similarity_map = similarity_map.permute(0, 3, 1, 2).to(device)  # Shape: (B, 1, H, W)
        # Resize to match GT label mask size
        similarity_map = torch.nn.functional.interpolate(similarity_map, size=label_masks.shape[2:], mode='bilinear', align_corners=False)

    if logit_scale is not None:
        similarity_map = similarity_map * logit_scale.exp()

    #similarity_map_resized =  similarity_map_resized * 10.0
    return similarity_map, label_masks

def eval_model(
        clip_model, sam, dataloader_val, classes, criterion,
        mask_prompt=False, include_backgrounds=False, SAM_text=False, return_IoU=False,
        features_input=False, is_vanilla_sam=False, points_from_gt=False):

    sam.eval()
    clip_model.eval()
    total_loss = 0
    total_IoU = 0
    total_images = 0
    with torch.no_grad():
        pbar = tqdm(dataloader_val, desc="Evaluating")
        for i, batch in enumerate(pbar):
            sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = \
                forward_supervised_SAM_CLIP_multiGPU(batch, clip_model, sam, classes,
                    has_mask_prompt=mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=points_from_gt)

            # Calculate loss
            loss = criterion(sam_output, label_masks)
            total_loss += loss.item()
            running_avg_loss = total_loss / (i + 1)

            # Compute per-image IoU
            if return_IoU:
                pred = get_binary_output(sam_output, out_tensor=True)
                target = label_masks

                for pred_i, target_i in zip(pred, target):
                    inter = (pred_i.bool() & target_i.bool()).sum().float()
                    union = (pred_i.bool() | target_i.bool()).sum().float()
                    if union == 0:
                        iou = 1.0  # perfect match when both empty
                    else:
                        iou = inter / (union + 1e-10)
                    total_IoU += iou
                    total_images += 1

                running_avg_IoU = total_IoU / total_images
                pbar.set_postfix({
                    'avg_val_loss': f'{running_avg_loss:.4f}',
                    'avg_IoU': f'{running_avg_IoU:.4f}'
                })
            else:
                pbar.set_postfix({'avg_val_loss': f'{running_avg_loss:.4f}'})

    avg_loss = total_loss / len(dataloader_val)

    if return_IoU:
        mean_IoU = (total_IoU / total_images).item()
        return mean_IoU, avg_loss
    else:
        return avg_loss

def eval_model_DDP(
        clip_model, sam, dataloader_val, classes, criterion, clip_crop_size,
        mask_prompt=False, include_backgrounds=False, SAM_text=False, return_IoU=False,
        features_input=False, is_vanilla_sam=False, points_from_gt=False, prompt_ensemble=False, 
        sim_func_surgery=False, norm_dim=1, parallel_sim=False, text_vis=False, 
        clip_decoder_head=False, clip_decoder=None, out_size_512=True):

    sam.eval()
    clip_model.eval()
    total_loss = 0.0
    total_IoU = 0.0
    total_images = 0

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    with torch.no_grad():
        dataloader_iter = tqdm(dataloader_val, desc="Evaluating") if rank == 0 else dataloader_val
        for i, batch in enumerate(dataloader_iter):
            sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = \
                forward_supervised_SAM_CLIP_multiGPU(batch, clip_model, sam, classes,
                    has_mask_prompt=mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam, clip_crop_size=clip_crop_size,
                    features_input=features_input, points_from_gt=points_from_gt, norm_dim=norm_dim,
                    prompt_ensemble=prompt_ensemble, sim_func_surgery=sim_func_surgery,
                    parallel_sim=parallel_sim, text_vis=text_vis, clip_decoder_head=clip_decoder_head, 
                    clip_decoder=clip_decoder)
            
            if out_size_512:
                sam_output = torch.nn.functional.interpolate(sam_output, size=(512, 512), mode="bilinear", align_corners=False)

            loss = criterion(sam_output, label_masks)
            total_loss += loss

            if return_IoU:
                pred = get_binary_output(sam_output, out_tensor=True)
                target = label_masks
                for pred_i, target_i in zip(pred, target):
                    inter = (pred_i.bool() & target_i.bool()).sum().float()
                    union = (pred_i.bool() | target_i.bool()).sum().float()
                    iou = 1.0 if union == 0 else inter / (union + 1e-10)
                    total_IoU += iou
                    total_images += 1

    # Sync losses and IoUs across all processes
    num_batches_tensor = torch.tensor(len(dataloader_val), device='cuda')
    if return_IoU:
        total_images_tensor = torch.tensor(total_images, device='cuda')

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        if return_IoU:
            dist.all_reduce(total_IoU, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_images_tensor, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss / num_batches_tensor).item()

    if return_IoU:
        mean_IoU = (total_IoU / total_images_tensor).item()
        return mean_IoU, avg_loss
    else:
        return avg_loss

def eval_model_DDP_per_class_IoU(
        clip_model, sam, dataloader_val, classes, criterion=None, clip_crop_size=224,
        mask_prompt=False, include_backgrounds=False, SAM_text=False, return_IoU=False,
        features_input=False, is_vanilla_sam=False, points_from_gt=False, prompt_ensemble=False, 
        sim_func_surgery=False, norm_dim=1, parallel_sim=False, text_vis=False, 
        clip_decoder_head=False, clip_decoder=None, out_size_512=True, logit_scale=None,
        return_MAE=False, use_soft_prompts=False, soft_prompt_sigmoid=False):

    def _eval_single(loader):
        sam.eval()
        clip_model.eval()

        total_loss = torch.tensor(0.0, device='cuda')
        # ----- MAE accumulators -----
        total_MAE = torch.tensor(0.0, device='cuda')
        total_px  = torch.tensor(0.0, device='cuda')   # number of images accumulated for MAE
        # ----- per-class IoU accumulators -----
        num_classes     = len(classes)
        per_class_inter = torch.zeros(num_classes, device='cuda')
        per_class_union = torch.zeros(num_classes, device='cuda')
        class_counts    = torch.zeros(num_classes, device='cuda')  # not strictly needed for IoU, kept for parity

        rank        = dist.get_rank()  if dist.is_initialized() else 0
        show_prog   = (rank == 0)

        class_name_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

        with torch.no_grad():
            loader_iter = tqdm(loader, desc="Evaluating", dynamic_ncols=True) if show_prog else loader

            for batch in loader_iter:
                # Forward
                sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = \
                    forward_supervised_SAM_CLIP_multiGPU(
                        batch, clip_model, sam, classes,
                        has_mask_prompt=mask_prompt, include_backgrounds=include_backgrounds,
                        SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam, clip_crop_size=clip_crop_size,
                        features_input=features_input, points_from_gt=points_from_gt, norm_dim=norm_dim,
                        prompt_ensemble=prompt_ensemble, sim_func_surgery=sim_func_surgery,
                        parallel_sim=parallel_sim, text_vis=text_vis, clip_decoder_head=clip_decoder_head, 
                        clip_decoder=clip_decoder, logit_scale=logit_scale, 
                        use_soft_prompts=use_soft_prompts, soft_prompt_sigmoid=soft_prompt_sigmoid)

                # Resize to 512 if requested
                preds = sam_output
                if out_size_512:
                    preds = torch.nn.functional.interpolate(
                        preds, size=(512, 512), mode="bilinear", align_corners=False
                    )

                # Loss (optional)
                if criterion is not None:
                    total_loss += criterion(preds, label_masks)

                # ----- MAE -----
                if return_MAE:
                    prob      = preds.sigmoid()
                    mae_batch = torch.abs(prob - label_masks.float()).mean()  # mean over (N,H,W)
                    total_MAE += mae_batch * preds.size(0)
                    total_px  += preds.size(0)

                # ----- IoU (per-class aggregated over samples’ class labels) -----
                if return_IoU:
                    bin_pred   = get_binary_output(preds, out_tensor=True)
                    targets    = label_masks
                    class_list = batch["text"]  # could be str or list(str)

                    for p_i, t_i, class_name in zip(bin_pred, targets, class_list):
                        # Normalize class name (use first synonym if list)
                        name = class_name[0] if isinstance(class_name, list) else class_name
                        cls_id = class_name_to_id[name]

                        inter = (p_i.bool() & t_i.bool()).sum().float()
                        union = (p_i.bool() | t_i.bool()).sum().float()

                        per_class_inter[cls_id] += inter
                        per_class_union[cls_id] += union
                        class_counts[cls_id]    += 1

                        if show_prog:
                            valid_mask = per_class_union > 0
                            per_cls_iou = torch.zeros_like(per_class_inter)
                            per_cls_iou[valid_mask] = per_class_inter[valid_mask] / (per_class_union[valid_mask] + 1e-10)
                            mean_iou_so_far = per_cls_iou[valid_mask].mean().item()

                            # Show top-3 classes so far
                            top_k = min(3, int(valid_mask.sum().item()))
                            if top_k > 0:
                                vals, idxs = per_cls_iou[valid_mask].topk(top_k)
                                # Map back to original indices
                                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                                topk_global   = valid_indices[idxs]
                                topk_display  = {classes[int(i)]: f"{float(v):.3f}" for i, v in zip(topk_global, vals)}

                                pf = {'Mean IoU': f"{mean_iou_so_far:.4f}"}
                                pf.update(topk_display)
                                loader_iter.set_postfix(pf)

        # ----- sync across ranks -----
        num_batches_tensor = torch.tensor(len(loader), device='cuda')

        if dist.is_initialized():
            # Loss
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            # MAE
            if return_MAE:
                dist.all_reduce(total_MAE, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_px,  op=dist.ReduceOp.SUM)
            # IoU
            if return_IoU:
                dist.all_reduce(per_class_inter, op=dist.ReduceOp.SUM)
                dist.all_reduce(per_class_union, op=dist.ReduceOp.SUM)
                dist.all_reduce(class_counts,    op=dist.ReduceOp.SUM)

        # ----- finalize metrics -----
        avg_loss = (total_loss / num_batches_tensor).item() if criterion is not None else None

        outputs = []

        if return_IoU:
            valid_mask   = per_class_union > 0
            per_cls_iou  = torch.zeros_like(per_class_inter)
            per_cls_iou[valid_mask] = per_class_inter[valid_mask] / (per_class_union[valid_mask] + 1e-10)
            mean_IoU = per_cls_iou[valid_mask].mean().item() if valid_mask.any() else 0.0
            outputs.append(mean_IoU)

        if return_MAE:
            avg_MAE = (total_MAE / total_px).item() if total_px.item() > 0 else 0.0
            outputs.append(avg_MAE)

        if avg_loss is not None:
            outputs.append(avg_loss)

        # Return single metric or tuple (match your other helper’s behavior)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # ----- multi-val support: dict of {name: DataLoader} -----
    if isinstance(dataloader_val, dict):
        per_set = {}
        for name, loader in dataloader_val.items():
            per_set[name] = _eval_single(loader)

        # Decide how to average based on requested metrics
        keys_cnt = len(next(iter(per_set.values()))) if isinstance(next(iter(per_set.values())), tuple) else 1

        def _get_idx_for(field):
            # mapping when multiple metrics are returned:
            # return_IoU only         -> (IoU, loss?) or IoU
            # return_MAE only         -> (MAE, loss?) or MAE
            # both IoU & MAE          -> (IoU, MAE, loss?) or (IoU, MAE)
            if return_IoU and return_MAE:
                return {'iou': 0, 'mae': 1, 'loss': 2 if keys_cnt >= 3 else None}
            elif return_IoU:
                return {'iou': 0, 'mae': None, 'loss': 1 if keys_cnt >= 2 else None}
            elif return_MAE:
                return {'iou': None, 'mae': 0, 'loss': 1 if keys_cnt >= 2 else None}
            else:
                return {'iou': None, 'mae': None, 'loss': 0 if keys_cnt >= 1 else None}

        idxs = _get_idx_for('dummy')

        def _extract(v, idx):
            if idx is None:
                return None
            return v[idx] if isinstance(v, tuple) else v

        # compute simple arithmetic means across sets (already DDP-synced inside _eval_single)
        agg = {}
        if idxs['iou'] is not None:
            agg_iou = sum(_extract(v, idxs['iou']) for v in per_set.values()) / max(len(per_set), 1)
            agg['avg_iou'] = agg_iou
        if idxs['mae'] is not None:
            agg_mae = sum(_extract(v, idxs['mae']) for v in per_set.values()) / max(len(per_set), 1)
            agg['avg_mae'] = agg_mae
        if idxs['loss'] is not None:
            agg_loss = sum(_extract(v, idxs['loss']) for v in per_set.values()) / max(len(per_set), 1)
            agg['avg_loss'] = agg_loss

        # Return (per_set, ...averages...) mirroring your example helper
        # Order the tuple to be predictable:
        if return_IoU and return_MAE and 'avg_loss' in agg:
            return per_set, agg['avg_iou'], agg['avg_mae'], agg['avg_loss']
        if return_IoU and return_MAE:
            return per_set, agg['avg_iou'], agg['avg_mae']
        if return_IoU and 'avg_loss' in agg:
            return per_set, agg['avg_iou'], agg['avg_loss']
        if return_MAE and 'avg_loss' in agg:
            return per_set, agg['avg_mae'], agg['avg_loss']
        if return_IoU:
            return per_set, agg['avg_iou']
        if return_MAE:
            return per_set, agg['avg_mae']
        return per_set  # loss-only or nothing requested

    # ----- single-loader path -----
    return _eval_single(dataloader_val)

def eval_model_DDP_per_class_IoU_v0(
        clip_model, sam, dataloader_val, classes, criterion=None, clip_crop_size=224,
        mask_prompt=False, include_backgrounds=False, SAM_text=False, return_IoU=False,
        features_input=False, is_vanilla_sam=False, points_from_gt=False, prompt_ensemble=False, 
        sim_func_surgery=False, norm_dim=1, parallel_sim=False, text_vis=False, 
        clip_decoder_head=False, clip_decoder=None, out_size_512=True):

    sam.eval()
    clip_model.eval()
    total_loss = 0.0

    num_classes = len(classes)
    per_class_inter = torch.zeros(num_classes, device='cuda')
    per_class_union = torch.zeros(num_classes, device='cuda')
    class_counts = torch.zeros(num_classes, device='cuda')

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    class_name_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

    with torch.no_grad():
        show_progress = (rank == 0)
        dataloader_iter = tqdm(dataloader_val, desc="Evaluating", dynamic_ncols=True) if show_progress else dataloader_val

        for batch in dataloader_iter:
            sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = \
                forward_supervised_SAM_CLIP_multiGPU(batch, clip_model, sam, classes,
                    has_mask_prompt=mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam, clip_crop_size=clip_crop_size,
                    features_input=features_input, points_from_gt=points_from_gt, norm_dim=norm_dim,
                    prompt_ensemble=prompt_ensemble, sim_func_surgery=sim_func_surgery,
                    parallel_sim=parallel_sim, text_vis=text_vis, clip_decoder_head=clip_decoder_head, 
                    clip_decoder=clip_decoder)

            if out_size_512:
                sam_output = torch.nn.functional.interpolate(
                    sam_output, size=(512, 512), mode="bilinear", align_corners=False
                )

            if criterion is not None:
                loss = criterion(sam_output, label_masks)
                total_loss += loss

            if return_IoU:
                pred = get_binary_output(sam_output, out_tensor=True)
                target = label_masks
                class_names = batch["text"]

                for pred_i, target_i, class_name in zip(pred, target, class_names):
                    #print(class_name[0])
                    #print(class_name)
                    if isinstance(class_name, list):
                        name = class_name[0]
                    else:
                        name = class_name
                    #print(name)
                    class_id = class_name_to_id[name]
                    inter = (pred_i.bool() & target_i.bool()).sum().float()
                    union = (pred_i.bool() | target_i.bool()).sum().float()

                    per_class_inter[class_id] += inter
                    per_class_union[class_id] += union

                    if show_progress:
                        valid_mask = per_class_union > 0
                        per_class_iou = torch.zeros_like(per_class_inter)
                        per_class_iou[valid_mask] = per_class_inter[valid_mask] / (per_class_union[valid_mask] + 1e-10)
                        mean_IoU_so_far = per_class_iou[valid_mask].mean().item()

                        # Prepare a string for top K classes to display (e.g., top 3 classes with IoU)
                        top_k = 3  # Adjust as needed
                        topk_classes = per_class_iou[valid_mask].topk(min(top_k, valid_mask.sum().item()))
                        topk_display = {}
                        for idx, iou_val in zip(topk_classes.indices.tolist(), topk_classes.values.tolist()):
                            class_name = classes[idx]
                            topk_display[class_name] = f"{iou_val:.3f}"

                        # Update tqdm postfix
                        postfix_dict = {'Mean IoU': f"{mean_IoU_so_far:.4f}"}
                        postfix_dict.update(topk_display)
                        dataloader_iter.set_postfix(postfix_dict) 


                    class_counts[class_id] += 1

    num_batches_tensor = torch.tensor(len(dataloader_val), device='cuda')

    if dist.is_initialized():
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        if criterion is not None:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if return_IoU:
            dist.all_reduce(per_class_inter, op=dist.ReduceOp.SUM)
            dist.all_reduce(per_class_union, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss / num_batches_tensor).item() if criterion is not None else None

    if return_IoU:
        valid_mask = per_class_union > 0
        per_class_iou = torch.zeros_like(per_class_inter)
        per_class_iou[valid_mask] = per_class_inter[valid_mask] / (per_class_union[valid_mask] + 1e-10)
        mean_IoU = per_class_iou[valid_mask].mean().item()

        return (mean_IoU, avg_loss) if avg_loss is not None else mean_IoU
    else:
        return avg_loss

def eval_IoU(clip_model, sam, dataloader_val, classes,
        mask_prompt=False, include_backgrounds=False, SAM_text=False,
        features_input=False, is_vanilla_sam=False, points_from_gt=False):
    
    sam.eval()
    clip_model.eval()
    total_IoU = 0
    total_images = 0
    with torch.no_grad():
        pbar = tqdm(dataloader_val, desc="Evaluating")
        for i, batch in enumerate(pbar):
            sam_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = \
                forward_supervised_SAM_CLIP_multiGPU(batch, clip_model, sam, classes,
                    has_mask_prompt=mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=points_from_gt)

            # Compute per-image IoU
            pred = get_binary_output(sam_output, out_tensor=True)
            target = label_masks

            for pred_i, target_i in zip(pred, target):
                inter = (pred_i.bool() & target_i.bool()).sum().float()
                union = (pred_i.bool() | target_i.bool()).sum().float()
                if union == 0:
                    iou = 1.0  # perfect match when both empty
                else:
                    iou = inter / (union + 1e-10)
                total_IoU += iou
                total_images += 1

            running_avg_IoU = total_IoU / total_images
            pbar.set_postfix({
                'avg_IoU': f'{running_avg_IoU:.4f}'
            })

    mean_IoU = (total_IoU / total_images).item()
    return mean_IoU

import torch
from tqdm import tqdm
import torch.distributed as dist

def evaluate_clip_model(
    clip_model,
    val_loader,
    classes,
    criterion,
    clip_crop_size=224,
    prompt_ensemble=False,
    norm_dim=1,
    out_size_512=False,
    logit_scale=None,
    clip_decoder_head=False, 
    clip_decoder=None
):
    
    """
    CLIP-only evaluation loop for semantic segmentation (DDP-safe).
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    clip_model.eval()
    device = next(clip_model.parameters()).device

    total_loss = 0.0
    total_inter = 0.0
    total_union = 0.0
    total_samples = 0

    
    with torch.no_grad():
        total_batches = len(val_loader)
        half_batches = total_batches // 10
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", disable=(rank != 0))):
            # if batch_idx >= half_batches:
            #     break  # Stop after half the validation set
            # Forward Pass
            similarity_map, label_masks = forward_supervised_clip(
                batch,
                clip_model,
                clip_crop_size=clip_crop_size,
                norm_dim=norm_dim,
                prompt_ensemble=prompt_ensemble,
                logit_scale=logit_scale,
                clip_decoder_head=clip_decoder_head, 
                clip_decoder=clip_decoder
            )

            if out_size_512:
                similarity_map = torch.nn.functional.interpolate(similarity_map, size=(512, 512), mode="bilinear", align_corners=False)

            #print(f"Similarity Map Stats -> min: {similarity_map.min().item()}, max: {similarity_map.max().item()}, mean: {similarity_map.mean().item()}")
            # Loss Calculation
            loss = criterion(similarity_map, label_masks)
            total_loss += loss.item()

            # Threshold similarity map to binary mask (Sigmoid + Threshold 0.5)
            preds = torch.sigmoid(similarity_map) > 0.8

            #print(preds.float().mean())
            

            # Flatten for IoU calculation
            preds_flat = preds.view(preds.size(0), -1).float()
            targets_flat = label_masks.view(label_masks.size(0), -1).float()

            intersection = (preds_flat * targets_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

            total_inter += intersection.sum().item()
            total_union += union.sum().item()
            total_samples += preds.size(0)

    # ----- Synchronize Across GPUs -----
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_inter_tensor = torch.tensor(total_inter, device=device)
    total_union_tensor = torch.tensor(total_union, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)

    if dist.is_initialized():
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_inter_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_union_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    # Compute Metrics (global averages)
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    avg_IoU = (total_inter_tensor.item() + 1e-6) / (total_union_tensor.item() + 1e-6)

    return avg_IoU, avg_loss


def train_supervised_final(
    trainable_sam, clip_model,
    dataloader_labeled, dataloader_val,
    classes, 
    epochs, save_path, save_ext,
    SAM_text=False, has_mask_prompt=False, include_backgrounds=False,
    features_input=False, is_vanilla_sam=False, lr=1e-4, multi_gpu=False):
    if multi_gpu:
        trainable_sam = nn.DataParallel(trainable_sam)
        clip_model = nn.DataParallel(clip_model)

    optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    clip_model.eval()
    trainable_sam.train()

    best_IoU, best_val = 0.01, 10000

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - Starting Training Loop")
        
        progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        total_loss = 0
        exception_counter = 0
                
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            try:
                sam_supervised_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                    batch, clip_model, trainable_sam, classes, 
                    has_mask_prompt=has_mask_prompt, include_backgrounds=False, 
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam, 
                    features_input=features_input, points_from_gt=True
                )

                loss = criterion(sam_supervised_output, label_masks)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
    
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(loss=avg_loss)
    
            except ValueError:
                exception_counter += 1
                continue

        IoU, val_loss = eval_model(
            clip_model, trainable_sam, dataloader_val, classes, 
            criterion=criterion, mask_prompt=has_mask_prompt, 
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False
        )
            
        if val_loss < best_val:
            best_val = val_loss
            
        if IoU > best_IoU:
            best_IoU = IoU
            print(f"best val_loss: {val_loss}")
            sam_save_path = os.path.join(save_path, "fine_tuned_supervised_sam_best_IoU_" + save_ext + ".pth")
            torch.save(
                trainable_sam.module.state_dict() if multi_gpu else trainable_sam.state_dict(),
                sam_save_path
            )

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/(len(dataloader_labeled)-exception_counter)}, Val_Loss: {val_loss}, Best_Val: {best_val}")
        print(f"IoU: {IoU}, Best_IoU: {best_IoU}")
        
    return trainable_sam


def train_supervised_final_for_scripts(
    trainable_sam, clip_model,
    dataloader_labeled, dataloader_val,
    classes, device_trainable, 
    epochs, save_path, save_ext, logger, split,
    SAM_text=False, has_mask_prompt=False, include_backgrounds=False,
    features_input=False, is_vanilla_sam=False, lr=1e-4, weight_decay=1e-4,
    resume_path=None, points_from_gt=True
    ):
    trainable_sam = nn.DataParallel(trainable_sam)
    clip_model = nn.DataParallel(clip_model)
    print("Using Parallel/MultiGPU")
    logger.info(f"Using Parallel/MultiGPU")

    wandb.init(
        project="sam-clip-supervised-"+split,
        name=f"run_{save_ext}",
        config={
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": dataloader_labeled.batch_size,
            "sam_type": "vanilla" if is_vanilla_sam else "parallel",
            "SAM_text": SAM_text
        }
    )

    optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    #optimizer = optim.AdamW(trainable_sam.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    clip_model.eval()
    trainable_sam.train()

    best_IoU, best_val = 0.01, 10000
    start_epoch = 0  

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # continue from next epoch
        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        logger.info(f"Resumed training from checkpoint: {resume_path}, starting at epoch {start_epoch}")
        logger.info(f"Best IoU recovered from checkpoint: {best_IoU}")
    else:
        if resume_path:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")

        best_IoU, best_val = eval_model(
        clip_model, trainable_sam, dataloader_val, classes,
        criterion=criterion, mask_prompt=has_mask_prompt,
        include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
        points_from_gt=False
        )
        logger.info(f"Initial Eval -> IoU: {best_IoU:.4f}, Val Loss: {best_val:.4f}")

    end_epochs = start_epoch + epochs
    for epoch in range(start_epoch, start_epoch+epochs):  
        print(f"\nEpoch {epoch+1}/{end_epochs} - Starting Training Loop")
        logger.info(f"\nEpoch {epoch+1}/{end_epochs} - Starting Training Loop")
        progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}", leave=True)

        total_loss = 0
        exception_counter = 0

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            try:  #got multiple values for argument 'has_mask_prompt'
                sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                    batch, clip_model, trainable_sam, classes,
                    has_mask_prompt=has_mask_prompt,
                    include_backgrounds=False, SAM_text=SAM_text,
                    is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input,
                    points_from_gt=points_from_gt
                )

                loss = criterion(sam_output, label_masks)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

                avg_loss = total_loss / (batch_idx + 1)
                #progress_bar.set_postfix(loss=avg_loss)
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "exceptions": exception_counter
                })
                       
                      

            except ValueError as e:
                # print(f"[Batch {batch_idx}] ValueError: {e}")
                # traceback.print_exc()  # Optional: prints the full stack trace

                exception_counter += 1
                # progress_bar.set_postfix({"exceptions":exception_counter})
                continue

        IoU, val_loss = eval_model(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / (len(dataloader_labeled) - exception_counter),
            "val_loss": val_loss,
            "IoU": IoU,
            "Best_Val": best_val,
            "Best_IoU": best_IoU
        })

        if val_loss < best_val:
            best_val = val_loss

        if IoU > best_IoU:
            best_IoU = IoU
            print(f"best val_loss: {val_loss}")
            sam_save_path = os.path.join(save_path, "sam_best_IoU_ckpt_" + save_ext + ".pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_sam.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val': best_val,
                'best_IoU': best_IoU
            }, sam_save_path)
            logger.info(f"New best IoU: {best_IoU:.4f}. Best model checkpoint saved to {sam_save_path}")

        
        # if (epoch + 1) % 10 == 0:
        #     checkpoint_path = os.path.join(save_path, f"sam_ckpt_epoch_{epoch+1}{save_ext}.pth")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': trainable_sam.module.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #         'best_val': best_val,
        #         'best_IoU': best_IoU
        #     }, checkpoint_path)
        #     logger.info(f"Checkpoint saved at {checkpoint_path}")


        avg_epoch_loss = total_loss / (len(dataloader_labeled) - exception_counter)
        print(f"Epoch {epoch+1}/{end_epochs}, Train Loss: {avg_epoch_loss}, Val_Loss: {val_loss}, Best_Val: {best_val}")
        print(f"IoU: {IoU}, Best_IoU: {best_IoU}")
        logger.info(f"Epoch {epoch+1}/{end_epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val:.4f} | IoU: {IoU:.4f} | Best IoU: {best_IoU:.4f}")
    
    checkpoint_path = os.path.join(save_path, f"sam_ckpt_epoch_{epoch+1}{save_ext}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': trainable_sam.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val': best_val,
        'best_IoU': best_IoU
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")


def train_supervised_final_for_scripts_DDP(
    trainable_sam, clip_model,
    dataloader_labeled, dataloader_val,
    classes, total_epochs, clip_crop_size,
    epochs, save_path, save_ext, logger, split, weight_decay=0.0,
    SAM_text=False, has_mask_prompt=False, points_from_gt_train=False, points_from_gt_eval=False, finetune_clip=False,
    include_backgrounds=False, features_input=False, is_vanilla_sam=False, use_scheduler=True,
    lr_IE=1e-4, lr_MD=5e-5, lr_clip=1e-6, lr_PE=1e-5, resume_path=None, ignore_wandb=False, 
    skip_init_eval=False, simple_loss=True, iou_loss=False, structure_loss=False, enhanced_structure_loss=False,
    prompt_ensemble=False, sim_func_surgery=False, norm_dim=1, options=None, final_div_factor=1e2, pct_start=0.2,
    div_factor=25, sam_cross=False, lr_cr = 1e-4, parallel_sim = False, finetune_neck=False, lr_neck=1e-5, 
    text_vis=False, clip_decoder_head=False, clip_decoder=None, lr_clip_dec_head=1e-4, out_size_512=False, 
    logit_scale=None, dataset='COCO', use_mae=False, camo_multi_val=False, bce_weight=1, dice_weight=1, iou_weight=1,
    wandb_extension='', use_soft_prompts=False, soft_prompt_sigmoid=False):

    import os, gc, math, csv
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    from tqdm import tqdm
    # assumes wandb, eval_model_DDP_per_class_IoU, forward_supervised_SAM_CLIP_multiGPU,
    # CombinedLoss, StructureLoss, EnhancedStructureLoss, BCEDiceLoss are available in scope.

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0 and not ignore_wandb:
        wandb.init(
            project=f"sam-clip-supervised-ddp-{dataset}-"+split+wandb_extension,
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr_IE,
                "sam_type": "vanilla" if is_vanilla_sam else "parallel",
                "SAM_text": SAM_text,
                "use_mae": use_mae,
                "multi_val": isinstance(dataloader_val, dict) or camo_multi_val
            }
        )

    if rank==0: 
        logger.info(options)

    # ----- collect trainable params by groups -----
    adapter_params = [
        p for n, p in trainable_sam.module.image_encoder.named_parameters()
        if any(k in n for k in ['Space_Adapter','MLP_Adapter','Depth_Adapter']) and p.requires_grad
    ]

    if finetune_neck:
        neck_params = [
            p for n, p in trainable_sam.module.image_encoder.neck.named_parameters() if p.requires_grad
        ]
    if sam_cross:
        sam_cross_params = [
            p for n, p in trainable_sam.module.image_encoder.cross_blocks.named_parameters() if p.requires_grad
        ]
    mask_decoder_params = [p for n, p in trainable_sam.module.mask_decoder.named_parameters() if p.requires_grad]

    if has_mask_prompt:
        prompt_encoder_params = [
            p for n, p in trainable_sam.module.prompt_encoder.named_parameters() if p.requires_grad
        ]

    if finetune_clip:
        clip_model_params = [p for n, p in clip_model.module.named_parameters() if p.requires_grad]

    if clip_decoder_head:
        clip_dec_h_params = [p for _, p in clip_decoder.module.named_parameters() if p.requires_grad]

    optimizer = optim.AdamW(
        [
            {'params': adapter_params,     'lr': lr_IE,          'weight_decay': weight_decay},
            {'params': mask_decoder_params,'lr': lr_MD,          'weight_decay': weight_decay},
            *([{'params': clip_model_params,   'lr': lr_clip,        'weight_decay': 0.0}] if finetune_clip else []),
            *([{'params': prompt_encoder_params,'lr': lr_PE,         'weight_decay': weight_decay}] if has_mask_prompt else []),
            *([{'params': sam_cross_params,    'lr': lr_cr,          'weight_decay': weight_decay}] if sam_cross else []),
            *([{'params': neck_params,         'lr': lr_neck,        'weight_decay': weight_decay}] if finetune_neck else []),
            *([{'params': clip_dec_h_params,   'lr': lr_clip_dec_head,'weight_decay': weight_decay}] if clip_decoder_head else []),
            *([{'params': [logit_scale],       'lr': 1e-6,           'weight_decay': weight_decay}] if logit_scale is not None else [])
        ]
    )

    if use_scheduler:
        max_lr = [lr_IE, lr_MD] \
                 + ([lr_clip] if finetune_clip else []) \
                 + ([lr_PE] if has_mask_prompt else []) \
                 + ([lr_cr] if sam_cross else []) \
                 + ([lr_clip_dec_head] if clip_decoder_head else []) \
                 + ([1e-5] if logit_scale is not None else []) \
                 + ([lr_neck] if finetune_neck else [])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(dataloader_labeled),
            epochs=total_epochs,
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

    # ----- criterion -----
    if simple_loss:
        if rank==0: logger.info('using simple BCE')
        criterion = nn.BCEWithLogitsLoss()
    else:
        if iou_loss:
            bce_weight, dice_weight, iou_weight = bce_weight, dice_weight, iou_weight
            criterion = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, iou_weight=iou_weight)
            if rank==0: logger.info(f'using BCE, Dice, IoU with weights: {bce_weight},{dice_weight},{iou_weight}')
        elif structure_loss:
            criterion = StructureLoss()
            if rank==0: logger.info('using Structure loss')
        elif enhanced_structure_loss:
            criterion = EnhancedStructureLoss()
            if rank==0: logger.info('using Enhanced Structure loss')
        else:
            bce_weight, dice_weight = bce_weight, dice_weight
            criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
            if rank==0: logger.info(f'using BCE+Dice with weights: {bce_weight},{dice_weight}')

    # ----- train/eval modes for CLIP -----
    if finetune_clip:
        clip_model.train()
    else:
        clip_model.eval()
    trainable_sam.train()

    start_epoch = 0
    best_IoU, best_val = 0.01, 1e9
    best_MAE = 1e9

    # per-set best MAE tracker (only used when use_mae and multi-val)
    best_mae_per_set = {}  # name -> best MAE

    # ---------- helpers ----------
    def _set_epoch_on_val_loader(epoch):
        if isinstance(dataloader_val, dict):
            for _name, _ldr in dataloader_val.items():
                if hasattr(_ldr, "sampler") and hasattr(_ldr.sampler, "set_epoch"):
                    _ldr.sampler.set_epoch(epoch)
        else:
            if hasattr(dataloader_val, "sampler") and hasattr(dataloader_val.sampler, "set_epoch"):
                dataloader_val.sampler.set_epoch(epoch)

    def _extract_mae_from_tuple_or_scalar(v):
        # v could be MAE or (MAE, loss) or (MAE,) etc.
        if isinstance(v, tuple):
            return float(v[0])
        return float(v)

    def _save_mae_csv(epoch_idx, avg_mae, per_set_dict):
        if rank != 0:
            return
        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, "per_set_mae_log.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["epoch", "avg_mae"]
                if isinstance(per_set_dict, dict):
                    header += list(per_set_dict.keys())
                writer.writerow(header)
            row = [epoch_idx + 1, f"{avg_mae:.6f}" if avg_mae is not None else ""]
            if isinstance(per_set_dict, dict):
                for name in per_set_dict.keys():
                    row.append(f"{per_set_dict[name]:.6f}")
            writer.writerow(row)

    def _run_eval():
        """Returns a dict: {'loss': float, 'iou': float|None, 'mae': float|None, 'per_set': dict|None, 'avg_mae': float|None}"""
        kwargs = dict(
            criterion=criterion, mask_prompt=has_mask_prompt,
            include_backgrounds=include_backgrounds, SAM_text=SAM_text, points_from_gt=points_from_gt_eval,
            clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim, parallel_sim=parallel_sim, 
            text_vis=text_vis, clip_decoder_head=clip_decoder_head, clip_decoder=clip_decoder,
            out_size_512=out_size_512, logit_scale=logit_scale, 
            use_soft_prompts=use_soft_prompts, soft_prompt_sigmoid=soft_prompt_sigmoid
        )
        if use_mae:
            ret = eval_model_DDP_per_class_IoU(
                clip_model, trainable_sam, dataloader_val, classes,
                return_IoU=False, return_MAE=True, **kwargs
            )
            if isinstance(dataloader_val, dict):
                # ret: (per_set, avg_mae, [avg_loss])
                per_set_ret = ret[0]
                avg_mae     = ret[1] if len(ret) > 1 else None
                avg_loss    = ret[2] if len(ret) > 2 else None

                # Normalize per-set to name->MAE scalar
                per_set_mae = {}
                for name, val in per_set_ret.items():
                    per_set_mae[name] = _extract_mae_from_tuple_or_scalar(val)

                return {'loss': avg_loss, 'iou': None, 'mae': avg_mae, 'per_set': per_set_mae, 'avg_mae': avg_mae}
            else:
                # single loader: ret could be (MAE, loss) or MAE or loss
                if isinstance(ret, tuple):
                    if len(ret) == 2:
                        avg_mae, avg_loss = ret
                    else:
                        avg_mae, avg_loss = ret[0], ret[-1]
                else:
                    avg_mae, avg_loss = ret, None
                return {'loss': avg_loss, 'iou': None, 'mae': float(avg_mae), 'per_set': None, 'avg_mae': float(avg_mae)}
        else:
            ret = eval_model_DDP_per_class_IoU(
                clip_model, trainable_sam, dataloader_val, classes,
                return_IoU=True, return_MAE=False, **kwargs
            )
            if isinstance(dataloader_val, dict):
                per_set, avg_iou = ret[0], (ret[1] if len(ret) > 1 else None)
                avg_loss = ret[2] if len(ret) > 2 else None
                return {'loss': avg_loss, 'iou': avg_iou, 'mae': None, 'per_set': per_set, 'avg_mae': None}
            else:
                if isinstance(ret, tuple):
                    if len(ret) == 2:
                        avg_iou, avg_loss = ret
                    else:
                        avg_iou, avg_loss = ret[0], ret[-1]
                else:
                    avg_iou, avg_loss = ret, None
                return {'loss': avg_loss, 'iou': avg_iou, 'mae': None, 'per_set': None, 'avg_mae': None}

    # ---------- resume ----------
    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location='cpu')
        except Exception as e:
            print("Checkpoint corrupted:", e)
        trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
        if finetune_clip and 'clip_model' in checkpoint and checkpoint['clip_model'] is not None:
            clip_model.module.load_state_dict(checkpoint['clip_model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0: logger.info('found optimizer checkpoint, loading.')
        except KeyError:
            pass
        if 'scheduler_state_dict' in checkpoint and use_scheduler:
            if rank==0: logger.info('found scheduler checkpoint, loading.')
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if clip_decoder_head and 'clip_decoder' in checkpoint and checkpoint['clip_decoder'] is not None:
            clip_decoder.module.load_state_dict(checkpoint['clip_decoder'])
        if logit_scale is not None and 'logit_scale' in checkpoint and checkpoint['logit_scale'] is not None:
            logit_scale.data = checkpoint['logit_scale']
            if rank == 0:
                logger.info(f"Loaded logit_scale: {logit_scale.exp().item():.4f}")

        # OPTIONAL: restore per-set bests if present
        if 'best_mae_per_set' in checkpoint and isinstance(checkpoint['best_mae_per_set'], dict):
            best_mae_per_set = {k: float(v) for k, v in checkpoint['best_mae_per_set'].items()}

        start_epoch = checkpoint['epoch'] + 1
        if start_epoch + epochs > total_epochs:
            raise ValueError(
                f"Training would exceed total_epochs={total_epochs}: "
                f"start_epoch={start_epoch}, requested epochs={epochs}."
            )

        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        best_MAE = checkpoint.get('best_MAE', best_MAE)
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        del checkpoint
        gc.collect()

    else:
        if rank == 0:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")
        best_IoU, best_val, best_MAE = 0.1, 1e9, 1e9
        if not skip_init_eval:
            _set_epoch_on_val_loader(0)
            init_res = _run_eval()
            if use_mae:
                init_mae = init_res['mae'] if init_res['mae'] is not None else float('inf')
                best_MAE = init_mae
                best_val = init_res['loss'] if init_res['loss'] is not None else best_val

                # NEW: print per-set MAEs at initial eval (if multi-val)
                if isinstance(dataloader_val, dict) and isinstance(init_res['per_set'], dict):
                    # initialize per-set bests
                    for name, m in init_res['per_set'].items():
                        best_mae_per_set[name] = m
                    if rank == 0:
                        parts = [f"{k}: {v:.4f}" for k, v in init_res['per_set'].items()]
                        logger.info("Initial Eval — Per-set MAEs: " + " | ".join(parts))

                if rank == 0:
                    logger.info(f"Initial Eval -> MAE: {init_mae:.4f}, Val Loss: {best_val if best_val<1e9 else float('nan'):.4f}")
            else:
                init_iou = init_res['iou'] if init_res['iou'] is not None else 0.0
                best_IoU = init_iou
                best_val = init_res['loss'] if init_res['loss'] is not None else best_val
                if rank == 0:
                    logger.info(f"Initial Eval -> IoU: {init_iou:.4f}, Val Loss: {best_val if best_val<1e9 else float('nan'):.4f}")
    
    end_epochs = start_epoch + epochs
    if end_epochs > total_epochs:
        raise ValueError(
            f"Training would exceed total_epochs={total_epochs}: "
            f"start_epoch={start_epoch}, requested epochs={epochs}."
        )

    for epoch in range(start_epoch, end_epochs):
        # ---- epoch seed for DDP samplers ----
        if hasattr(dataloader_labeled, "sampler") and hasattr(dataloader_labeled.sampler, "set_epoch"):
            dataloader_labeled.sampler.set_epoch(epoch)

        logger.info(f"\nEpoch {epoch+1}/{end_epochs} - Training")
        progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}") if rank == 0 else dataloader_labeled

        total_loss = 0.0
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # === Supervised step ===
            try:
                sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                    batch, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                    has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=points_from_gt_train, prompt_ensemble=prompt_ensemble,
                    sim_func_surgery=sim_func_surgery, norm_dim=norm_dim, parallel_sim=parallel_sim, text_vis=text_vis,
                    clip_decoder_head=clip_decoder_head, clip_decoder=clip_decoder, logit_scale=logit_scale, 
                    use_soft_prompts=use_soft_prompts, soft_prompt_sigmoid=soft_prompt_sigmoid
                )
            except UnboundLocalError:
                print('error: no points')
                continue

            if out_size_512:
                sam_output = torch.nn.functional.interpolate(sam_output, size=(512, 512), mode="bilinear", align_corners=False)

            loss = criterion(sam_output, label_masks)
            loss.backward()

            # if rank == 0 and batch_idx == 0:
            #     clip_grad_params = 0
            #     clip_nonzero_grad_params = 0
            #     clip_grad_norm = 0.0

            #     for name, p in clip_model.module.named_parameters():
            #         if p.requires_grad:
            #             if p.grad is not None:
            #                 clip_grad_params += 1
            #                 g = p.grad.detach()
            #                 gsum = g.abs().sum().item()
            #                 if gsum > 0:
            #                     clip_nonzero_grad_params += 1
            #                     clip_grad_norm += g.norm().item()
            #                     print(f"[CLIP GRAD] {name}: mean={g.abs().mean().item():.6e}, norm={g.norm().item():.6e}")

            #     print(f"[CLIP] params with grad tensor: {clip_grad_params}")
            #     print(f"[CLIP] params with nonzero grad: {clip_nonzero_grad_params}")
            #     print(f"[CLIP] total grad norm sum: {clip_grad_norm:.6e}")


            total_loss += loss.item()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            
            if rank == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(loss=avg_loss)

        # ---------- Validation ----------
        _set_epoch_on_val_loader(epoch)

        do_val = (epoch >= 0) #or (epoch % 5 == 0 and epoch >= 1)
        if do_val:
            res = _run_eval()
            val_loss = res['loss'] if res['loss'] is not None else float('nan')
            IoU     = res['iou']
            avg_MAE = res['avg_mae']
            per_set = res['per_set']  # None or dict(name->MAE)

            # track best general val loss
            if not math.isnan(val_loss):
                best_val = min(best_val, val_loss)

            # ===== MAE-specific handling =====
            if use_mae:
                # Save CSV of per-set MAEs each eval
                if isinstance(per_set, dict):
                    _save_mae_csv(epoch, avg_MAE, per_set)

                # Print a neat summary of all per-set MAEs
                if rank == 0 and isinstance(per_set, dict) and len(per_set) > 0:
                    parts = [f"{k}: {v:.4f}" for k, v in per_set.items()]
                    logger.info(f"Per-set MAEs @ epoch {epoch+1}: " + " | ".join(parts))

                # Global best MAE across sets (avg)
                if avg_MAE is not None and avg_MAE < best_MAE:
                    best_MAE = avg_MAE
                    if rank == 0:
                        ckpt_path = os.path.join(save_path, f"supervised_best_MAE_{save_ext}.pth")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': trainable_sam.module.state_dict(),
                            'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                            'clip_decoder': clip_decoder.module.state_dict() if clip_decoder_head else None,
                            'logit_scale': logit_scale.data if logit_scale is not None else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                            'val_loss': val_loss,
                            'best_val': best_val,
                            'best_IoU': best_IoU,
                            'best_MAE': best_MAE,
                            'best_mae_per_set': best_mae_per_set  # persist set-wise bests too
                        }, ckpt_path)
                        logger.info(f"Best avg MAE {best_MAE:.4f} - Saved checkpoint to {ckpt_path}")

                # Per-set best MAE tracking & W&B logging
                if isinstance(per_set, dict):
                    for name, cur_mae in per_set.items():
                        prev_best = best_mae_per_set.get(name, float('inf'))
                        # Log current MAE
                        if rank == 0 and not ignore_wandb:
                            wandb.log({f"{name}/MAE": cur_mae, "epoch": epoch + 1})
                        # Update & log best if improved
                        if cur_mae < prev_best:
                            best_mae_per_set[name] = cur_mae
                            if rank == 0 and not ignore_wandb:
                                wandb.log({f"{name}/Best_MAE": cur_mae, "epoch": epoch + 1})
                            if rank == 0:
                                logger.info(f"New best MAE for {name}: {cur_mae:.4f}")

            # ===== IoU path unchanged =====
            else:
                if IoU is not None and IoU > best_IoU:
                    best_IoU = IoU
                    if rank == 0:
                        ckpt_path = os.path.join(save_path, f"supervised_best_IoU_{save_ext}.pth")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': trainable_sam.module.state_dict(),
                            'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                            'clip_decoder': clip_decoder.module.state_dict() if clip_decoder_head else None,
                            'logit_scale': logit_scale.data if logit_scale is not None else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                            'val_loss': val_loss,
                            'best_val': best_val,
                            'best_IoU': best_IoU,
                            'best_MAE': best_MAE,
                            'best_mae_per_set': best_mae_per_set
                        }, ckpt_path)
                        logger.info(f"Best IoU {best_IoU:.4f} - Saved checkpoint to {ckpt_path}")

            # ----- WandB logging (global) -----
            if rank == 0 and not ignore_wandb:
                train_loss = total_loss / len(dataloader_labeled)
                lr_logs = {f"lr/group_{i}": pg['lr'] for i, pg in enumerate(optimizer.param_groups)}
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "Best_val": best_val,
                    "Best_IoU": best_IoU,
                    "Best_MAE": best_MAE
                }
                if IoU is not None: log_dict["IoU"] = IoU
                if avg_MAE is not None: log_dict["MAE"] = avg_MAE
                wandb.log({**log_dict, **lr_logs})
                if use_mae:
                    logger.info(f"MAE: {avg_MAE:.4f}, Best MAE: {best_MAE:.4f}")
                else:
                    logger.info(f"IoU: {IoU:.4f}, Best IoU: {best_IoU:.4f}")

            # ----- periodic checkpoint -----
            # if rank == 0:
            #     ckpt_path = os.path.join(save_path, f"supervised_sam_saved_{save_ext}.pth")
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': trainable_sam.module.state_dict(),
            #         'clip_model': clip_model.module.state_dict() if finetune_clip else None,
            #         'clip_decoder': clip_decoder.module.state_dict() if clip_decoder_head else None,
            #         'logit_scale': logit_scale.data if logit_scale is not None else None,
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
            #         'val_loss': val_loss,
            #         'best_val': best_val,
            #         'best_IoU': best_IoU,
            #         'best_MAE': best_MAE,
            #         'best_mae_per_set': best_mae_per_set
            #     }, ckpt_path)

    # ----- final checkpoint -----
    # if rank == 0:
    #     ckpt_path = os.path.join(save_path, f"supervised_sam_epoch_{epoch+1}{save_ext}.pth")
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': trainable_sam.module.state_dict(),
    #         'clip_model': clip_model.module.state_dict() if finetune_clip else None,
    #         'clip_decoder': clip_decoder.module.state_dict() if clip_decoder_head else None,
    #         'logit_scale': logit_scale.data if logit_scale is not None else None,
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
    #         'val_loss': locals().get('val_loss', float('nan')),
    #         'best_val': best_val,
    #         'best_IoU': best_IoU,
    #         'best_MAE': best_MAE,
    #         'best_mae_per_set': best_mae_per_set
    #     }, ckpt_path)
    # return trainable_sam

def debug_check_clip_gradients(clip_model, batch, criterion, clip_crop_size=224, prompt_ensemble=False, norm_dim=1):
    """
    Debug function to print gradients of CLIP parameters after backward.
    """

    # Set to train mode and zero gradients
    clip_model.train()
    clip_model.zero_grad()

    # Forward Pass
    clip_output, gt_masks = forward_supervised_clip(batch, clip_model, clip_crop_size, norm_dim, prompt_ensemble)

    # Compute Loss
    loss = criterion(clip_output, gt_masks)
    print(f"Loss: {loss.item()}")

    # Backward Pass
    loss.backward()

    # Check gradients
    print("\n--- Gradient Check ---")
    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"Parameter: {name} | Grad Norm: {grad_norm:.6f}")
            else:
                print(f"Parameter: {name} | NO GRADIENT!")


def train_supervised_clip_DDP(
    clip_model, labeled_loader,
    val_loader,
    classes,
    total_epochs,
    clip_crop_size,
    epochs,
    save_path,
    save_ext,
    logger,
    weight_decay=0.0,
    finetune_clip=False,
    use_scheduler=True,
    lr_clip=1e-6,
    resume_path=None,
    ignore_wandb=False,
    skip_init_eval=False,
    iou_loss=False,
    bce_dice_loss=False,
    structure_loss=False,
    enhanced_structure_loss=False,
    prompt_ensemble=False,
    norm_dim=1,
    final_div_factor=1e2,
    pct_start=0.2,
    div_factor=25,
    clip_decoder_head=False,
    clip_decoder=None,
    lr_clip_dec_head=1e-4,
    out_size_512=False,
    dataset='COCO',
    logit_scale=None):

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # ----- Initialize WandB -----
    if rank == 0 and not ignore_wandb:
        wandb.init(
            project=f"clip-supervised-ddp-{dataset}",
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr_clip,
                "finetune_clip": finetune_clip,
                "clip_decoder_head": clip_decoder_head
            }
        )

    # ----- Collect Trainable Parameters -----
    params = []    
    if finetune_clip:
        clip_params = [param for name, param in clip_model.module.named_parameters() if param.requires_grad]
        params.append({'params': clip_params, 'lr': lr_clip, 'weight_decay': weight_decay})
        logger.info("Finetuning CLIP.")

    params.append({'params': [logit_scale], 'lr': lr_clip})

    if clip_decoder_head:
        clip_dec_h_params = [param for _, param in clip_decoder.module.named_parameters() if param.requires_grad]
        params.append({'params': clip_dec_h_params, 'lr': lr_clip_dec_head, 'weight_decay': weight_decay})
        logger.info("Using CLIP decoder head.")

    print(f"Number of trainable clip params: {sum(p.numel() for p in clip_params)}")

    
    optimizer = optim.AdamW(params)
    # ----- Scheduler -----
    if use_scheduler:
        max_lr = [lr_clip] + ([lr_clip_dec_head] if clip_decoder_head else [])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(labeled_loader),
            epochs=total_epochs,
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )    # ----- Loss Function -----
    criterion = select_loss_function(
        simple_loss=not(iou_loss or structure_loss or enhanced_structure_loss or bce_dice_loss),
        iou_loss=iou_loss,
        bce_dice_loss=bce_dice_loss,
        structure_loss=structure_loss,
        enhanced_structure_loss=enhanced_structure_loss,
        logger=logger,
        rank=rank
    )    # ----- Resume from Checkpoint -----
    scheduler_obj = scheduler if use_scheduler else None

    start_epoch, best_val, best_IoU = load_checkpoint_clip_only(
        clip_model, clip_decoder, optimizer, scheduler_obj,
        resume_path, finetune_clip, clip_decoder_head,
        use_scheduler, logger, rank, logit_scale=logit_scale
    )
     # ----- Optional Initial Evaluation -----
    if not skip_init_eval and start_epoch == 0:

        initial_IoU, initial_val_loss = evaluate_clip_model(
            clip_model, val_loader, classes, criterion, clip_crop_size, 
            prompt_ensemble, norm_dim, out_size_512, logit_scale=logit_scale,
            clip_decoder_head=clip_decoder_head, clip_decoder=clip_decoder
        )
        if rank == 0:
            logger.info(f"Initial Eval -> IoU: {initial_IoU:.4f}, Val Loss: {initial_val_loss:.4f}")
        best_IoU, best_val = initial_IoU, initial_val_loss

    total_batches = len(labeled_loader)
    half_batches = total_batches // 10

    for i, param_group in enumerate(optimizer.param_groups):
        print(f"LR group {i}: {param_group['lr']}")

    # ----- Training Loop -----
    end_epochs = start_epoch + epochs
    for epoch in range(start_epoch, end_epochs):
        labeled_loader.sampler.set_epoch(epoch)

        if rank == 0:
            progress_bar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{end_epochs}")
        else:
            progress_bar = labeled_loader

        total_loss = 0.0
        clip_model.train()

        initial_clip_params = {}
        for name, param in clip_model.named_parameters():
            if param.requires_grad:
                initial_clip_params[name] = param.clone().detach()


        for batch_idx, batch in enumerate(progress_bar):
            # if batch_idx >= half_batches:
            #     break
            optimizer.zero_grad()

            # === Forward Pass ===
            clip_output, gt_masks = forward_supervised_clip(batch, clip_model, clip_crop_size, logit_scale=logit_scale)

            #print(f"Similarity Map Stats -> min: {clip_output.min().item()}, max: {clip_output.max().item()}, mean: {clip_output.mean().item()}")
            #print(torch.mean(clip_output))

            if out_size_512:                
                clip_output = torch.nn.functional.interpolate(clip_output, size=(512, 512), mode="bilinear", align_corners=False)

            loss = criterion(clip_output, gt_masks)
            loss.backward()

            # for name, param in clip_model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f"Grad Norm for {name}: {param.grad.norm().item():.6f}")

            optimizer.step()
            
            # for name, param in clip_model.named_parameters():
            #     if param.requires_grad:
            #         diff = (param.detach() - initial_clip_params[name]).abs().sum()
            #         print(f"Param: {name} | Diff Sum: {diff.item():.6f}")

            if use_scheduler:
                scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            if rank == 0:
                progress_bar.set_postfix(loss=avg_loss)

        if epoch < 50:
            val_loss, best_val, best_IoU = 100, 100, 0.01
        else:
        # ----- Validation -----
            val_IoU, val_loss = evaluate_clip_model(
                clip_model, val_loader, classes, criterion, clip_crop_size, 
                prompt_ensemble, norm_dim, out_size_512, logit_scale=logit_scale,
                clip_decoder_head=clip_decoder_head, clip_decoder=clip_decoder
            )

            # ----- Logging -----
            if rank == 0:
                # ----- Save Best Model -----
                if val_IoU > best_IoU and rank == 0:
                    best_IoU = val_IoU
                    save_checkpoint_clip_only(
                        epoch, clip_model, clip_decoder, optimizer, scheduler if use_scheduler else None, val_loss, best_val, best_IoU,
                        save_path, save_ext, finetune_clip, clip_decoder_head, use_scheduler, logger, best=True, logit_scale=logit_scale
                    )
                if not ignore_wandb:
                    lr_logs = {f"lr/group_{i}": param_group['lr'] for i, param_group in enumerate(optimizer.param_groups)}
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "val_loss": val_loss,
                        "IoU": val_IoU,
                        "Best_val": best_val,
                        "Best_IoU": best_IoU,
                        **lr_logs
                    })

                logger.info(f"Epoch {epoch+1}/{end_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val:.4f} | IoU: {val_IoU:.4f} | Best IoU: {best_IoU:.4f}")
            
        # ----- Periodic Checkpoint -----
        if rank == 0:
            save_checkpoint_clip_only(
                epoch, clip_model, clip_decoder, optimizer, scheduler if use_scheduler else None, val_loss, best_val, best_IoU,
                save_path, save_ext, finetune_clip, clip_decoder_head, use_scheduler, logger, best=False, logit_scale=logit_scale
            )
    # Final Save after Training
    if rank == 0:
        save_checkpoint_clip_only(
            epoch, clip_model, clip_decoder, optimizer, scheduler if use_scheduler else None, val_loss, best_val, best_IoU,
            save_path, save_ext, finetune_clip, clip_decoder_head, use_scheduler, logger, final=True, logit_scale=logit_scale
        )



# isn't for scripts (no logging or wandb), and isn't for mutli-gpu (dataparallel)
def train_combined(
    frozen_sam, trainable_sam, clip_model,
    dataloader_labeled, dataloader_unlabeled, dataloader_val,
    classes, device_trainable, device_frozen, 
    epochs, save_path, save_ext, unsupervised_per_supervised = 64,
    SAM_text=False, has_mask_prompt=False, include_backgrounds=False,
    features_input=False, is_vanilla_sam=False, decay = 0.99, lr=1e-4, a = 0.2):
    optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    #criterion = SAMCLIPSupervisedLoss()
    criterion = nn.BCEWithLogitsLoss()

    clip_model.eval()
    frozen_sam.eval()
    trainable_sam.train()

    
    best_IoU, best_val = eval_model(clip_model, trainable_sam, dataloader_val, device_trainable, classes, 
                                    criterion=criterion, mask_prompt=has_mask_prompt, 
                                    include_backgrounds=False, SAM_text=SAM_text, return_IoU = True,
                                   points_from_gt=False)

    print(f'initial eval: {best_val}, initial IoU: {best_IoU}')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - Starting Training Loop")
        
        progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        total_loss = 0
        
        total_confidence_ratio = 0

        loss_unsupervised_weight = get_dynamic_unsup_weight(epoch,epochs,a=a)
        
        for batch_idx, batch in enumerate(progress_bar):
            
            optimizer.zero_grad()

            sam_supervised_output, clip_image_embeddings, text_features, label_masks, similarity_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(batch,device_trainable, clip_model,trainable_sam,classes, has_mask_prompt = has_mask_prompt, include_backgrounds = False, SAM_text = SAM_text, is_vanilla_sam = is_vanilla_sam, features_input = features_input, points_from_gt=True)

            loss = criterion(sam_supervised_output, label_masks)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()


            for i in range(unsupervised_per_supervised):
                optimizer.zero_grad()
                batch = next(iter(dataloader_unlabeled))

            
                

                frozen_sam_output, clip_image_embeddings, text_features, _, similarity_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU(batch,device_frozen, clip_model,frozen_sam,classes, has_mask_prompt = has_mask_prompt, include_backgrounds = False, SAM_text = SAM_text, is_vanilla_sam = is_vanilla_sam, features_input = features_input, points_from_gt=False)
            
                pseudo_labels, confidence_mask, confidence_ratio = generate_pseudo_labels(frozen_sam_output, threshold=0.9)

                #print(f"Confidence ratio: {confidence_ratio.item() * 100:.2f}% pixels used")

                total_confidence_ratio += confidence_ratio.item()

                
                # pseudo_labels = get_binary_output(frozen_sam_output, out_tensor=True).to(device_trainable)
    
                inputs_trainable = [inp.to(device_trainable) for inp in inputs]
                
                sam_output = trainable_sam(inputs_trainable)
                
                loss_unsupervised = loss_unsupervised_weight * masked_loss(sam_output, pseudo_labels, confidence_mask)                
                
                # Train with pseudo-labels
                # loss_unsupervised = loss_unsupervised_weight * criterion(
                #     sam_output, clip_image_embeddings,
                #     text_features, pseudo_labels, None)
                
                loss_unsupervised.backward()
                optimizer.step()
                
                total_loss += loss_unsupervised.item()
                
    
                frozen_sam = update_ema_frozen(frozen_sam,trainable_sam,decay=decay)

            avg_loss = total_loss / ((batch_idx + 1)*(unsupervised_per_supervised + 1))

            avg_conf_ratio = total_confidence_ratio / ((batch_idx+1)*unsupervised_per_supervised)
    
            progress_bar.set_postfix(loss=avg_loss, conf_ratio=f"{avg_conf_ratio * 100:.2f}%")

            # if batch_idx%200==0:
            #     IoU, val = eval_model(clip_model, trainable_sam, dataloader_val, device_trainable, classes, 
            #                         criterion=criterion, mask_prompt=has_mask_prompt, 
            #                         include_backgrounds=False, SAM_text=SAM_text, return_IoU = True,
            #                        points_from_gt=False)
                
            #     print(f'batch: {batch_idx}, val: {val}, IoU: {IoU}')
           

        IoU, val_loss = eval_model(clip_model, trainable_sam, dataloader_val, device_trainable, classes, 
                                    criterion=criterion, mask_prompt=has_mask_prompt, 
                                    include_backgrounds=False, SAM_text=SAM_text, return_IoU = True,
                                   points_from_gt=False)
            
        if val_loss<best_val:
            best_val = val_loss
            
        if IoU > best_IoU:
            best_IoU = IoU
            print(f"best val_loss: {val_loss}")
            #clip_save_path = os.path.join(save_path,"fine_tuned_clip_joint_best"+save_ext+".pth")
            sam_save_path = os.path.join(save_path,"fine_tuned_semisupervised_sam_best_IoU_"+save_ext+".pth")
            #torch.save(clip_model.state_dict(), clip_save_path)
            torch.save(trainable_sam.state_dict(), sam_save_path)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/((unsupervised_per_supervised+1)*len(dataloader_labeled))}, Val_Loss: {val_loss}, Best_Val: {best_val}")
        print(f"IoU: {IoU}, Best_IoU: {best_IoU}")
        
    return trainable_sam


def train_combined_final_for_scripts(
    frozen_sam, trainable_sam, clip_model,
    dataloader_labeled, dataloader_unlabeled, dataloader_val,
    classes,
    epochs, save_path, save_ext, logger, split,
    unsupervised_per_supervised=64, SAM_text=False, has_mask_prompt=False,
    include_backgrounds=False, features_input=False, is_vanilla_sam=False,
    decay=0.99, lr=1e-4, a=0.2, resume_path=None, multi_gpu = True, ignore_wandb = False, skip_init_eval=False
):
    # Enable multi-GPU
    if multi_gpu:
        trainable_sam = nn.DataParallel(trainable_sam)
        clip_model = nn.DataParallel(clip_model)
        frozen_sam = nn.DataParallel(frozen_sam)
        logger.info("Using Parallel/MultiGPU")
    else:
        logger.info("Using Single GPU or CPU")

    if not ignore_wandb:
        wandb.init(
            project="sam-clip-semisupervised-"+split,
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr,
                "sam_type": "vanilla" if is_vanilla_sam else "parallel",
                "SAM_text": SAM_text
            }
        )

    optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    clip_model.eval()
    frozen_sam.eval()
    trainable_sam.train()

    
    start_epoch = 0

    # Resume support
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        if multi_gpu:
            trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
            frozen_sam.module.load_state_dict(checkpoint['frozen_sam_state_dict'])
        else:
            trainable_sam.load_state_dict(checkpoint['model_state_dict'])
            frozen_sam.load_state_dict(checkpoint['frozen_sam_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
    
    else:
        logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")
        best_IoU, best_val = 0.1, 10000
        if not skip_init_eval:
            best_IoU, best_val = eval_model(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False
            )
        logger.info(f"Initial Eval -> IoU: {best_IoU:.4f}, Val Loss: {best_val:.4f}")
    
    end_epochs = start_epoch + epochs

    for epoch in range(start_epoch, end_epochs):
        logger.info(f"\nEpoch {epoch+1}/{end_epochs} - Training")
        progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        total_loss = 0
        total_confidence_ratio = 0
        exception_counter = 0

        loss_unsupervised_weight = get_dynamic_unsup_weight(epoch, epochs, a=a) if epoch>40 else a

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_size = dataloader_labeled.batch_size
            # try:
            # Supervised step
            sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                batch, clip_model, trainable_sam, classes,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=True
            )

            loss = criterion(sam_output, label_masks)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            # Unsupervised steps
            for _ in range(unsupervised_per_supervised):
                optimizer.zero_grad()
                # try:
                batch_unlabeled = next(iter(dataloader_unlabeled))
                frozen_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU(
                    batch_unlabeled, clip_model, frozen_sam, classes,
                    has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=False
                )

                pseudo_labels, conf_mask, conf_ratio = generate_pseudo_labels(frozen_output, threshold=0.9)
                total_confidence_ratio += conf_ratio.mean().item()

                train_output = trainable_sam(inputs)

                
                unsup_loss = loss_unsupervised_weight * masked_loss(train_output, pseudo_labels, conf_mask)
                unsup_loss.backward()
                total_loss += unsup_loss.item()
                optimizer.step()

                frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay)
                # except Exception as e:
                #     print(f'exception occured: {e}')
                #     exception_counter += 1
                #     continue

            avg_loss = total_loss / ((batch_idx + 1) * (unsupervised_per_supervised + 1))
            avg_conf_ratio = total_confidence_ratio / ((batch_idx + 1) * unsupervised_per_supervised)
            progress_bar.set_postfix(loss=avg_loss, conf_ratio=f"{avg_conf_ratio * 100:.2f}%")

            # except Exception:
            #     exception_counter += 1
            #     continue

        # Validation after epoch
        IoU, val_loss = eval_model(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion, mask_prompt=has_mask_prompt,
            include_backgrounds=include_backgrounds, SAM_text=SAM_text,
            return_IoU=True, points_from_gt=False
        )

        if not ignore_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": total_loss / (len(dataloader_labeled) * (unsupervised_per_supervised + 1)),
                "val_loss": val_loss,
                "IoU": IoU,
                "confidence_ratio": avg_conf_ratio,
                "Best_IoU": best_IoU
            })

        # Save best IoU model
        if IoU > best_IoU:
            best_IoU = IoU
            ckpt_path = os.path.join(save_path, f"semisup_sam_best_IoU_{save_ext}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_sam.module.state_dict() if multi_gpu else trainable_sam.state_dict(),
                'frozen_sam_state_dict': frozen_sam.module.state_dict() if multi_gpu else frozen_sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val': best_val,
                'best_IoU': best_IoU
            }, ckpt_path)
            logger.info(f"Best IoU {best_IoU:.4f} - Saved checkpoint to {ckpt_path}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_sam.module.state_dict() if multi_gpu else trainable_sam.state_dict(),
                'frozen_sam_state_dict': frozen_sam.module.state_dict() if multi_gpu else frozen_sam.state_dict(),
                'val_loss': val_loss,
                'best_val': best_val,
                'best_IoU': best_IoU
            }, ckpt_path)


    ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': trainable_sam.module.state_dict() if multi_gpu else trainable_sam.state_dict(),
        'frozen_sam_state_dict': frozen_sam.module.state_dict() if multi_gpu else frozen_sam.state_dict(),
        'val_loss': val_loss,
        'best_val': best_val,
        'best_IoU': best_IoU
    }, ckpt_path)
    return trainable_sam

    
def train_combined_final_for_scripts_DDP(
    frozen_sam, trainable_sam, clip_model,
    dataloader_labeled, dataloader_unlabeled, dataloader_val,
    classes, total_epochs, epochs, save_path, save_ext, logger, split,
    unsupervised_per_supervised=64, SAM_text=False, has_mask_prompt=False, weight_decay=0.0,
    include_backgrounds=False, features_input=False, is_vanilla_sam=False, finetune_clip=False,
    use_scheduler=True, lr_IE=1e-4, lr_MD=5e-5, lr_clip=1e-6, lr_PE=1e-5, clip_crop_size=224,
    decay_sup_start=0.95, decay_unsup_start=0.999, a=0.2, resume_path=None, ignore_wandb = False, skip_init_eval=False,
    conf_thresh=0.9, simple_loss=True, iou_loss=False, prompt_ensemble=False, sim_func_surgery=False, 
    norm_dim=1, pct_start=0.2, final_div_factor=1e2, options=None
    ):
    # Enable multi-GPU
    # if multi_gpu:
    #     logger.info("Running in DDP mode. Model already wrapped in DistributedDataParallel.")
    # else:
    #     logger.info("Using Single GPU or CPU")

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0 and not ignore_wandb:
        wandb.init(
            project="sam-clip-semisupervised-ddp-"+split,
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr_IE,
                "sam_type": "vanilla" if is_vanilla_sam else "parallel",
                "SAM_text": SAM_text
            }
        )

    if rank==0: 
        logger.info(options)

    adapter_params = [
    param for name, param in trainable_sam.module.image_encoder.named_parameters()
    if any(block in name for block in ['Space_Adapter', 'MLP_Adapter', 'Depth_Adapter']) and param.requires_grad
    ]

    # Get trainable mask decoder parameters
    mask_decoder_params = [
        param for name, param in trainable_sam.module.mask_decoder.named_parameters()
        if param.requires_grad
    ]

    if has_mask_prompt:
        prompt_encoder_params = [
            param for name,param in trainable_sam.module.prompt_encoder.mask_downscaling.named_parameters()
            if param.requires_grad
        ]

    if finetune_clip:
        clip_model_params = [
            param for name, param in clip_model.module.named_parameters()
            if param.requires_grad
        ]
    optimizer = optim.AdamW(
    [
        {'params': adapter_params, 'lr': lr_IE, 'weight_decay': weight_decay},
        {'params': mask_decoder_params, 'lr': lr_MD, 'weight_decay': weight_decay},
        *(
            [{'params': clip_model_params, 'lr': lr_clip, 'weight_decay': 0.0}]
            if finetune_clip else []
        ),
        *(
            [{'params': prompt_encoder_params, 'lr':lr_PE, 'weight_decay':weight_decay}]
            if has_mask_prompt else []
        )
    ]
    )

    if use_scheduler:
        max_lr = [lr_IE, lr_MD] + ([lr_clip] if finetune_clip else []) + ([lr_PE] if has_mask_prompt else [])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader_labeled)*(unsupervised_per_supervised+1),
        epochs=total_epochs,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=5e2
    )
        
    # if use_scheduler:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_epochs,
    #     eta_min=1e-6
    #     )

    if simple_loss:
        if rank==0: 
            logger.info('using simple BCE')
        criterion_sup = nn.BCEWithLogitsLoss()
    else:
        if iou_loss:
            bce_weight = 0.2
            dice_weight = 1.5
            iou_weight = 0.8
            criterion_sup = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, iou_weight=iou_weight)
            iou_f = IoULoss(eps=1e-6)
            if rank==0:
                logger.info('using BCE, Dice, and IoU loss with weights:',bce_weight,dice_weight,iou_weight)

        else:       
            bce_weight = 0.5
            dice_weight = 1.5
            if rank==0:
                logger.info('using BCE and Dice, with weights:',bce_weight,dice_weight)
            criterion_sup = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    #optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    #criterion = nn.BCEWithLogitsLoss()
    if finetune_clip:
        clip_model.train()
    else:
        clip_model.eval()

    trainable_sam.train()
    #clip_model.eval()
    frozen_sam.eval()

    
    start_epoch = 0

    best_IoU, best_val = 0.01, 10000

    # Resume support
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
        frozen_sam.load_state_dict(checkpoint['frozen_sam_state_dict'])
        if finetune_clip and 'clip_model' in checkpoint:
            clip_model.module.load_state_dict(checkpoint['clip_model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0:
                logger.info('found optimizier checkpoint, loading.')
        except KeyError:
            pass
        if 'scheduler_state_dict' in checkpoint:
            if rank==0:
                logger.info('found scheduler checkpoint, loading.')
            if use_scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if start_epoch + epochs > total_epochs:
            raise ValueError(
                f"Training would exceed total_epochs={total_epochs}: "
                f"start_epoch={start_epoch}, requested epochs={epochs}."
            )
        
        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()

    
    else:
        if rank == 0:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")
        best_IoU, best_val = 0.1, 10000
        if not skip_init_eval:
            dataloader_val.sampler.set_epoch(0)
            best_IoU, best_val = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )
        
        if rank == 0:
            logger.info(f"Initial Eval -> IoU: {best_IoU:.4f}, Val Loss: {best_val:.4f}")
    
    end_epochs = start_epoch + epochs
    if end_epochs > total_epochs:
        raise ValueError(
            f"Training would exceed total_epochs={total_epochs}: "
            f"start_epoch={start_epoch}, requested epochs={epochs}."
        )

    for epoch in range(start_epoch, end_epochs):
        dataloader_labeled.sampler.set_epoch(epoch)
        dataloader_unlabeled.sampler.set_epoch(epoch)
        

        logger.info(f"\nEpoch {epoch+1}/{end_epochs} - Training")
        if rank == 0:
            progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        else:
            progress_bar = dataloader_labeled  # Other ranks don't use tqdm

        #progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        total_loss = 0
        total_confidence_ratio = 0
        #exception_counter = 0
        

        loss_unsupervised_weight = cosine_rampup(current_epoch=epoch, rampup_epochs=10, max_value=a)
        confidence_threshold = cosine_threshold_scheduler(current_epoch=epoch, warmup_epochs=10, start=0.7, end=conf_thresh)
        
        unlabeled_iter = iter(dataloader_unlabeled)

        decay_sup = ema_decay_schedule(epoch, total_epochs, start=decay_sup_start, end=0.9999)
        decay_unsup = ema_decay_schedule(epoch, total_epochs, start=decay_unsup_start, end=0.99999)

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # === Supervised step ===
            sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                batch, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

            loss = criterion_sup(sam_output, label_masks)
            loss.backward()
            total_loss += loss.item()
            # if finetune_clip:
            #     similarity_loss = clip_sim_loss_f(sim_prompt.squeeze(1), label_masks.squeeze(1))
            #     contrastive_loss = clip_contrastive_loss(clip_emb,text_feat)
            #     clip_loss = similarity_loss + contrastive_loss
            #     clip_loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_sup)

            # === Unsupervised steps ===
            if rank == 0:
                inner_bar = tqdm(range(unsupervised_per_supervised), leave=False, desc="Unsupervised steps", dynamic_ncols=True)
            else:
                inner_bar = range(unsupervised_per_supervised)

            for _ in inner_bar:
                optimizer.zero_grad()

                try:
                    batch_unlabeled = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(dataloader_unlabeled)
                    batch_unlabeled = next(unlabeled_iter)

                frozen_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU(
                    batch_unlabeled, clip_model, frozen_sam, classes, clip_crop_size=clip_crop_size,
                    has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                    sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
                )

                pseudo_labels, conf_mask, conf_ratio = generate_pseudo_labels(frozen_output, threshold=confidence_threshold)
                total_confidence_ratio += conf_ratio.mean().item()

                train_output = trainable_sam(inputs)

                msk_loss = masked_loss(train_output, pseudo_labels, conf_mask)
                dice_l = dice_loss(train_output,pseudo_labels)
                if iou_loss:
                    iou_l = iou_f(train_output,pseudo_labels)
                    unsup_loss = loss_unsupervised_weight * (msk_loss + dice_l + iou_l)
                else:
                    unsup_loss = loss_unsupervised_weight * (msk_loss + dice_l)
                unsup_loss.backward()
                total_loss += unsup_loss.item()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()


                frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_unsup)

            avg_loss = total_loss / ((batch_idx + 1) * (unsupervised_per_supervised + 1))
            avg_conf_ratio = total_confidence_ratio / ((batch_idx + 1) * unsupervised_per_supervised)
            if rank == 0:
                progress_bar.set_postfix(loss=avg_loss, conf_ratio=f"{avg_conf_ratio * 100:.2f}%")


        # Validation after epoch
        dataloader_val.sampler.set_epoch(epoch) 
        IoU, val_loss = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

        if rank == 0 and not ignore_wandb:
            train_loss = total_loss / (len(dataloader_labeled) * (unsupervised_per_supervised + 1))
            lr_logs = {
                f"lr/group_{i}": param_group['lr']
                for i, param_group in enumerate(optimizer.param_groups)
            }

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "IoU": IoU,
                "confidence_ratio": avg_conf_ratio,
                "Best_val": best_val,
                "Best_IoU": best_IoU,
                **lr_logs
            })
            
            logger.info(f"Epoch {epoch+1}/{end_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val:.4f} | IoU: {IoU:.4f} | Best IoU: {best_IoU:.4f}")



        # Save best IoU model
        if IoU > best_IoU:
            best_IoU = IoU
            if rank == 0:
                ckpt_path = os.path.join(save_path, f"semisup_sam_best_IoU_{save_ext}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_sam.module.state_dict(),
                    'frozen_sam_state_dict': frozen_sam.state_dict(),
                    'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                    'val_loss': val_loss,
                    'best_val': best_val,
                    'best_IoU': best_IoU
                }, ckpt_path)
                logger.info(f"Best IoU {best_IoU:.4f} - Saved checkpoint to {ckpt_path}")

        # # Periodic checkpoint
        # if (epoch + 1) % 10 == 0 and rank == 0:
        #     ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': trainable_sam.module.state_dict(),
        #         'frozen_sam_state_dict': frozen_sam.state_dict(),
        #         'val_loss': val_loss,
        #         'best_val': best_val,
        #         'best_IoU': best_IoU
        #     }, ckpt_path)
   
    if rank == 0:
        ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainable_sam.module.state_dict(),
            'frozen_sam_state_dict': frozen_sam.state_dict(),
            'clip_model': clip_model.module.state_dict() if finetune_clip else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
            'val_loss': val_loss,
            'best_val': best_val,
            'best_IoU': best_IoU
        }, ckpt_path)
    return trainable_sam


def train_combined_eqsup_final_for_scripts_DDP(
    frozen_sam, trainable_sam, clip_model,
    dataloader_labeled, dataloader_unlabeled, dataloader_val,
    classes, total_epochs, epochs, save_path, save_ext, logger, split,
    unsupervised_per_supervised=64, SAM_text=False, has_mask_prompt=False, weight_decay=0.0,
    include_backgrounds=False, features_input=False, is_vanilla_sam=False, finetune_clip=False,
    use_scheduler=True, lr_IE=1e-4, lr_MD=5e-5, lr_clip=1e-6, lr_PE=1e-5, clip_crop_size=224,
    decay_sup_start=0.95, decay_unsup_start=0.999, a=0.2, resume_path=None, ignore_wandb = False, skip_init_eval=False,
    conf_thresh=0.9, simple_loss=True, iou_loss=False, prompt_ensemble=False, sim_func_surgery=False, 
    norm_dim=1, pct_start=0.2, final_div_factor=1e2, options=None
    ):
    # Enable multi-GPU
    # if multi_gpu:
    #     logger.info("Running in DDP mode. Model already wrapped in DistributedDataParallel.")
    # else:
    #     logger.info("Using Single GPU or CPU")

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0 and not ignore_wandb:
        wandb.init(
            project="sam-clip-semisupervised-ddp-"+split,
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr_IE,
                "sam_type": "vanilla" if is_vanilla_sam else "parallel",
                "SAM_text": SAM_text
            }
        )

    if rank==0: 
        logger.info(options)

    adapter_params = [
    param for name, param in trainable_sam.module.image_encoder.named_parameters()
    if any(block in name for block in ['Space_Adapter', 'MLP_Adapter', 'Depth_Adapter']) and param.requires_grad
    ]

    # Get trainable mask decoder parameters
    mask_decoder_params = [
        param for name, param in trainable_sam.module.mask_decoder.named_parameters()
        if param.requires_grad
    ]

    if has_mask_prompt:
        prompt_encoder_params = [
            param for name,param in trainable_sam.module.prompt_encoder.mask_downscaling.named_parameters()
            if param.requires_grad
        ]

    if finetune_clip:
        clip_model_params = [
            param for name, param in clip_model.module.named_parameters()
            if param.requires_grad
        ]
    optimizer = optim.AdamW(
    [
        {'params': adapter_params, 'lr': lr_IE, 'weight_decay': weight_decay},
        {'params': mask_decoder_params, 'lr': lr_MD, 'weight_decay': weight_decay},
        *(
            [{'params': clip_model_params, 'lr': lr_clip, 'weight_decay': 0.0}]
            if finetune_clip else []
        ),
        *(
            [{'params': prompt_encoder_params, 'lr':lr_PE, 'weight_decay':weight_decay}]
            if has_mask_prompt else []
        )
    ]
    )

    if use_scheduler:
        max_lr = [lr_IE, lr_MD] + ([lr_clip] if finetune_clip else []) + ([lr_PE] if has_mask_prompt else [])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader_unlabeled)*2,#*(unsupervised_per_supervised+1),
        epochs=total_epochs,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=5e2
    )
        
    # if use_scheduler:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_epochs,
    #     eta_min=1e-6
    #     )

    if simple_loss:
        if rank==0: 
            logger.info('using simple BCE')
        criterion_sup = nn.BCEWithLogitsLoss()
    else:
        if iou_loss:
            bce_weight = 0.2
            dice_weight = 1.5
            iou_weight = 0.8
            criterion_sup = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, iou_weight=iou_weight)
            iou_f = IoULoss(eps=1e-6)
            if rank==0:
                logger.info('using BCE, Dice, and IoU loss with weights:',bce_weight,dice_weight,iou_weight)

        else:       
            bce_weight = 0.5
            dice_weight = 1.5
            if rank==0:
                logger.info('using BCE and Dice, with weights:',bce_weight,dice_weight)
            criterion_sup = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    #optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    #criterion = nn.BCEWithLogitsLoss()
    if finetune_clip:
        clip_model.train()
    else:
        clip_model.eval()

    trainable_sam.train()
    #clip_model.eval()
    frozen_sam.eval()

    
    start_epoch = 0

    best_IoU, best_val = 0.01, 10000

    # Resume support
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
        frozen_sam.load_state_dict(checkpoint['frozen_sam_state_dict'])
        if finetune_clip and 'clip_model' in checkpoint:
            clip_model.module.load_state_dict(checkpoint['clip_model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0:
                logger.info('found optimizier checkpoint, loading.')
        except KeyError:
            pass
        if 'scheduler_state_dict' in checkpoint:
            if rank==0:
                logger.info('found scheduler checkpoint, loading.')
            if use_scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if start_epoch + epochs > total_epochs:
            raise ValueError(
                f"Training would exceed total_epochs={total_epochs}: "
                f"start_epoch={start_epoch}, requested epochs={epochs}."
            )
        
        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()

    
    else:
        if rank == 0:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")
        best_IoU, best_val = 0.1, 10000
        if not skip_init_eval:
            dataloader_val.sampler.set_epoch(0)
            best_IoU, best_val = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )
        
        if rank == 0:
            logger.info(f"Initial Eval -> IoU: {best_IoU:.4f}, Val Loss: {best_val:.4f}")
    
    end_epochs = start_epoch + epochs
    if end_epochs > total_epochs:
        raise ValueError(
            f"Training would exceed total_epochs={total_epochs}: "
            f"start_epoch={start_epoch}, requested epochs={epochs}."
        )

    for epoch in range(start_epoch, end_epochs):
        dataloader_labeled.sampler.set_epoch(epoch)
        dataloader_unlabeled.sampler.set_epoch(epoch)

        if rank == 0:
            progress_bar = tqdm(dataloader_unlabeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        else:
            progress_bar = dataloader_unlabeled

        labeled_iter = iter(dataloader_labeled)
        total_loss = 0.0
        total_confidence_ratio = 0.0

        loss_unsupervised_weight = cosine_rampup(epoch, 10, a)
        confidence_threshold = cosine_threshold_scheduler(epoch, 10, 0.7, conf_thresh)
        decay_sup = ema_decay_schedule(epoch, total_epochs, start=decay_sup_start, end=0.9999)
        decay_unsup = ema_decay_schedule(epoch, total_epochs, start=decay_unsup_start, end=0.99999)

        for batch_idx, batch_unlabeled in enumerate(progress_bar):
            optimizer.zero_grad()

            # === Supervised step ===
            try:
                batch_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(dataloader_labeled)
                batch_labeled = next(labeled_iter)

            sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                batch_labeled, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

            loss_sup = criterion_sup(sam_output, label_masks)
            loss_sup.backward()
            total_loss += loss_sup.item()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_sup)

            # === Unsupervised step === ##########


            #optimizer.zero_grad()

            frozen_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU(
                batch_unlabeled, clip_model, frozen_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

            pseudo_labels, conf_mask, conf_ratio = generate_pseudo_labels(frozen_output, threshold=confidence_threshold)
            total_confidence_ratio += conf_ratio.mean().item()

            train_output = trainable_sam(inputs)
            msk_loss = masked_loss(train_output, pseudo_labels, conf_mask)
            dice_l = dice_loss(train_output, pseudo_labels)
            if iou_loss:
                iou_l = iou_f(train_output, pseudo_labels)
                loss_unsup = loss_unsupervised_weight * (msk_loss + dice_l + iou_l)
            else:
                loss_unsup = loss_unsupervised_weight * (msk_loss + dice_l)

            loss_unsup.backward()
            total_loss += loss_unsup.item()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay_unsup)

            avg_loss = total_loss / ((batch_idx + 1) * 2)  # 2 steps per batch_idx
            avg_conf_ratio = total_confidence_ratio / (batch_idx + 1)
            if rank == 0:
                progress_bar.set_postfix(loss=avg_loss, conf_ratio=f"{avg_conf_ratio * 100:.2f}%")



        # Validation after epoch
        dataloader_val.sampler.set_epoch(epoch) 
        IoU, val_loss = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

        if rank == 0 and not ignore_wandb:
            train_loss = total_loss / (len(dataloader_unlabeled)*2)
            lr_logs = {
                f"lr/group_{i}": param_group['lr']
                for i, param_group in enumerate(optimizer.param_groups)
            }

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "IoU": IoU,
                "confidence_ratio": avg_conf_ratio,
                "Best_val": best_val,
                "Best_IoU": best_IoU,
                **lr_logs
            })
            
            logger.info(f"Epoch {epoch+1}/{end_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val:.4f} | IoU: {IoU:.4f} | Best IoU: {best_IoU:.4f}")



        # Save best IoU model
        if IoU > best_IoU:
            best_IoU = IoU
            if rank == 0:
                ckpt_path = os.path.join(save_path, f"semisup_sam_best_IoU_{save_ext}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_sam.module.state_dict(),
                    'frozen_sam_state_dict': frozen_sam.state_dict(),
                    'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                    'val_loss': val_loss,
                    'best_val': best_val,
                    'best_IoU': best_IoU
                }, ckpt_path)
                logger.info(f"Best IoU {best_IoU:.4f} - Saved checkpoint to {ckpt_path}")

        # # Periodic checkpoint
        # if (epoch + 1) % 10 == 0 and rank == 0:
        #     ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': trainable_sam.module.state_dict(),
        #         'frozen_sam_state_dict': frozen_sam.state_dict(),
        #         'val_loss': val_loss,
        #         'best_val': best_val,
        #         'best_IoU': best_IoU
        #     }, ckpt_path)
   
    if rank == 0:
        ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainable_sam.module.state_dict(),
            'frozen_sam_state_dict': frozen_sam.state_dict(),
            'clip_model': clip_model.module.state_dict() if finetune_clip else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
            'val_loss': val_loss,
            'best_val': best_val,
            'best_IoU': best_IoU
        }, ckpt_path)
    return trainable_sam


def train_combined_consistency_for_scripts_DDP(
    frozen_sam, trainable_sam, clip_model,
    dataloader_labeled, dataloader_unlabeled, dataloader_val,
    classes, total_epochs, epochs, save_path, save_ext, logger, split,
    unsupervised_per_supervised=64, SAM_text=False, has_mask_prompt=False, weight_decay=0.0,
    include_backgrounds=False, features_input=False, is_vanilla_sam=False, finetune_clip=False,
    use_scheduler=True, lr_IE=1e-4, lr_MD=5e-5, lr_clip=1e-6, lr_PE=1e-5, clip_crop_size=224,
    decay_sup_start=0.95, decay_unsup_start=0.999, a=0.2, resume_path=None, ignore_wandb = False, skip_init_eval=False,
    conf_thresh=0.9, simple_loss=True, iou_loss=False, prompt_ensemble=False, sim_func_surgery=False, 
    norm_dim=1, pct_start=0.2, final_div_factor=1e2, options=None
    ):
    # Enable multi-GPU
    # if multi_gpu:
    #     logger.info("Running in DDP mode. Model already wrapped in DistributedDataParallel.")
    # else:
    #     logger.info("Using Single GPU or CPU")

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0 and not ignore_wandb:
        wandb.init(
            project="sam-clip-semisupervised-ddp-"+split,
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr_IE,
                "sam_type": "vanilla" if is_vanilla_sam else "parallel",
                "SAM_text": SAM_text
            }
        )

    if rank==0: 
        logger.info(options)

    adapter_params = [
    param for name, param in trainable_sam.module.image_encoder.named_parameters()
    if any(block in name for block in ['Space_Adapter', 'MLP_Adapter', 'Depth_Adapter']) and param.requires_grad
    ]

    # Get trainable mask decoder parameters
    mask_decoder_params = [
        param for name, param in trainable_sam.module.mask_decoder.named_parameters()
        if param.requires_grad
    ]

    if has_mask_prompt:
        prompt_encoder_params = [
            param for name,param in trainable_sam.module.prompt_encoder.mask_downscaling.named_parameters()
            if param.requires_grad
        ]

    if finetune_clip:
        clip_model_params = [
            param for name, param in clip_model.module.named_parameters()
            if param.requires_grad
        ]
    optimizer = optim.AdamW(
    [
        {'params': adapter_params, 'lr': lr_IE, 'weight_decay': weight_decay},
        {'params': mask_decoder_params, 'lr': lr_MD, 'weight_decay': weight_decay},
        *(
            [{'params': clip_model_params, 'lr': lr_clip, 'weight_decay': 0.0}]
            if finetune_clip else []
        ),
        *(
            [{'params': prompt_encoder_params, 'lr':lr_PE, 'weight_decay':weight_decay}]
            if has_mask_prompt else []
        )
    ]
    )

    if use_scheduler:
        max_lr = [lr_IE, lr_MD] + ([lr_clip] if finetune_clip else []) + ([lr_PE] if has_mask_prompt else [])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader_labeled)*(unsupervised_per_supervised+1),
        epochs=total_epochs,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=5e2
    )
        
    # if use_scheduler:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_epochs,
    #     eta_min=1e-6
    #     )

    if simple_loss:
        if rank==0: 
            logger.info('using simple BCE')
        criterion_sup = nn.BCEWithLogitsLoss()
    else:
        if iou_loss:
            bce_weight = 0.2
            dice_weight = 1.5
            iou_weight = 0.8
            criterion_sup = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, iou_weight=iou_weight)
            iou_f = IoULoss(eps=1e-6)
            if rank==0:
                logger.info('using BCE, Dice, and IoU loss with weights:',bce_weight,dice_weight,iou_weight)

        else:       
            bce_weight = 0.5
            dice_weight = 1.5
            if rank==0:
                logger.info('using BCE and Dice, with weights:',bce_weight,dice_weight)
            criterion_sup = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    #optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    #criterion = nn.BCEWithLogitsLoss()
    if finetune_clip:
        clip_model.train()
    else:
        clip_model.eval()

    trainable_sam.train()
    #clip_model.eval()
    frozen_sam.eval()

    
    start_epoch = 0

    best_IoU, best_val = 0.01, 10000

    # Resume support
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
        frozen_sam.load_state_dict(checkpoint['frozen_sam_state_dict'])
        if finetune_clip and 'clip_model' in checkpoint:
            clip_model.module.load_state_dict(checkpoint['clip_model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0:
                logger.info('found optimizier checkpoint, loading.')
        except KeyError:
            pass
        if 'scheduler_state_dict' in checkpoint:
            if rank==0:
                logger.info('found scheduler checkpoint, loading.')
            if use_scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if start_epoch + epochs > total_epochs:
            raise ValueError(
                f"Training would exceed total_epochs={total_epochs}: "
                f"start_epoch={start_epoch}, requested epochs={epochs}."
            )
        
        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()

    
    else:
        if rank == 0:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")
        best_IoU, best_val = 0.1, 10000
        if not skip_init_eval:
            dataloader_val.sampler.set_epoch(0)
            best_IoU, best_val = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )
        
        if rank == 0:
            logger.info(f"Initial Eval -> IoU: {best_IoU:.4f}, Val Loss: {best_val:.4f}")
    
    end_epochs = start_epoch + epochs
    if end_epochs > total_epochs:
        raise ValueError(
            f"Training would exceed total_epochs={total_epochs}: "
            f"start_epoch={start_epoch}, requested epochs={epochs}."
        )

    for epoch in range(start_epoch, end_epochs):
        dataloader_labeled.sampler.set_epoch(epoch)
        dataloader_unlabeled.sampler.set_epoch(epoch)
        

        logger.info(f"\nEpoch {epoch+1}/{end_epochs} - Training")
        if rank == 0:
            progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        else:
            progress_bar = dataloader_labeled  # Other ranks don't use tqdm

        #progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        total_loss = 0
        total_confidence_ratio = 0
        #exception_counter = 0
        

        loss_unsupervised_weight = cosine_rampup(current_epoch=epoch, rampup_epochs=10, max_value=a)
        confidence_threshold = cosine_threshold_scheduler(current_epoch=epoch, warmup_epochs=10, start=0.7, end=conf_thresh)
        
        unlabeled_iter = iter(dataloader_unlabeled)

        decay_sup = ema_decay_schedule(epoch, total_epochs, start=decay_sup_start, end=0.9999)
        decay_unsup = ema_decay_schedule(epoch, total_epochs, start=decay_unsup_start, end=0.99999)

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # === Supervised step ===
            sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                batch, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

            loss = criterion_sup(sam_output, label_masks)
            loss.backward()
            total_loss += loss.item()
            # if finetune_clip:
            #     similarity_loss = clip_sim_loss_f(sim_prompt.squeeze(1), label_masks.squeeze(1))
            #     contrastive_loss = clip_contrastive_loss(clip_emb,text_feat)
            #     clip_loss = similarity_loss + contrastive_loss
            #     clip_loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_sup)

            # === Unsupervised steps ===
            if rank == 0:
                inner_bar = tqdm(range(unsupervised_per_supervised), leave=False, desc="Unsupervised steps", dynamic_ncols=True)
            else:
                inner_bar = range(unsupervised_per_supervised)

            for _ in inner_bar:
                optimizer.zero_grad()

                try:
                    batch_unlabeled = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(dataloader_unlabeled)
                    batch_unlabeled = next(unlabeled_iter)

                
                imgs_weak = batch_unlabeled['image_weak']
                imgs_strong = batch_unlabeled['image_strong']
                img_class = batch_unlabeled['text']

                frozen_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU_2(
                    imgs_weak,img_class,label_masks, clip_model, frozen_sam, classes, clip_crop_size=clip_crop_size,
                    has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                    sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
                )

                pseudo_labels, conf_mask, conf_ratio = generate_pseudo_labels(frozen_output, threshold=confidence_threshold)
                

                train_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU_2(
                    imgs_strong,img_class,label_masks, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                    has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                    SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                    features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                    sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
                )
                # logger.info('#############################')
                # logger.info(train_output.requires_grad)

                #train_output = trainable_sam(inputs)

                msk_loss = masked_loss(train_output, pseudo_labels, conf_mask)
                dice_l = dice_loss(train_output,pseudo_labels)
                if iou_loss:
                    iou_l = iou_f(train_output,pseudo_labels)
                    unsup_loss = loss_unsupervised_weight * (msk_loss + dice_l + iou_l)
                else:
                    unsup_loss = loss_unsupervised_weight * (msk_loss + dice_l)
                unsup_loss.backward()
                total_loss += unsup_loss.item()
                total_confidence_ratio += conf_ratio.mean().item()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()


                frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_unsup)

            avg_loss = total_loss / ((batch_idx + 1) * (unsupervised_per_supervised + 1))
            avg_conf_ratio = total_confidence_ratio / ((batch_idx + 1) * unsupervised_per_supervised)
            if rank == 0:
                progress_bar.set_postfix(loss=avg_loss, conf_ratio=f"{avg_conf_ratio * 100:.2f}%")


        # Validation after epoch
        dataloader_val.sampler.set_epoch(epoch) 
        IoU, val_loss = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

        if rank == 0 and not ignore_wandb:
            train_loss = total_loss / (len(dataloader_labeled) * (unsupervised_per_supervised + 1))
            lr_logs = {
                f"lr/group_{i}": param_group['lr']
                for i, param_group in enumerate(optimizer.param_groups)
            }

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "IoU": IoU,
                "confidence_ratio": avg_conf_ratio,
                "Best_val": best_val,
                "Best_IoU": best_IoU,
                **lr_logs
            })
            
            logger.info(f"Epoch {epoch+1}/{end_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val:.4f} | IoU: {IoU:.4f} | Best IoU: {best_IoU:.4f}")



        # Save best IoU model
        if IoU > best_IoU:
            best_IoU = IoU
            if rank == 0:
                ckpt_path = os.path.join(save_path, f"semisup_sam_best_IoU_{save_ext}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_sam.module.state_dict(),
                    'frozen_sam_state_dict': frozen_sam.state_dict(),
                    'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                    'val_loss': val_loss,
                    'best_val': best_val,
                    'best_IoU': best_IoU
                }, ckpt_path)
                logger.info(f"Best IoU {best_IoU:.4f} - Saved checkpoint to {ckpt_path}")

        # # Periodic checkpoint
        # if (epoch + 1) % 10 == 0 and rank == 0:
        #     ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': trainable_sam.module.state_dict(),
        #         'frozen_sam_state_dict': frozen_sam.state_dict(),
        #         'val_loss': val_loss,
        #         'best_val': best_val,
        #         'best_IoU': best_IoU
        #     }, ckpt_path)
   
    if rank == 0:
        ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainable_sam.module.state_dict(),
            'frozen_sam_state_dict': frozen_sam.state_dict(),
            'clip_model': clip_model.module.state_dict() if finetune_clip else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
            'val_loss': val_loss,
            'best_val': best_val,
            'best_IoU': best_IoU
        }, ckpt_path)
    return trainable_sam


def train_combined_consistency_for_scripts_DDP_eqsup(
    frozen_sam, trainable_sam, clip_model,
    dataloader_labeled, dataloader_unlabeled, dataloader_val,
    classes, total_epochs, epochs, save_path, save_ext, logger, split,
    unsupervised_per_supervised=2, SAM_text=False, has_mask_prompt=False, weight_decay=0.0,
    include_backgrounds=False, features_input=False, is_vanilla_sam=False, finetune_clip=False,
    use_scheduler=True, lr_IE=1e-4, lr_MD=5e-5, lr_clip=1e-6, lr_PE=1e-5, clip_crop_size=224,
    decay_sup_start=0.95, decay_unsup_start=0.999, a=0.2, resume_path=None, ignore_wandb = False, skip_init_eval=False,
    conf_thresh=0.9, simple_loss=True, iou_loss=False, prompt_ensemble=False, sim_func_surgery=False, 
    norm_dim=1, pct_start=0.2, final_div_factor=1e2, sam_cross=False, lr_cr=5e-5, options=None):
    # Enable multi-GPU
    # if multi_gpu:
    #     logger.info("Running in DDP mode. Model already wrapped in DistributedDataParallel.")
    # else:
    #     logger.info("Using Single GPU or CPU")

    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0 and not ignore_wandb:
        wandb.init(
            project="sam-clip-semisupervised-ddp-"+split,
            name=f"run_{save_ext}",
            config={
                "epochs": epochs,
                "learning_rate": lr_IE,
                "sam_type": "vanilla" if is_vanilla_sam else "parallel",
                "SAM_text": SAM_text
            }
        )

    if rank==0: 
        logger.info(options)

    adapter_params = [
    param for name, param in trainable_sam.module.image_encoder.named_parameters()
    if any(block in name for block in ['Space_Adapter', 'MLP_Adapter', 'Depth_Adapter']) and param.requires_grad
    ]

    if sam_cross:
        sam_cross_params = [
            param for name,param in trainable_sam.module.image_encoder.cross_blocks.named_parameters() if param.requires_grad
        ]

    # Get trainable mask decoder parameters
    mask_decoder_params = [
        param for name, param in trainable_sam.module.mask_decoder.named_parameters()
        if param.requires_grad
    ]

    if has_mask_prompt:
        prompt_encoder_params = [
            param for name,param in trainable_sam.module.prompt_encoder.named_parameters()
            if param.requires_grad
        ]

    if finetune_clip:
        clip_model_params = [
            param for name, param in clip_model.module.named_parameters()
            if param.requires_grad
        ]
    optimizer = optim.AdamW(
    [
        {'params': adapter_params, 'lr': lr_IE, 'weight_decay': weight_decay},
        {'params': mask_decoder_params, 'lr': lr_MD, 'weight_decay': weight_decay},
        *(
            [{'params': clip_model_params, 'lr': lr_clip, 'weight_decay': 0.0}]
            if finetune_clip else []
        ),
        *(
            [{'params': sam_cross_params, 'lr':lr_cr, 'weight_decay':weight_decay}] if sam_cross else []
        ),
        *(
            [{'params': prompt_encoder_params, 'lr':lr_PE, 'weight_decay':weight_decay}]
            if has_mask_prompt else []
        )
    ]
    )

    if use_scheduler:
        max_lr = [lr_IE, lr_MD] + ([lr_clip] if finetune_clip else []) + ([lr_PE] if has_mask_prompt else []) + ([lr_cr] if sam_cross else [])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader_unlabeled)*2,
        epochs=total_epochs,
        pct_start=0.2,
        anneal_strategy='cos',
        final_div_factor=5e2
    )
        
    # if use_scheduler:
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_epochs,
    #     eta_min=1e-6
    #     )

    if simple_loss:
        if rank==0: 
            logger.info('using simple BCE')
        criterion_sup = nn.BCEWithLogitsLoss()
    else:
        if iou_loss:
            bce_weight = 0.2
            dice_weight = 1.5
            iou_weight = 0.8
            criterion_sup = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight, iou_weight=iou_weight)
            iou_f = IoULoss(eps=1e-6)
            if rank==0:
                logger.info('using BCE, Dice, and IoU loss with weights:',bce_weight,dice_weight,iou_weight)

        else:       
            bce_weight = 0.5
            dice_weight = 1.5
            if rank==0:
                logger.info('using BCE and Dice, with weights:',bce_weight,dice_weight)
            criterion_sup = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    #optimizer = optim.Adam(trainable_sam.parameters(), lr=lr)
    #criterion = nn.BCEWithLogitsLoss()
    if finetune_clip:
        clip_model.train()
    else:
        clip_model.eval()

    trainable_sam.train()
    #clip_model.eval()
    frozen_sam.eval()

    
    start_epoch = 0

    best_IoU, best_val = 0.01, 10000

    # Resume support
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        trainable_sam.module.load_state_dict(checkpoint['model_state_dict'])
        frozen_sam.load_state_dict(checkpoint['frozen_sam_state_dict'])
        if finetune_clip and 'clip_model' in checkpoint:
            clip_model.module.load_state_dict(checkpoint['clip_model'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0:
                logger.info('found optimizier checkpoint, loading.')
        except KeyError:
            pass
        if 'scheduler_state_dict' in checkpoint:
            if rank==0:
                logger.info('found scheduler checkpoint, loading.')
            if use_scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if start_epoch + epochs > total_epochs:
            raise ValueError(
                f"Training would exceed total_epochs={total_epochs}: "
                f"start_epoch={start_epoch}, requested epochs={epochs}."
            )
        
        best_val = checkpoint.get('val_loss', best_val)
        best_IoU = checkpoint.get('best_IoU', best_IoU)
        if rank == 0:
            logger.info(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()

    
    else:
        if rank == 0:
            logger.warning(f"Checkpoint not found at {resume_path}. Starting from scratch.")
        best_IoU, best_val = 0.1, 10000
        if not skip_init_eval:
            dataloader_val.sampler.set_epoch(0)
            best_IoU, best_val = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )
        
        if rank == 0:
            logger.info(f"Initial Eval -> IoU: {best_IoU:.4f}, Val Loss: {best_val:.4f}")
    
    end_epochs = start_epoch + epochs
    if end_epochs > total_epochs:
        raise ValueError(
            f"Training would exceed total_epochs={total_epochs}: "
            f"start_epoch={start_epoch}, requested epochs={epochs}."
        )

    for epoch in range(start_epoch, end_epochs):
        dataloader_labeled.sampler.set_epoch(epoch)
        dataloader_unlabeled.sampler.set_epoch(epoch)
        

        logger.info(f"\nEpoch {epoch+1}/{end_epochs} - Training")
        if rank == 0:
            progress_bar = tqdm(dataloader_unlabeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        else:
            progress_bar = dataloader_unlabeled  # Other ranks don't use tqdm

        #progress_bar = tqdm(dataloader_labeled, desc=f"Epoch {epoch+1}/{end_epochs}")
        labeled_iter = iter(dataloader_labeled)
        total_loss = 0
        total_confidence_ratio = 0
        #exception_counter = 0
        

        loss_unsupervised_weight = cosine_rampup(current_epoch=epoch, rampup_epochs=10, max_value=a)
        confidence_threshold = cosine_threshold_scheduler(current_epoch=epoch, warmup_epochs=10, start=0.7, end=conf_thresh)
        decay_sup = ema_decay_schedule(epoch, total_epochs, start=decay_sup_start, end=0.9999)
        decay_unsup = ema_decay_schedule(epoch, total_epochs, start=decay_unsup_start, end=0.99999)

        for batch_idx, batch_unlabeled in enumerate(progress_bar):
            optimizer.zero_grad()

            # === Supervised step ===
            try:
                batch_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(dataloader_labeled)
                batch_labeled = next(labeled_iter)

            sam_output, clip_emb, text_feat, label_masks, sim_prompt, _ = forward_supervised_SAM_CLIP_multiGPU(
                batch_labeled, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

            loss_sup = criterion_sup(sam_output, label_masks)
            loss_sup.backward()
            total_loss += loss_sup.item()
            # if finetune_clip:
            #     similarity_loss = clip_sim_loss_f(sim_prompt.squeeze(1), label_masks.squeeze(1))
            #     contrastive_loss = clip_contrastive_loss(clip_emb,text_feat)
            #     clip_loss = similarity_loss + contrastive_loss
            #     clip_loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_sup)

            # === Unsupervised steps ===
            
            optimizer.zero_grad()


            imgs_weak = batch_unlabeled['image_weak']
            imgs_strong = batch_unlabeled['image_strong']
            img_class = batch_unlabeled['text']

            frozen_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU_2(
                imgs_weak,img_class,label_masks, clip_model, frozen_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

            pseudo_labels, conf_mask, conf_ratio = generate_pseudo_labels(frozen_output, threshold=confidence_threshold)
            

            train_output, clip_emb, text_feat, _, sim_prompt, inputs = forward_supervised_SAM_CLIP_multiGPU_2(
                imgs_strong,img_class,label_masks, clip_model, trainable_sam, classes, clip_crop_size=clip_crop_size,
                has_mask_prompt=has_mask_prompt, include_backgrounds=include_backgrounds,
                SAM_text=SAM_text, is_vanilla_sam=is_vanilla_sam,
                features_input=features_input, points_from_gt=False, prompt_ensemble=prompt_ensemble,
                sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )
            # logger.info('#############################')
            # logger.info(train_output.requires_grad)

            #train_output = trainable_sam(inputs)

            msk_loss = masked_loss(train_output, pseudo_labels, conf_mask)
            dice_l = dice_loss(train_output,pseudo_labels)
            if iou_loss:
                iou_l = iou_f(train_output,pseudo_labels)
                unsup_loss = loss_unsupervised_weight * (msk_loss + dice_l + iou_l)
            else:
                unsup_loss = loss_unsupervised_weight * (msk_loss + dice_l)
            unsup_loss.backward()
            total_loss += unsup_loss.item()
            total_confidence_ratio += conf_ratio.mean().item()
            optimizer.step()
            if use_scheduler:
                scheduler.step()


            frozen_sam = update_ema_frozen(frozen_sam, trainable_sam, decay=decay_unsup)

        avg_loss = total_loss / ((batch_idx + 1) * 2)
        avg_conf_ratio = total_confidence_ratio / ((batch_idx + 1))
        if rank == 0:
            progress_bar.set_postfix(loss=avg_loss, conf_ratio=f"{avg_conf_ratio * 100:.2f}%")


        # Validation after epoch
        dataloader_val.sampler.set_epoch(epoch) 
        IoU, val_loss = eval_model_DDP(
            clip_model, trainable_sam, dataloader_val, classes,
            criterion=criterion_sup, mask_prompt=has_mask_prompt,
            include_backgrounds=False, SAM_text=SAM_text, return_IoU=True,
            points_from_gt=False, clip_crop_size=clip_crop_size, prompt_ensemble=prompt_ensemble, 
            sim_func_surgery=sim_func_surgery, norm_dim=norm_dim
            )

        if rank == 0 and not ignore_wandb:
            train_loss = total_loss / (len(dataloader_labeled) * 2)
            lr_logs = {
                f"lr/group_{i}": param_group['lr']
                for i, param_group in enumerate(optimizer.param_groups)
            }

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "IoU": IoU,
                "confidence_ratio": avg_conf_ratio,
                "Best_val": best_val,
                "Best_IoU": best_IoU,
                **lr_logs
            })
            
            logger.info(f"Epoch {epoch+1}/{end_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val:.4f} | IoU: {IoU:.4f} | Best IoU: {best_IoU:.4f}")



        # Save best IoU model
        if IoU > best_IoU:
            best_IoU = IoU
            if rank == 0:
                ckpt_path = os.path.join(save_path, f"semisup_sam_best_IoU_{save_ext}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_sam.module.state_dict(),
                    'frozen_sam_state_dict': frozen_sam.state_dict(),
                    'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                    'val_loss': val_loss,
                    'best_val': best_val,
                    'best_IoU': best_IoU
                }, ckpt_path)
                logger.info(f"Best IoU {best_IoU:.4f} - Saved checkpoint to {ckpt_path}")

        #last epoch save
        if rank == 0:
            ckpt_path = os.path.join(save_path, f"semisup_sam_saved_{save_ext}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainable_sam.module.state_dict(),
                'frozen_sam_state_dict': frozen_sam.state_dict(),
                'clip_model': clip_model.module.state_dict() if finetune_clip else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
                'val_loss': val_loss,
                'best_val': best_val,
                'best_IoU': best_IoU
            }, ckpt_path)
   
    if rank == 0:
        ckpt_path = os.path.join(save_path, f"semisup_sam_epoch_{epoch+1}{save_ext}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainable_sam.module.state_dict(),
            'frozen_sam_state_dict': frozen_sam.state_dict(),
            'clip_model': clip_model.module.state_dict() if finetune_clip else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if use_scheduler else None,
            'val_loss': val_loss,
            'best_val': best_val,
            'best_IoU': best_IoU
        }, ckpt_path)
    return trainable_sam

