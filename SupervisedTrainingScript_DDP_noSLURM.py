import os
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import COCOBinaryMaskDataset, COCOBinaryMaskDataset_wAUG, COCOBinaryMaskDataset_wNewAUG, ADE20KBinaryMaskDataset_wNewAUG, ADE20KBinaryMaskDataset, PascalVOCBinaryMaskDataset_wNewAUG, PascalVOCBinaryMaskDataset, PascalVOCBinaryMaskDatasetUnified, CamouflagedBinaryMaskDataset_wTextEmb, CropsBinaryMaskDataset_wNewAUG
from develop_semivl import CLIP_Decoder_Head
from CLIP_SAM_Utils_Final_MultiGPU import initialize_clip, initialize_sam, load_sam
from Training_Functions_multi_gpu import train_supervised_final_for_scripts_DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
from CLIP_SAM_Utils_Final_MultiGPU import (
    count_params,
    count_trainable_blocks_sam_image_encoder,
    count_trainable_blocks_clip_vision,
    format_param_pct,
)


def setup_logger(log_dir="logs", log_filename="train_semi.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Semi-supervised training of SAM+CLIP on COCO")
    parser.add_argument("--labeled_json", type=str, help="Path to labeled samples JSON")
    parser.add_argument("--val_json", type=str, default="COCO_samples_val.json", help="Path to validation samples JSON")
    parser.add_argument("--sam_ckpt", type=str, required=True, help="Path to sam base checkpoint")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr_IE", type=float, default=1e-4, help="Learning rate for Image Encoder")
    parser.add_argument("--lr_MD", type=float, default=5e-5, help="Learning rate for Mask Decoder") 
    parser.add_argument("--lr_PE", type=float, default=1e-5, help="Learning rate for Prompt Encoder") 
    parser.add_argument("--lr_clip",type=float, default=1e-6)  
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store logs")
    parser.add_argument("--split", type=str, required=True, help="Split name, e.g. 1_64")
    parser.add_argument("--save_path", type=str, default=".", help="Directory to save model checkpoints")
    parser.add_argument("--version", type=int, default=1, help="Experiment version suffix")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to resume training from checkpoint")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--ignore_wandb', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    #parser.add_argument('--world_size', type=int)
    #parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:3456')
    parser.add_argument('--simple_load', type=lambda x: x.lower() == 'true', default=False)
    
    parser.add_argument('--skip_init_eval', type=lambda x: x.lower() == 'true', default=False)
    #parser.add_argument('--points_from_gt', type=lambda x: x.lower() == 'true', default=True, nargs='?', const=True, help='Use ground truth points (default: True). Pass "false" to disable.')
    parser.add_argument("--points_from_gt_train", type=lambda x: x.lower() == 'true', default=True, nargs='?', const=True)
    parser.add_argument("--points_from_gt_eval", type=lambda x: x.lower() == 'true', default=True, nargs='?', const=True)
    parser.add_argument('--simple_loss', type=lambda x: x.lower() == 'true', default=True, nargs='?', const=True)
    parser.add_argument('--total_epochs', type=int, default=30)
    parser.add_argument('--finetune_clip', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_scheduler', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--clip_crop_size', type=int, default=224)
    parser.add_argument('--include_backgrounds', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--mask_prompts',type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--prompt_ensemble',type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--sim_func_surgery',type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--norm_dim', type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument('--iou_loss', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--structure_loss', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--enhanced_structure_loss', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--sam_cross', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--lr_cr", type=float, default=1e-4, help="Learning rate for sam cross block")
    parser.add_argument('--finetune_PE', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--parallel_sim', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--finetune_neck', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--lr_neck", type=float, default=1e-5, help="Learning rate neck")
    parser.add_argument("--text_vis", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--text_vis_cross", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--clip_decoder_head", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--lr_clip_dec_head", type=float, default=1e-4, help="Learning rate neck")
    parser.add_argument("--out_size_512",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--ade20k",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--pascal",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--crops",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--clip_logscale",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--load_clip",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--clip_ckpt", type=str, default='', help="Path to sam base checkpoint")
    parser.add_argument("--use_mae", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--camouflaged", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--bce_weight",type=float, default=1)
    parser.add_argument("--dice_weight",type=float, default=1)
    parser.add_argument("--iou_weight",type=float, default=1)
    parser.add_argument("--no_text",type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--SAM_IE", type=str, default='Vanilla')
    parser.add_argument("--clip_type",type=str, default='CS-ViT-B/16')
    parser.add_argument("--use_soft_prompts", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--soft_prompt_sigmoid", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--wandb_extension", type=str, default='')
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank if only training on one gpu")
    

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def main_worker(local_rank, args):
    # Initialize the distributed environment
    set_seed(seed=42)
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    #dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=ngpus_per_node, rank=rank)
    dist.init_process_group(backend=args.dist_backend)

    logger = setup_logger(args.log_dir) if dist.get_rank() == 0 else logging.getLogger("dummy")
    # torch.set_num_threads(8)
    # torch.set_num_interop_threads(2)

    # === Dataset Loading ===
    logger.info("Loading datasets...")


    COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "dining table", "dog", "horse", "motorbike", "person", 
    "potted plant", "sheep", "sofa", "train", "tv"
    ]

    ADE_CLASSES = [s.strip() for s in """
    wall, building, sky, floor, tree, ceiling, road, bed, windowpane, grass,
    cabinet, sidewalk, person, earth, door, table, mountain, plant, curtain, chair,
    car, water, painting, sofa, shelf, house, sea, mirror, rug, field,
    armchair, seat, fence, desk, rock, wardrobe, lamp, bathtub, railing, cushion,
    base, box, column, signboard, chest of drawers, counter, sand, sink, skyscraper, fireplace,
    refrigerator, grandstand, path, stairs, runway, case, pool table, pillow, screen door, stairway,
    river, bridge, bookcase, blind, coffee table, toilet, flower, book, hill, bench,
    countertop, stove, palm, kitchen island, computer, swivel chair, boat, bar, arcade machine, hovel,
    bus, towel, light, truck, tower, chandelier, awning, streetlight, booth, television receiver,
    airplane, dirt track, apparel, pole, land, bannister, escalator, ottoman, bottle, buffet,
    poster, stage, van, ship, fountain, conveyer belt, canopy, washer, plaything, swimming pool,
    stool, barrel, basket, waterfall, tent, bag, minibike, cradle, oven, ball,
    food, step, tank, trade name, microwave, pot, animal, bicycle, lake, dishwasher,
    screen, blanket, sculpture, hood, sconce, vase, traffic light, tray, ashcan, fan,
    pier, crt screen, plate, monitor, bulletin board, shower, radiator, glass, clock, flag
    """.split(",") if s.strip()]


    img_size, out_size = 1024, 256
    if args.out_size_512:
        out_size = 512
        logger.info('Using outsize 512')

    if args.ade20k:
        labeled_dataset = ADE20KBinaryMaskDataset_wNewAUG(args.labeled_json, img_size=img_size, out_size=out_size)
        val_dataset = ADE20KBinaryMaskDataset(args.val_json, img_size=img_size, out_size=out_size)
        dataset = 'ADE20K'
        classes = ADE_CLASSES
    elif args.pascal:
        labeled_dataset = PascalVOCBinaryMaskDatasetUnified(args.labeled_json, img_size=img_size, out_size=out_size, use_syn=True, use_aug=True)
        val_dataset = PascalVOCBinaryMaskDatasetUnified(args.val_json, img_size=img_size, out_size=out_size, use_syn=True, use_aug=False)
        dataset = 'PascalVOC'
        classes = PASCAL_CLASSES
    elif args.camouflaged:
        labeled_dataset = CamouflagedBinaryMaskDataset_wTextEmb(split='TrainDataset', img_size=img_size, out_size=out_size, emb_dir='camouflaged_text_embeds')
         # --- three validation sets ---
        val_datasets = {
            "CAMO":       CamouflagedBinaryMaskDataset_wTextEmb(split='TestDataset/CAMO', img_size=img_size, out_size=out_size, emb_dir='camouflaged_text_embeds'),
            "COD10K":     CamouflagedBinaryMaskDataset_wTextEmb(split='TestDataset/COD10K', img_size=img_size, out_size=out_size, emb_dir='camouflaged_text_embeds'),
            "CHAMELEON":  CamouflagedBinaryMaskDataset_wTextEmb(split='TestDataset/CHAMELEON', img_size=img_size, out_size=out_size, emb_dir='camouflaged_text_embeds'),
        } 
        dataset = 'Camouflaged'
        classes=['Camouflaged object']
    elif args.crops:
        labeled_dataset = CropsBinaryMaskDataset_wNewAUG(args.labeled_json, img_size=img_size, out_size=out_size, use_aug=False)
        val_dataset = CropsBinaryMaskDataset_wNewAUG(args.val_json, img_size=img_size, out_size=out_size, use_aug=False)
        print('Not using Aug for train dataset in Crops')
        dataset = 'Crops'
        classes = ["Canola","Kochia","Soybean","Sunflower"]
    else:
        labeled_dataset = COCOBinaryMaskDataset_wNewAUG(args.labeled_json, img_size=img_size, out_size=out_size)
        val_dataset = COCOBinaryMaskDataset(args.val_json, img_size=img_size, out_size=out_size)
        dataset = 'COCO'
        classes = COCO_CLASSES

    augmentations = True
    new_aug = True

    

    # labeled_dataset = COCOBinaryMaskDataset_wNewAUG(args.labeled_json, img_size=img_size, out_size=out_size)
    # augmentations = True
    # new_aug = True
    # val_dataset = COCOBinaryMaskDataset(args.val_json, img_size=img_size, out_size=out_size)

    labeled_sampler = DistributedSampler(labeled_dataset)
    #val_sampler = DistributedSampler(val_dataset, shuffle=False)

    persistent_workers = False
    prefetch_factor = 1
    if args.ade20k or args.pascal:
        from torch.utils.data._utils.collate import default_collate
        def custom_collate(batch):
            collated = default_collate([{k: v for k, v in sample.items() if k != 'text'} for sample in batch])
            collated["text"] = [sample["text"] for sample in batch]  # keeps list of strings per sample
            return collated
        labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, sampler=labeled_sampler,
                                    num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers, 
                                    prefetch_factor=prefetch_factor, collate_fn=custom_collate)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers, 
                            prefetch_factor=prefetch_factor, collate_fn=custom_collate)

    elif args.camouflaged:
        labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, sampler=labeled_sampler,
                                    num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers, 
                                    prefetch_factor=prefetch_factor)
        # build a dict of val loaders
        val_samplers = {k: DistributedSampler(v, shuffle=False) for k, v in val_datasets.items()}
        val_loaders  = {
            k: DataLoader(
                v, batch_size=args.batch_size, sampler=val_samplers[k],
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
            )
            for k, v in val_datasets.items()
        }
    else:
        labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, sampler=labeled_sampler,
                                    num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers, 
                                    prefetch_factor=prefetch_factor)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    # ===  Initialization ===
    logger.info("Initializing models...")


    if args.simple_load:
        if args.sam_cross:
            logger.info("Using cross attention IE")
            trainable_sam = initialize_sam(device, IE_type='Parallel_Text_Cross', MD_type='Vanilla', freeze=True)
        elif args.parallel_sim:
            logger.info("Using Parallel Sim for IE")
            trainable_sam = initialize_sam(device, IE_type='Parallel_Sim', MD_type='Vanilla', freeze=True)
        else:
            trainable_sam = initialize_sam(device, IE_type='Parallel_Text', MD_type='Vanilla', freeze=True)
        #trainable_sam = load_sam(trainable_sam,args.supervised_checkpoint)
    else:
        if args.sam_cross:
            logger.info("Using cross attention IE")
            trainable_sam = initialize_sam(device, IE_type='Parallel_Text_Cross', MD_type='Vanilla', freeze=True, sam_ckpt=args.sam_ckpt)
        elif args.parallel_sim:
            logger.info("Using Parallel Sim for IE")
            trainable_sam = initialize_sam(device, IE_type='Parallel_Sim', MD_type='Vanilla', freeze=True, sam_ckpt=args.sam_ckpt)
        elif args.text_vis:
            logger.info("Using vision embeddings in IE")
            trainable_sam = initialize_sam(device, IE_type='Parallel_Text_Vis', MD_type='Vanilla', freeze=True, sam_ckpt=args.sam_ckpt, vit_type='vit_b')
        elif args.text_vis_cross:
            trainable_sam = initialize_sam(device, IE_type='Parallel_Text_Vis_Cross', MD_type='Vanilla', freeze=True, sam_ckpt=args.sam_ckpt)
        elif args.no_text:
            trainable_sam = initialize_sam(device, IE_type=args.SAM_IE, MD_type='Vanilla', freeze=True, sam_ckpt=args.sam_ckpt)
        else:
            trainable_sam = initialize_sam(device, IE_type='Parallel_Text', MD_type='Vanilla', freeze=True, sam_ckpt=args.sam_ckpt)
        #trainable_sam.load_state_dict(supervised_ckpt['model_state_dict'])

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in trainable_sam.parameters() )
        logger.info(f"SAM total parameters: {total_params:,}")

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in trainable_sam.image_encoder.parameters())
        logger.info(f"SAM IE parameters: {total_params:,}")



    if args.sam_cross:
        for name,param in trainable_sam.image_encoder.cross_blocks.named_parameters():
            param.requires_grad = True

    # for name,param in trainable_sam.image_encoder.cross_blocks.named_parameters():
    #     print(name,param.requires_grad)


    clip_model = initialize_clip(device, size=args.clip_crop_size, type=args.clip_type)

    if args.clip_logscale:
        import torch.nn as nn
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    else:
        logit_scale = None

    if args.load_clip:
        logger.info("Loading CLIP pretrained checkpoint")
        clip_ckpt = torch.load(args.clip_ckpt)
        clip_model.load_state_dict(clip_ckpt['clip_model'])
        if logit_scale is not None and 'logit_scale' in clip_ckpt:
            logit_scale.data = clip_ckpt['logit_scale']
    # if not args.finetune_clip:
    #     for param in clip_model.parameters():
    #         param.requires_grad = False
    
    # else:
    #     for name,param in clip_model.visual.transformer.resblocks.named_parameters():
    #         if 'attn' in name:    
    #             param.requires_grad = True

    for param in clip_model.parameters():
        param.requires_grad = False

    if args.finetune_clip:
        #logger.info(f"Type of positional_embedding: {type(clip_model.positional_embedding)}")
        
        clip_model.positional_embedding.requires_grad = True ### This is for text embeddings
        #clip_model.visual.positional_embedding.requires_grad = True  ### This is for visual embeddings

        # number of transformer blocks
        resblocks = clip_model.visual.transformer.resblocks
        num_blocks = len(resblocks)

        # how many blocks from the end you want to fine-tune
        K = 12
        trainable_blocks = range(num_blocks - K, num_blocks)

        for i in trainable_blocks:
            block = resblocks[i]
            for name, param in block.named_parameters():
                if "attn" in name:
                    param.requires_grad = True


        # for name,param in clip_model.visual.transformer.resblocks.named_parameters():
        #     if 'attn' in name:    
        #         param.requires_grad = True
    
    if args.clip_decoder_head:
        clip_decoder = CLIP_Decoder_Head(text_in_channels=512, text_channels=128, channels=128, 
                             conv1_ksize=7, num_layers=2, num_heads=4, up_channels=(64, 32, 16),
                             skip_channels=(0, 0))
        clip_decoder = clip_decoder.to(device)
        clip_decoder = torch.nn.parallel.DistributedDataParallel(clip_decoder, device_ids=[local_rank],find_unused_parameters=True, static_graph=True)
    else: 
        clip_decoder = None
    

    # --- Param totals & percentages ---
    sam_total, sam_train = count_params(trainable_sam)
    clip_total, clip_train = count_params(clip_model)

    pct_sam = format_param_pct(sam_train, sam_total)
    pct_clip = format_param_pct(clip_train, clip_total)
    pct_both = format_param_pct(sam_train + clip_train, sam_total + clip_total)

    # --- Trainable block counts (0..12 typically) ---
    sam_ie_trainable_blocks = count_trainable_blocks_sam_image_encoder(trainable_sam)
    clip_v_trainable_blocks = count_trainable_blocks_clip_vision(clip_model)

    # If block lists weren't found, show "N/A"
    sam_ie_str = str(sam_ie_trainable_blocks) if sam_ie_trainable_blocks is not None else "N/A"
    clip_v_str = str(clip_v_trainable_blocks) if clip_v_trainable_blocks is not None else "N/A"
    if dist.get_rank() == 0:
        logger.info(
            f"[Trainability] SAM IE trainable blocks: {sam_ie_str} | "
            f"CLIP vision trainable blocks: {clip_v_str}"
        )

        logger.info(
            f"[Trainability] SAM trainable params: {sam_train:,}/{sam_total:,} ({pct_sam:.3f}%) | "
            f"CLIP trainable params: {clip_train:,}/{clip_total:,} ({pct_clip:.3f}%) | "
            f"TOTAL trainable params: {sam_train + clip_train:,}/{sam_total + clip_total:,} ({pct_both:.3f}%)"
        )

        
    # DDP wrapping
    trainable_sam = torch.nn.parallel.DistributedDataParallel(trainable_sam, device_ids=[local_rank], find_unused_parameters=True)
    if args.mask_prompts:
        for name,param in trainable_sam.module.prompt_encoder.mask_downscaling.named_parameters():
            param.requires_grad = True

    if args.finetune_PE:
        for name,param in trainable_sam.module.prompt_encoder.named_parameters():
            param.requires_grad = True

    if args.finetune_neck:
        for name,param in trainable_sam.module.image_encoder.neck.named_parameters():
            param.requires_grad = True

    if args.finetune_clip:
        clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, device_ids=[local_rank],find_unused_parameters=True, static_graph=True)

    # === Training Config ===
    save_ext = f'_v{args.version}_split_{args.split}_Supervised_{dataset}_IEParaTextVisMDVanilla_lr{args.lr_IE}_batchsize{args.batch_size}'
    

    is_vanilla_sam = False
    SAM_text = False if args.no_text else True
    features_input = False
    final_div_factor = 5
    pct_start = 0.2
    div_factor=5


    logger.info("Using the following options:")
    options = f"points_from_gt_train: {args.points_from_gt_train} \n \
                points_from_gt_eval: {args.points_from_gt_eval} \n \
                simple_loss: {args.simple_loss} \n \
                total_epochs: {args.total_epochs} \n \
                finetune_clip: {args.finetune_clip} \n \
                clip_crop_size: {args.clip_crop_size} \n \
                include_backgrounds: {args.include_backgrounds} \n \
                use_scheduler: {args.use_scheduler} \n \
                lr_IE: {args.lr_IE} \n \
                lr_MD: {args.lr_MD} \n \
                lr_PE: {args.lr_PE} \n \
                lr_clip: {args.lr_clip} \n \
                prompt_ensemble: {args.prompt_ensemble} \n \
                sim_func_surgery: {args.sim_func_surgery} \n \
                norm_dim: {args.norm_dim} \n \
                pct_start: {pct_start} \n \
                div_factor: {div_factor} \n \
                final_div_factor: {final_div_factor} \n \
                weight_decay: {args.weight_decay} \n \
                Augmentations? {augmentations} \n \
                New Aug? {new_aug} \n \
                Fine tune whole PE? {args.finetune_PE} \n \
                Finetune neck? {args.finetune_neck} \n \
                Text_Vis? {args.text_vis} \n \
                Text Vis Cross? {args.text_vis_cross} \n \
                Clip decoder head? {args.clip_decoder_head} \n \
                lr_clip_dec_head: {args.lr_clip_dec_head} \n \
                out_size_512: {args.out_size_512} \n \
                dataset: {dataset} \n \
                clip logit scale?: {args.clip_logscale} \n \
                load clip? {args.load_clip} \n \
                bce weight? {args.bce_weight} \n \
                dice_weight? {args.dice_weight} \n \
                iou_weight? {args.iou_weight} \n \
                SAM IE: {args.SAM_IE} \n \
                clip type: {args.clip_type} \n \
                Using soft prompts: {args.use_soft_prompts} \n \
                Sigmoid soft prompts: {args.soft_prompt_sigmoid} \n \
                SAM IE trainable blocks: {sam_ie_str} \n \
                CLIP vision trainable blocks: {clip_v_str} \n \
                SAM trainable params pct: {pct_sam:.3f}% ({sam_train:,}/{sam_total:,}) \n \
                CLIP trainable params pct: {pct_clip:.3f}% ({clip_train:,}/{clip_total:,}) \n \
                TOTAL trainable params pct: {pct_both:.3f}% ({sam_train + clip_train:,}/{sam_total + clip_total:,}) \n \
                "
    logger.info(options)
 
    # === Start Training ===
    logger.info("Starting supervised training...")

    if args.text_vis_cross:
        args.text_vis = True
        args.sam_cross = True

    
    
    train_supervised_final_for_scripts_DDP(
    trainable_sam, clip_model, labeled_loader, val_loaders if args.camouflaged else val_loader,
    classes=classes, clip_crop_size=args.clip_crop_size, total_epochs=args.total_epochs,
    epochs=args.epochs, save_path=args.save_path, save_ext=save_ext, logger=logger, split=args.split,
    SAM_text=SAM_text, has_mask_prompt=args.mask_prompts, points_from_gt_train=args.points_from_gt_train, points_from_gt_eval=args.points_from_gt_eval, simple_loss=args.simple_loss,
    include_backgrounds=args.include_backgrounds, features_input=features_input, is_vanilla_sam=is_vanilla_sam,
    lr_IE=args.lr_IE, lr_MD=args.lr_MD, lr_clip=args.lr_clip, lr_PE=args.lr_PE, resume_path=args.resume_checkpoint, ignore_wandb=args.ignore_wandb, 
    skip_init_eval=args.skip_init_eval, finetune_clip=args.finetune_clip, use_scheduler=args.use_scheduler, prompt_ensemble=args.prompt_ensemble, 
    sim_func_surgery=args.sim_func_surgery, norm_dim=args.norm_dim, iou_loss=args.iou_loss, structure_loss=args.structure_loss,
    enhanced_structure_loss=args.enhanced_structure_loss, options=options,
    final_div_factor=final_div_factor, pct_start=pct_start, div_factor=div_factor, weight_decay=args.weight_decay, 
    sam_cross=args.sam_cross, lr_cr = args.lr_cr, parallel_sim = args.parallel_sim, finetune_neck=args.finetune_neck, text_vis=args.text_vis,
    clip_decoder_head=args.clip_decoder_head, clip_decoder=clip_decoder, lr_clip_dec_head=args.lr_clip_dec_head, out_size_512=args.out_size_512, 
    dataset=dataset, logit_scale=logit_scale, use_mae=args.use_mae, camo_multi_val=args.camouflaged, 
    bce_weight=args.bce_weight, dice_weight=args.dice_weight, iou_weight=args.iou_weight, wandb_extension=args.wandb_extension, 
    use_soft_prompts=args.use_soft_prompts, soft_prompt_sigmoid=args.soft_prompt_sigmoid
)

    logger.info("Supervised training completed.")

def main():
    args = parse_args()
    if args.ade20k and args.pascal:
        raise ValueError("Only one of --ade20k or --pascal can be set to True. Please choose one.")
    if args.num_gpus > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # specify which gpu to use if using one gpu, default to gpu 0 if not specified
        local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
    #local_rank = 1
    main_worker(local_rank, args)


if __name__ == "__main__":
    main()
    dist.destroy_process_group()


# rsync -av \
#   --exclude='*.pth' \
#   --exclude='*.ckpt' \
#   --exclude='wandb/' \
#   --exclude='__pycache__/' \
#   --exclude='.git/' \
#   --exclude='.log/' \
#   --exclude='.json/' \
#   --exclude='.png/' \
#   --exclude='.jpg/' \
#   ./ \
#   ../SAM_PTx/

# rsync -av --dry-run \
#   --exclude='*.pth' \
#   --exclude='*.pt' \
#   --exclude='*.ckpt' \
#   --exclude='wandb/' \
#   --exclude='__pycache__/' \
#   --exclude='.git/' \
#   --exclude='*.log' \
#   --exclude='*.json' \
#   --exclude='*.png' \
#   --exclude='*.jpg' \
#   --exclude='*.jpeg' \
#   --exclude='*.xml' \
#   --exclude='*.jsonl' \
#   --exclude='*.xml' \
#   --exclude='*.npy' \
#   ./ ../SAM_PTx/