# CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning for Promptable Segmentation

**Submitted to ECCV 2026 (under review)**

Shayan Jalilian, Abdul Bais — University of Regina

---

## Overview

CLIP-Guided SAM is a parameter-efficient segmentation framework that conditions the
[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
with semantic signals from [CLIP](https://github.com/openai/CLIP) via **internal semantic
conditioning** — injecting CLIP-derived text, vision, and similarity features directly
into SAM's image encoder through lightweight multi-modal adapters.

Unlike prior vision-language + SAM pipelines that use CLIP only to generate spatial
prompts as a separate external stage, CLIP-Guided SAM integrates semantic information
into SAM's internal feature representations, preserving SAM's original promptable interface
while enabling concept-specific segmentation.

Key properties:
- **Internal conditioning:** CLIP text, vision, and similarity features injected into SAM's encoder via multi-modal adapters
- **Parameter-efficient:** only adapters and mask decoder are fine-tuned; SAM backbone stays frozen
- **Two operating modes:** Manual (text + spatial prompts) and Semi-Automatic (text-only)
- **Low labeled-data settings:** designed for general benchmarks and specialized downstream tasks
- **Train–test prompt consistency:** a design principle shown to be critical for robustness


<!-- ---

## Results

Evaluated against multiple baselines including SAM+PEFT (without semantic conditioning),
vision-language + SAM pipelines, SAM 3, and semi-supervised segmentation methods.
CLIP-Guided SAM achieves superior or competitive performance while remaining
parameter-efficient in both training and deployment.

Benchmarks include semantic segmentation and camouflaged object detection datasets.

--- -->

<!-- ## Repository Structure

```
CLIP_Guided_SAM/
├── SimAda/models/sam/modeling/
│   ├── image_encoder_para_text_vis.py   # Primary CLIP-Guided SAM implementation
│   ├── image_encoder_para_text.py       # SAM-PTx baseline (text-only conditioning)
│   ├── image_encoder_mix.py             # SU-SAM baseline
│   ├── image_encoder_noadpt_text_vis.py # Ablation: no adapter variant
│   └── ...                              # Supporting modules (mask decoder, prompt encoder, etc.)
├── CLIP_Surgery/                        # CLIP Surgery utilities (text + vision encoder)
├── CLIP_SAM_Utils_Final_MultiGPU.py    # Shared utilities for multi-GPU runs
├── Training_Functions_multi_gpu.py
├── datasets.py                          # Dataset loading
├── SupervisedTrainingScript_DDP_noSLURM.py  # Training entry point
├── Supervised_DDP_NoSLURM.sh           # Launch script (no SLURM)
```

> Note: Other files within `SimAda/` are retained from the original
> [SU-SAM](https://github.com/zongzi13545329/SimAda) repository but are not used in this project.

--- -->

<!-- ## Setup

Download SAM checkpoints from the
[official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and place them in the root directory.

--- -->

## Training

```bash
bash train.sh
```

---

## Operating Modes

**Manual mode** — interactive segmentation using both text and spatial prompts (points/boxes).
Text conditions the encoder; spatial prompts are passed through SAM's prompt encoder as usual.

**Semi-Automatic (text-only) mode** — concept-specific segmentation using only textual input,
without spatial prompts. Suitable for applications requiring automated, prompt-free inference.

---

## Citation

```bibtex
@inproceedings{jalilian2026clipguidedsam,
  title     = {CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning
               for Promptable Segmentation},
  author    = {Jalilian, Shayan and Bais, Abdul},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

This work builds on [SAM](https://github.com/facebookresearch/segment-anything),
[CLIP](https://github.com/openai/CLIP),
[CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery),
and [SU-SAM](https://github.com/zongzi13545329/SimAda).

