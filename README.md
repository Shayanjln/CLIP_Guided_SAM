# CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning for Promptable Segmentation

Shayan Jalilian, Abdul Bais — University of Regina

## Paper

Preprint available on arXiv: [CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning for Promptable Segmentation](https://arxiv.org/abs/2605.24807)

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





---

## Operating Modes

**Manual mode** — interactive segmentation using both text and spatial prompts (points/boxes).
Text conditions the encoder; spatial prompts are passed through SAM's prompt encoder as usual.

**Semi-Automatic (text-only) mode** — concept-specific segmentation using only textual input,
without spatial prompts. Suitable for applications requiring automated, prompt-free inference.

---

## Training

```bash
bash train.sh
```


## Results on ADE20K, 1/16 split
![Results on ADE20K 1/16](figures/CLIP_Guided_SAM_Visualization_ADE_16.png)


---

## Acknowledgements

This work builds on [SAM](https://github.com/facebookresearch/segment-anything),
[CLIP](https://github.com/openai/CLIP),
[CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery),
and [SU-SAM](https://github.com/zongzi13545329/SimAda).

