# CLIP from Scratch

A full PyTorch implementation of **Contrastive Language-Image Pretraining (CLIP)** trained on CC3M (~2.9M image-text pairs) using 2× NVIDIA A5000 GPUs. Achieves **47.68% zero-shot** and **78.76% linear probe** accuracy on CIFAR-10.

---

## Overview

This project reimplements the CLIP architecture from the ground up — ViT-B/16 image encoder, GPT-2-style causal text encoder, symmetric InfoNCE contrastive loss — and trains it end-to-end on real web-scale data using distributed training across 2 GPUs.

Every component is implemented from scratch: patch embedding, multi-head self-attention, causal masking, contrastive loss with cross-GPU all_gather, mixed precision training, and cosine LR scheduling with warmup.

---

## Architecture

```
Image  →  ViT-B/16 Encoder  →  Linear(768, 512)  →  L2 Norm  →  512-d embedding
                                                                          ↓
                                                               InfoNCE Contrastive Loss
                                                                          ↑
Text   →  GPT-2 Encoder     →  Linear(512, 512)  →  L2 Norm  →  512-d embedding
```

### Image Encoder — ViT-B/16

| Parameter | Value |
|---|---|
| Input size | 224 × 224 |
| Patch size | 16 × 16 |
| Number of patches | 196 |
| Model dim | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| MLP hidden dim | 3072 |
| Activation | GELU |
| Dropout | 0.1 |

The image is split into 196 non-overlapping 16×16 patches. Each patch is projected to 768-d via a Conv2d layer (equivalent to flatten + linear). A learnable `[CLS]` token is prepended and 1D positional embeddings are added. After 12 Pre-LN Transformer blocks, the `[CLS]` token output is passed through a linear projection head to 512-d.

### Text Encoder — GPT-2 Style

| Parameter | Value |
|---|---|
| Vocabulary size | 50,257 (BPE) |
| Context length | 77 tokens |
| Model dim | 512 |
| Attention heads | 8 |
| Transformer layers | 12 |
| MLP hidden dim | 2048 |
| Activation | GELU |
| Masking | Causal (upper triangular) |

Uses `tiktoken` GPT-2 BPE tokenizer. The `<|endoftext|>` token (id 50256) serves as both SOS and EOS. Text is formatted as `[EOS] tokens... [EOS] [EOS] [EOS]...` with EOS used for padding. The embedding at the EOS position is used as the text representation, projected to 512-d.

### Contrastive Loss

Symmetric InfoNCE loss over the full gathered batch across both GPUs:

```
L = (L_image→text + L_text→image) / 2
```

where `L_image→text = CrossEntropy(τ · I · Tᵀ, labels)` and labels are the diagonal indices. Temperature `τ` is a learnable parameter initialized to `exp(log(1/0.07)) = 14.3`.

---

## Training Details

| Parameter | Value |
|---|---|
| Dataset | CC3M (~2.9M pairs) |
| Epochs | 15 |
| Batch size per GPU | 160 |
| Effective batch size | 640 (160 × 2 GPUs × 2 accumulation steps) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| β1, β2 | 0.9, 0.98 |
| Weight decay | 0.2 |
| LR schedule | Cosine decay with linear warmup |
| Warmup steps | 10% of total steps |
| Min LR | 10% of peak LR |
| Gradient clipping | 1.0 |
| Mixed precision | bf16 (torch.autocast) |
| Hardware | 2× NVIDIA A5000 (24GB each) |
| DDP backend | NCCL |

### Key implementation decisions

**Cross-GPU all_gather with gradients:** Used `torch.distributed.nn.all_gather` which natively supports gradient flow, avoiding the need for a custom autograd function. Gradients flow back to each GPU's local embeddings correctly.

**Gradient accumulation with DDP:** `model.no_sync()` skips the DDP gradient sync on accumulation steps. `all_gather` is called only on sync steps to avoid inconsistency between accumulated gradients and gathered embeddings.

---

## Results

### Zero-Shot Classification

Zero-shot classification uses the model directly without any fine-tuning. Class names are encoded as text prompts (`"a photo of a {class}"`) and the image is matched to the closest class embedding by cosine similarity.

| Dataset | Classes | Zero-Shot Accuracy |
|---|---|---|
| CIFAR-10 | 10 | **47.60%** |
| CIFAR-100 | 100 | **23.51%** |
| Food101 | 101 | **6.61%** |
| Oxford-IIIT Pet | 37 | **5.81%** |

### Linear Probe Classification

Frozen image embeddings are extracted and a logistic regression classifier is trained on top. No part of the CLIP model is fine-tuned.

| Dataset | Classes | Linear Probe Accuracy | Gain over Zero-Shot |
|---|---|---|---|
| CIFAR-10 | 10 | **78.76%** | +31.08% |
| CIFAR-100 | 100 | **56.22%** | +32.54% |
| Food101 | 101 | **44.21%** | +35.87% |
| Oxford-IIIT Pet | 37 | **49.36%** | +43.58% |

The 78.76% linear probe on CIFAR-10 — achieved with **zero labeled training data** — is comparable to a lightly supervised CNN, demonstrating that the ViT image encoder learns rich, transferable visual representations purely from noisy web captions.

### MS-COCO Retrieval (5K test set)

| Metric | Text→Image | Image→Text |
|---|---|---|
| R@1 | 8.82% | 5.84% |
| R@5 | 22.38% | 17.90% |
| R@10 | 31.48% | 27.06% |

The model retrieves the correct image in the top 10 results 32% of the time across a pool of 5,000 images — 160× better than random chance. T2I outperforming I2T reflects the stronger capacity of the ViT-B/16 image encoder relative to the 512-dim text encoder.


## Analysis

### Why zero-shot varies across datasets

Zero-shot accuracy is directly tied to how frequently class names appear in CC3M alt-text captions:

- **CIFAR-10 (47.60%)** — Common objects (airplane, car, dog) appear frequently in web alt-text
- **CIFAR-100 (23.51%)** — Mixed — common superclasses work, fine-grained subclasses don't
- **Food101 (6.61%)** — Specific dish names (bibimbap, croque madame) are rare in CC3M
- **Oxford Pet (5.81%)** — Specific breed names (Abyssinian, Basset Hound) essentially absent from CC3M

This matches findings in the original CLIP paper on training data distribution bias.

### Image vs text encoder balance

T2I retrieval outperforming I2T reveals an encoder capacity imbalance — the ViT-B/16 image encoder (768-dim, 86M params) is larger than the text encoder (512-dim). The image embeddings are more discriminative than text embeddings, causing the model to perform better when the query is text (finding a distinctive image) than when the query is an image (finding a distinctive caption among 5000 similar ones).

---

## Installation

```bash
git clone https://github.com/your-username/clip-from-scratch
cd clip-from-scratch
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.0.0
tiktoken>=0.5.0
numpy
Pillow
scikit-learn   # for linear probe
pycocotools    # for COCO retrieval eval
tensorboard    # for logging
```

---

## Usage

### Training

For single GPU use **clip.ipynb**

```bash
python -m torch.distributed.run --nproc_per_node=2   clip_ddp.py
```

Edit `TrainingConfiguration` in `clip_ddp.py` to set your dataset path and hyperparameters.

### Evaluation

For Evaluations use **evaluations.ipynb** 

---

## References

- Radford et al. — [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (OpenAI CLIP)
- Dosovitskiy et al. — [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT)
- Sharma et al. — [Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset](https://aclanthology.org/P18-1238/) (CC3M)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — open source CLIP reproduction for reference

---

## Related Projects

- [Transformer from Scratch (EN-HI Translation)](https://github.com/CloudedLeopard17/English_Hindi_Translation)
- [Vision Transformer (ViT) on Food101 — From Scratch](https://github.com/CloudedLeopard17/ViT)
