# Tokenized Video Generation

This repository contains a small-scale tokenized video generation pipeline built from scratch in PyTorch. It demonstrates the core principles behind state-of-the-art video generation models like **VideoGPT** and **VideoPoet**.

## Architecture Overview

The pipeline consists of two main components:
1. **VQ-VAE (Tokenizer)**: Compresses raw video clips into a discrete grid of latent tokens.
2. **Autoregressive Transformer**: A decoder-only language model that learns to predict the next video token in the sequence.

```text
  [Raw Video Clip]
         │
         ▼
  ┌──────────────┐
  │ VQ-VAE       │
  │ Encoder      │ (3D ResNet)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Vector       │
  │ Quantizer    │ (Codebook Lookup)
  └──────┬───────┘
         ▼
  [Discrete Tokens] ────► ┌──────────────┐
                          │ Transformer  │ (Next-Token Prediction)
  [Generated Tokens] ◄────│ Decoder      │
         │                └──────────────┘
         ▼
  ┌──────────────┐
  │ VQ-VAE       │
  │ Decoder      │ (3D Transposed CNN)
  └──────┬───────┘
         ▼
  [Generated Clip]
```

## How this relates to VideoPoet / VideoGPT
This pipeline is architecturally identical to the foundation of **VideoGPT** and **VideoPoet** on a smaller scale. 
- **VideoGPT** introduced using VQ-VAE to tokenize video spatio-temporally, followed by an autoregressive transformer to model the discrete latent space. 
- **VideoPoet** extends this paradigm by training a massive language model (LLM) on these video tokens alongside text and audio tokens, allowing for multi-modal conditional generation using standard LLM training objectives (next-token prediction). 
This repo implements the exact same "tokenize -> autoregressive language model -> decode" loop, scaled down for rapid iteration on consumer hardware.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Data
Downloads the Moving MNIST dataset (`mnist_test_seq.npy`):
```bash
python data/download_moving_mnist.py
```

### 2. Train VQ-VAE
Trains the video tokenizer. Includes AMP and logs to wandb by default.
```bash
python vqvae/train.py --epochs 100 --batch_size 16 --resolution 32
```
*Note: Check `outputs/vqvae_recon/` for reconstructed `.gif` samples during training.*

### 3. Train Transformer
Caches tokens from the frozen VQ-VAE and trains the autoregressive model.
```bash
python transformer/train.py --epochs 100 --batch_size 32
```

### 4. Generate
Samples novel token sequences and decodes them to `.gif`:
```bash
python generate.py --batch_size 16 --temperature 1.0 --top_k 100
```
*Outputs are saved to `outputs/generated/`.*

### 5. Evaluate
Computes VQ-VAE reconstruction MSE/PSNR, Transformer perplexity, and generates a side-by-side results grid.
```bash
python evaluate.py
```

## Results

*(After running evaluate.py, the results grid will be saved here)*
![Results Grid](outputs/results_grid.gif)

Columns: **Real Clip** | **VQ-VAE Reconstruction** | **Transformer Generated (From Scratch)**
