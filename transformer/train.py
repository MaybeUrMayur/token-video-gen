import os
import sys
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from transformer.model import VideoGPT
from transformer.config import get_transformer_args
from vqvae.model import VQVAE3D
from data.dataset import get_dataloader

def precompute_tokens(vqvae, dataloader, device, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path)
        
    print(f"Precomputing tokens and saving to {cache_path}...")
    vqvae.eval()
    all_tokens = []
    
    with torch.no_grad():
        for videos in tqdm(dataloader):
            videos = videos.to(device)
            indices = vqvae.encode_to_indices(videos) # (B, T, H, W)
            # Flatten spatial and temporal dimensions for autoregressive modeling
            indices = indices.view(indices.size(0), -1) # (B, T*H*W)
            all_tokens.append(indices.cpu())
            
    all_tokens = torch.cat(all_tokens, dim=0)
    torch.save(all_tokens, cache_path)
    return all_tokens

def train():
    args = get_transformer_args()
    
    if not args.no_wandb:
        import wandb
        wandb.init(project="token-video-gen", name="transformer_training", config=args)
        
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # 1. Load frozen VQ-VAE
    vqvae = VQVAE3D(
        in_channels=args.in_channels,
        hidden_dims=args.vq_hidden_dims,
        codebook_size=args.codebook_size,
        embed_dim=args.vq_embed_dim
    ).to(device)
    
    if os.path.exists(args.vqvae_ckpt):
        vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
        print(f"Loaded VQ-VAE from {args.vqvae_ckpt}")
    else:
        print(f"Warning: VQ-VAE checkpoint not found at {args.vqvae_ckpt}. Precomputing with untrained VQ-VAE!")
        
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
        
    # 2. Prepare Data (Precompute tokens)
    dataloader = get_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        resolution=args.resolution,
        clip_length=args.clip_length,
        max_clips=args.max_clips,
        num_workers=args.num_workers,
        shuffle=False # Shuffle manually during training
    )
    
    cache_dir = os.path.join(project_root, "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tokens_res{args.resolution}_clip{args.clip_length}.pt")
    
    tokens = precompute_tokens(vqvae, dataloader, device, cache_path) # (N, seq_len)
    seq_len = tokens.size(1)
    print(f"Max sequence length per clip: {seq_len}")
    
    # 3. Setup Transformer
    model = VideoGPT(
        vocab_size=args.codebook_size,
        max_seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler()
    
    # Create a simple dataset for the tokens
    token_dataset = torch.utils.data.TensorDataset(tokens)
    token_loader = torch.utils.data.DataLoader(token_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Train loop
    model.train()
    
    for epoch in range(args.epochs):
        loop = tqdm(token_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        total_loss = 0
        
        for (batch_tokens,) in loop:
            batch_tokens = batch_tokens.to(device)
            
            # Inputs: all but last token, Targets: all but first token
            x = batch_tokens[:, :-1]
            y = batch_tokens[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == "cuda":
                with autocast():
                    logits, loss = model(x, targets=y)
                    
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(token_loader)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "perplexity": perplexity
            })
            
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"transformer_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train()
