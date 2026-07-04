import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import imageio
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from vqvae.model import VQVAE3D
from vqvae.config import get_vqvae_args
from data.dataset import get_dataloader

def save_reconstruction_gif(original, recon, filepath):
    # original, recon shape: (C, T, H, W) in [-1, 1]
    # Convert to [0, 255] (T, H, W, C)
    orig_np = ((original.detach().cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    recon_np = ((recon.detach().cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    
    orig_np = np.transpose(orig_np, (1, 2, 3, 0)) # (T, H, W, C)
    recon_np = np.transpose(recon_np, (1, 2, 3, 0))
    
    # Repeat grayscale to RGB for imageio just in case, or keep as is. imageio handles (H,W,1) fine, but let's drop C if C=1
    if orig_np.shape[-1] == 1:
        orig_np = orig_np.squeeze(-1)
        recon_np = recon_np.squeeze(-1)
        
    # Concatenate side by side: (T, H, W*2)
    frames = np.concatenate([orig_np, recon_np], axis=2)
    
    # Save as gif
    imageio.mimsave(filepath, frames, fps=4)

def train():
    args = get_vqvae_args()
    
    if not args.no_wandb:
        import wandb
        wandb.init(project="token-video-gen", name="vqvae_training", config=args)
        
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.recon_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    model = VQVAE3D(
        in_channels=args.in_channels,
        hidden_dims=args.hidden_dims,
        codebook_size=args.codebook_size,
        embed_dim=args.embed_dim,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    dataloader = get_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        resolution=args.resolution,
        clip_length=args.clip_length,
        max_clips=args.max_clips,
        num_workers=args.num_workers
    )
    
    model.train()
    
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        
        # To track codebook usage
        used_indices = set()
        
        for batch_idx, videos in enumerate(loop):
            videos = videos.to(device) # (B, C, T, H, W)
            
            optimizer.zero_grad(set_to_none=True)
            
            if device.type == "cuda":
                with autocast():
                    reconstructed, vq_loss, perplexity, indices = model(videos)
                    recon_loss = F.mse_loss(reconstructed, videos)
                    loss = recon_loss + vq_loss
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed, vq_loss, perplexity, indices = model(videos)
                recon_loss = F.mse_loss(reconstructed, videos)
                loss = recon_loss + vq_loss
                
                loss.backward()
                optimizer.step()
            
            used_indices.update(indices.view(-1).tolist())
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            
            loop.set_postfix(loss=loss.item(), recon=recon_loss.item(), vq=vq_loss.item())
            
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_vq = total_vq_loss / len(dataloader)
        avg_perp = total_perplexity / len(dataloader)
        codebook_usage = len(used_indices) / args.codebook_size * 100
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | Codebook Usage: {codebook_usage:.2f}%")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "recon_loss": avg_recon,
                "vq_loss": avg_vq,
                "perplexity": avg_perp,
                "codebook_usage_percent": codebook_usage
            })
            
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"vqvae_epoch_{epoch+1}.pt"))
            
            # Save reconstruction sample (first item in last batch)
            model.eval()
            with torch.no_grad():
                sample_video = videos[0:1] # (1, C, T, H, W)
                sample_recon, _, _, _ = model(sample_video)
                gif_path = os.path.join(args.recon_dir, f"recon_epoch_{epoch+1}.gif")
                save_reconstruction_gif(sample_video[0], sample_recon[0], gif_path)
            model.train()

if __name__ == "__main__":
    train()
