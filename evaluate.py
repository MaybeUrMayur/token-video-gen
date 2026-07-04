import os
import argparse
import torch
import torch.nn.functional as F
import imageio
import numpy as np
from tqdm import tqdm

from vqvae.model import VQVAE3D
from transformer.model import VideoGPT
from data.dataset import get_dataloader

def calculate_psnr(mse):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(2.0) - 10 * np.log10(mse) # Range is [-1, 1], so max diff is 2.0

def evaluate(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading models...")
    vqvae = VQVAE3D(
        in_channels=args.in_channels,
        hidden_dims=args.vq_hidden_dims,
        codebook_size=args.codebook_size,
        embed_dim=args.vq_embed_dim
    ).to(device)
    if os.path.exists(args.vqvae_ckpt):
        vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
    vqvae.eval()
    
    # Calculate seq_len
    dummy_input = torch.zeros(1, args.in_channels, args.clip_length, args.resolution, args.resolution).to(device)
    dummy_indices = vqvae.encode_to_indices(dummy_input)
    latent_shape = dummy_indices.shape[1:]
    seq_len = int(np.prod(latent_shape))
    
    transformer = VideoGPT(
        vocab_size=args.codebook_size,
        max_seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0
    ).to(device)
    if os.path.exists(args.transformer_ckpt):
        transformer.load_state_dict(torch.load(args.transformer_ckpt, map_location=device))
    transformer.eval()

    dataloader = get_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        resolution=args.resolution,
        clip_length=args.clip_length,
        max_clips=args.batch_size, # Only need a few for evaluation
        shuffle=False
    )
    
    print("Evaluating VQ-VAE and Transformer...")
    total_mse = 0
    total_perplexity = 0
    used_indices = set()
    
    real_clips = []
    recon_clips = []
    
    with torch.no_grad():
        for videos in dataloader:
            videos = videos.to(device)
            
            # VQ-VAE Eval
            reconstructed, vq_loss, perplexity, indices = vqvae(videos)
            mse = F.mse_loss(reconstructed, videos).item()
            total_mse += mse
            
            used_indices.update(indices.view(-1).tolist())
            
            if len(real_clips) == 0:
                real_clips = videos.cpu()
                recon_clips = reconstructed.cpu()
                
            # Transformer Eval
            flat_indices = indices.view(indices.size(0), -1)
            x = flat_indices[:, :-1]
            y = flat_indices[:, 1:]
            
            logits, loss = transformer(x, targets=y)
            total_perplexity += np.exp(loss.item())
            
    avg_mse = total_mse / len(dataloader)
    avg_psnr = calculate_psnr(avg_mse)
    avg_perp = total_perplexity / len(dataloader)
    codebook_usage = len(used_indices) / args.codebook_size * 100
    
    print("\n--- Evaluation Results ---")
    print(f"VQ-VAE Reconstruction MSE:  {avg_mse:.4f}")
    print(f"VQ-VAE Reconstruction PSNR: {avg_psnr:.2f} dB")
    print(f"VQ-VAE Codebook Usage:      {codebook_usage:.2f}%")
    print(f"Transformer Perplexity:     {avg_perp:.4f}")
    
    # Generate novel clips
    print("\nGenerating novel clips for results grid...")
    start_tokens = torch.randint(0, args.codebook_size, (min(4, args.batch_size), 1)).to(device)
    with torch.no_grad():
        generated_tokens = transformer.generate(start_tokens, max_new_tokens=seq_len-1, temperature=1.0, top_k=100)
        generated_indices = generated_tokens.view(start_tokens.size(0), *latent_shape)
        generated_clips = vqvae.decode_from_indices(generated_indices).cpu()
        
    # Build Results Grid (Real vs Recon vs Generated)
    # We will pick 4 samples
    num_samples = min(4, args.batch_size)
    
    def process_clip(clip):
        # to [0, 255] (T, H, W, C)
        clip_np = ((clip.numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        clip_np = np.transpose(clip_np, (1, 2, 3, 0))
        if clip_np.shape[-1] == 1:
            clip_np = np.repeat(clip_np, 3, axis=-1) # Convert to RGB for grid
        return clip_np
        
    grid_frames = []
    T = args.clip_length
    for t in range(T):
        rows = []
        for i in range(num_samples):
            real_f = process_clip(real_clips[i:i+1])[0, t]
            recon_f = process_clip(recon_clips[i:i+1])[0, t]
            gen_f = process_clip(generated_clips[i:i+1])[0, t]
            
            # Combine horizontally with a small white border
            border = np.ones((args.resolution, 2, 3), dtype=np.uint8) * 255
            row = np.concatenate([real_f, border, recon_f, border, gen_f], axis=1)
            rows.append(row)
            
        # Combine vertically with a border
        h_border = np.ones((2, row.shape[1], 3), dtype=np.uint8) * 255
        full_frame = rows[0]
        for row in rows[1:]:
            full_frame = np.concatenate([full_frame, h_border, row], axis=0)
            
        grid_frames.append(full_frame)
        
    grid_path = os.path.join(args.output_dir, "results_grid.gif")
    imageio.mimsave(grid_path, grid_frames, fps=4)
    print(f"Saved results grid to {grid_path}")
    print("Columns: Real | Reconstructed | Generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/mnist_test_seq.npy")
    parser.add_argument("--vqvae_ckpt", type=str, default="checkpoints/vqvae/vqvae_epoch_100.pt")
    parser.add_argument("--transformer_ckpt", type=str, default="checkpoints/transformer/transformer_epoch_100.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Model config
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--vq_embed_dim", type=int, default=64)
    parser.add_argument("--vq_hidden_dims", type=int, nargs='+', default=[64, 128])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    
    args = parser.parse_args()
    evaluate(args)
