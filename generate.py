import os
import argparse
import torch
import imageio
import numpy as np
from tqdm import tqdm

from vqvae.model import VQVAE3D
from transformer.model import VideoGPT

def generate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # 1. Load VQ-VAE
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
        print(f"Error: VQ-VAE checkpoint not found at {args.vqvae_ckpt}")
        return
        
    vqvae.eval()
    
    # 2. Determine sequence length from latent grid shape
    # For Moving MNIST: clip_length=8, resolution=32, stride=2 temporally and spatially.
    # We used 3D Convs. Encoder: 2 layers of stride 2.
    # T -> T/4? Wait, our encoder has len(hidden_dims) = 2.
    # Spatial: 32 -> 16 -> 8
    # Temporal: 8 -> 4 -> 2
    # Grid shape: (2, 8, 8) => 128 tokens
    # Let's derive it by passing a dummy tensor
    dummy_input = torch.zeros(1, args.in_channels, args.clip_length, args.resolution, args.resolution).to(device)
    with torch.no_grad():
        dummy_indices = vqvae.encode_to_indices(dummy_input)
    latent_shape = dummy_indices.shape[1:] # (T', H', W')
    seq_len = int(np.prod(latent_shape))
    print(f"Latent grid shape: {latent_shape}, Sequence length: {seq_len}")
    
    # 3. Load Transformer
    model = VideoGPT(
        vocab_size=args.codebook_size,
        max_seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.0
    ).to(device)
    
    if os.path.exists(args.transformer_ckpt):
        model.load_state_dict(torch.load(args.transformer_ckpt, map_location=device))
        print(f"Loaded Transformer from {args.transformer_ckpt}")
    else:
        print(f"Error: Transformer checkpoint not found at {args.transformer_ckpt}")
        return
        
    model.eval()
    
    # 4. Generate
    print(f"Generating {args.batch_size} samples...")
    # Start with a random token or a predefined start token. 
    # Since we didn't train with a start token, we can just randomly sample the first token,
    # or start with a batch of randomly selected tokens from the codebook.
    start_tokens = torch.randint(0, args.codebook_size, (args.batch_size, 1)).to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            start_tokens,
            max_new_tokens=seq_len - 1,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        # Reshape to latent grid
        generated_indices = generated_tokens.view(args.batch_size, *latent_shape)
        
        # Decode
        reconstructed_clips = vqvae.decode_from_indices(generated_indices) # (B, C, T, H, W)
        
    # 5. Save
    for i in range(args.batch_size):
        clip = reconstructed_clips[i] # (C, T, H, W)
        # to [0, 255] (T, H, W, C)
        clip_np = ((clip.cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        clip_np = np.transpose(clip_np, (1, 2, 3, 0))
        
        if clip_np.shape[-1] == 1:
            clip_np = clip_np.squeeze(-1)
            
        save_path = os.path.join(args.output_dir, f"sample_{i}.gif")
        imageio.mimsave(save_path, clip_np, fps=4)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", type=str, default="checkpoints/vqvae/vqvae_epoch_100.pt")
    parser.add_argument("--transformer_ckpt", type=str, default="checkpoints/transformer/transformer_epoch_100.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/generated")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configuration (should match training)
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
    generate(args)
