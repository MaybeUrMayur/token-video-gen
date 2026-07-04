import argparse

import torch

def get_vqvae_args():
    parser = argparse.ArgumentParser(description="VQ-VAE Training")
    parser.add_argument("--data_path", type=str, default="data/mnist_test_seq.npy")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model args
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[64, 128])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    
    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints/vqvae")
    parser.add_argument("--recon_dir", type=str, default="outputs/vqvae_recon")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    
    return parser.parse_args()
