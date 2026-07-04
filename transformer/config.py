import argparse

import torch

def get_transformer_args():
    parser = argparse.ArgumentParser(description="Transformer Training")
    parser.add_argument("--data_path", type=str, default="data/mnist_test_seq.npy")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # VQVAE args to load the frozen model
    parser.add_argument("--vqvae_ckpt", type=str, default="checkpoints/vqvae/vqvae_epoch_100.pt")
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--vq_embed_dim", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--vq_hidden_dims", type=int, nargs='+', default=[64, 128])
    
    # Transformer args
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints/transformer")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--no_wandb", action="store_true")
    
    return parser.parse_args()
