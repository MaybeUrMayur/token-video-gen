import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class MovingMNISTDataset(Dataset):
    def __init__(self, data_path="data/mnist_test_seq.npy", resolution=32, clip_length=8, max_clips=None):
        """
        Moving MNIST dataset.
        data_path: path to the .npy file
        resolution: int, downsample resolution (e.g., 32 or 16)
        clip_length: int, number of frames per sequence
        max_clips: int, limit the dataset size for fast iteration
        """
        super().__init__()
        self.resolution = resolution
        self.clip_length = clip_length
        
        # Load data. Original shape is (T, B, W, H) -> (20, 10000, 64, 64)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}. Run download_moving_mnist.py first.")
            
        data = np.load(data_path)
        
        # Reshape to (B, T, H, W)
        data = np.transpose(data, (1, 0, 2, 3))
        
        # Convert to torch tensor
        self.data = torch.from_numpy(data).float()
        
        # We can extract multiple non-overlapping clips of `clip_length` from each 20-frame video
        # E.g., if clip_length=8, we can get 2 clips per video (frames 0-7, 8-15)
        num_clips_per_video = self.data.shape[1] // self.clip_length
        total_videos = self.data.shape[0]
        
        # Reshape into a flat list of clips
        clips = []
        for i in range(num_clips_per_video):
            start = i * self.clip_length
            end = start + self.clip_length
            clips.append(self.data[:, start:end, :, :])
            
        self.clips = torch.cat(clips, dim=0) # Shape: (B * num_clips_per_video, clip_length, H, W)
        
        if max_clips is not None:
            self.clips = self.clips[:max_clips]
            
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx] # (T, H, W)
        
        # Add channel dim: (T, C, H, W)
        clip = clip.unsqueeze(1)
        
        # Downsample if needed
        if self.resolution != 64: # Original is 64x64
            # We resize frame by frame or batch resize
            clip = F.resize(clip, [self.resolution, self.resolution], antialias=True)
            
        # Normalize to [-1, 1]. Original is [0, 255]
        clip = (clip / 255.0) * 2.0 - 1.0
        
        return clip

def get_dataloader(data_path="data/mnist_test_seq.npy", batch_size=16, resolution=32, clip_length=8, max_clips=None, num_workers=4, shuffle=True):
    dataset = MovingMNISTDataset(
        data_path=data_path, 
        resolution=resolution, 
        clip_length=clip_length, 
        max_clips=max_clips
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
