import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)

class Encoder3D(nn.Module):
    def __init__(self, in_channels, hidden_dims):
        super().__init__()
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.res_blocks = nn.Sequential(
            ResBlock3D(hidden_dims[-1], hidden_dims[-1]),
            ResBlock3D(hidden_dims[-1], hidden_dims[-1])
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.encoder(x)
        x = self.res_blocks(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResBlock3D(in_channels, in_channels),
            ResBlock3D(in_channels, in_channels)
        )
        
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(hidden_dims[i+1]),
                    nn.ReLU()
                )
            )
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(hidden_dims[-1], out_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh() # Assuming input normalized to [-1, 1]
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, C, T, H, W) -> C is embedding_dim
        # Convert inputs from BCHW to BHWC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Permute back to (B, C, T, H, W)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[1], input_shape[2], input_shape[3]) # (B, T, H, W)
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices

class VQVAE3D(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=[64, 128], codebook_size=512, embed_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder3D(in_channels, hidden_dims)
        self.pre_vq_conv = nn.Conv3d(hidden_dims[-1], embed_dim, kernel_size=1)
        self.vq = VectorQuantizer(codebook_size, embed_dim, commitment_cost)
        self.decoder = Decoder3D(embed_dim, hidden_dims, in_channels)

    def forward(self, x):
        # x: (B, C, T, H, W)
        encoded = self.encoder(x)
        quantized, vq_loss, perplexity, indices = self.vq(self.pre_vq_conv(encoded))
        decoded = self.decoder(quantized)
        return decoded, vq_loss, perplexity, indices

    def encode_to_indices(self, x):
        encoded = self.encoder(x)
        _, _, _, indices = self.vq(self.pre_vq_conv(encoded))
        return indices # (B, T, H, W)

    def decode_from_indices(self, indices):
        # indices: (B, T, H, W)
        flat_indices = indices.view(-1)
        quantized = self.vq._embedding(flat_indices) # (B*T*H*W, embed_dim)
        quantized = quantized.view(indices.shape[0], indices.shape[1], indices.shape[2], indices.shape[3], -1) # (B, T, H, W, C)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous() # (B, C, T, H, W)
        return self.decoder(quantized)
