import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# Component 1: Sinusoidal Time Embedding
# ============================================

class TimeEmbedding(nn.Module):
    """
    Converts integer timestep (e.g. 500) into a rich vector (e.g. 512 dims).
    Same math as Transformer positional encoding, then projected through
    two linear layers.

    Input:  (batch_size,)        — integer timesteps
    Output: (batch_size, embed_dim)  — embedding vectors
    """

    def __init__(self, embed_dim):
        super().__init__()
        # embed_dim is typically base_channels * 4 (e.g. 512)
        self.embed_dim = embed_dim
        # sin/cos concat yields embed_dim, not embed_dim // 2
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        # t is shape (batch_size,)

        half_dim = self.embed_dim // 2

        # Build frequencies that span many scales
        # range [0, 1, 2, ..., half_dim-1] divided by half_dim
        # multiplied by -log(10000) then exponentiated
        # Result: frequencies from high to low
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )

        # Multiply each timestep by all frequencies
        # t[:, None] is (batch, 1)
        # frequencies[None, :] is (1, half_dim)
        # Result: (batch, half_dim)
        args = t[:, None].float() * frequencies[None, :]

        # Sin and cos give different information, concatenate them
        # Result: (batch, embed_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Project through two linear layers with activation
        embedding = self.linear1(embedding)
        embedding = F.silu(embedding)
        embedding = self.linear2(embedding)

        return embedding  # (batch, embed_dim)


# ============================================
# Component 2: Residual Block with Time Conditioning
# ============================================

class ResidualBlock(nn.Module):
    """
    The core repeating unit. Used ~20 times throughout the U-Net.

    Does: GroupNorm -> SiLU -> Conv -> [add time info] -> GroupNorm -> SiLU -> Conv
    Plus a skip connection from input to output.

    Input:  x         (batch, in_channels, height, width)
            time_emb  (batch, time_embed_dim)
    Output:           (batch, out_channels, height, width)
    """

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()

        # First convolution path
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection: maps time vector to channel count
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # Second convolution path
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        # If channel count changes, need 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):

        residual = x

        # First half: norm -> activate -> convolve
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Inject time information
        # Project time_emb from (batch, time_embed_dim) to (batch, out_channels)
        # Reshape to (batch, out_channels, 1, 1) for broadcasting across pixels
        time_signal = self.time_proj(F.silu(time_emb))
        time_signal = time_signal[:, :, None, None]  # add spatial dims
        h = h + time_signal

        # Second half: norm -> activate -> convolve
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Add skip connection
        return h + self.shortcut(residual)


# ============================================
# Component 3: Self-Attention Block
# ============================================

class AttentionBlock(nn.Module):
    """
    Self-attention over all spatial positions.
    Lets distant parts of the image influence each other.
    Only used at 16x16 resolution for efficiency.

    Input:  (batch, channels, height, width)
    Output: (batch, channels, height, width)  — same shape
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)

        # Q, K, V projections as 1x1 convolutions
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Scaling factor for dot product attention
        self.scale = channels ** -0.5

    def forward(self, x):

        residual = x
        h = self.norm(x)

        # Compute queries, keys, values
        q = self.query(h)   # (batch, channels, H, W)
        k = self.key(h)
        v = self.value(h)

        # Reshape from (batch, channels, H, W) to (batch, channels, H*W)
        # This flattens the spatial dimensions so we can do attention
        # across all positions
        b, c, H, W = q.shape
        q = q.reshape(b, c, H * W)
        k = k.reshape(b, c, H * W)
        v = v.reshape(b, c, H * W)

        # Compute attention weights
        # q.transpose: (batch, H*W, channels)
        # k:           (batch, channels, H*W)
        # matmul:      (batch, H*W, H*W)  — attention matrix
        attention = torch.bmm(q.transpose(1, 2), k) * self.scale
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        # v:         (batch, channels, H*W)
        # attention: (batch, H*W, H*W)
        # result:    (batch, channels, H*W)
        out = torch.bmm(v, attention.transpose(1, 2))

        # Reshape back to spatial: (batch, channels, H, W)
        out = out.reshape(b, c, H, W)
        out = self.out_proj(out)

        # Residual connection
        return out + residual


# ============================================
# Component 4: The Full U-Net
# ============================================

class UNet(nn.Module):
    """
    Complete U-Net for DDPM noise prediction.

    Input:
        x: noisy images  (batch, 3, 32, 32)
        t: timesteps      (batch,)

    Output:
        predicted noise   (batch, 3, 32, 32)

    Structure:
        Downsampling:  32x32 (ch1) -> 16x16 (ch2) -> 8x8 (ch3) -> 4x4 (ch3)
        Bottleneck:    4x4 (ch3) with attention
        Upsampling:    4x4 (ch3) -> 8x8 (ch3) -> 16x16 (ch2) -> 32x32 (ch1)
        Attention at 16x16 in both down and up paths
        Skip connections between matching resolutions
    """

    def __init__(self, image_channels=3, base_channels=128):
        super().__init__()

        # Channel counts at each resolution
        ch1 = base_channels       # 128, at 32x32
        ch2 = base_channels * 2   # 256, at 16x16
        ch3 = base_channels * 4   # 512, at 8x8 and bottleneck

        # Time embedding dimension
        time_dim = base_channels * 4  # 512

        # --- Time embedding ---
        self.time_embed = TimeEmbedding(time_dim)

        # --- Initial conv: 3 channels -> ch1 channels ---
        self.input_conv = nn.Conv2d(image_channels, ch1, kernel_size=3, padding=1)

        # --- Downsampling path ---

        # Level 1: 32x32, ch1 channels
        self.down1_block1 = ResidualBlock(ch1, ch1, time_dim)
        self.down1_block2 = ResidualBlock(ch1, ch1, time_dim)
        self.downsample1 = nn.Conv2d(ch1, ch1, kernel_size=4, stride=2, padding=1)
        # Output: 16x16

        # Level 2: 16x16, ch2 channels (ATTENTION HERE)
        self.down2_block1 = ResidualBlock(ch1, ch2, time_dim)
        self.down2_block2 = ResidualBlock(ch2, ch2, time_dim)
        self.down2_attention = AttentionBlock(ch2)
        self.downsample2 = nn.Conv2d(ch2, ch2, kernel_size=4, stride=2, padding=1)
        # Output: 8x8

        # Level 3: 8x8, ch3 channels
        self.down3_block1 = ResidualBlock(ch2, ch3, time_dim)
        self.down3_block2 = ResidualBlock(ch3, ch3, time_dim)
        self.downsample3 = nn.Conv2d(ch3, ch3, kernel_size=4, stride=2, padding=1)
        # Output: 4x4

        # --- Bottleneck: 4x4 ---
        self.bottleneck_block1 = ResidualBlock(ch3, ch3, time_dim)
        self.bottleneck_attention = AttentionBlock(ch3)
        self.bottleneck_block2 = ResidualBlock(ch3, ch3, time_dim)

        # --- Upsampling path ---
        # IMPORTANT: first block at each level takes DOUBLE channels
        # because skip connections concatenate down features with up features

        # Level 3: 4x4 -> 8x8
        self.upsample3 = nn.ConvTranspose2d(ch3, ch3, kernel_size=4, stride=2, padding=1)
        self.up3_block1 = ResidualBlock(ch3 + ch3, ch3, time_dim)  # ch3*2 input from concat
        self.up3_block2 = ResidualBlock(ch3, ch3, time_dim)

        # Level 2: 8x8 -> 16x16 (ATTENTION HERE)
        self.upsample2 = nn.ConvTranspose2d(ch3, ch2, kernel_size=4, stride=2, padding=1)
        self.up2_block1 = ResidualBlock(ch2 + ch2, ch2, time_dim)  # ch2*2 input from concat
        self.up2_block2 = ResidualBlock(ch2, ch2, time_dim)
        self.up2_attention = AttentionBlock(ch2)

        # Level 1: 16x16 -> 32x32
        self.upsample1 = nn.ConvTranspose2d(ch2, ch1, kernel_size=4, stride=2, padding=1)
        self.up1_block1 = ResidualBlock(ch1 + ch1, ch1, time_dim)  # ch1*2 input from concat
        self.up1_block2 = ResidualBlock(ch1, ch1, time_dim)

        # --- Output: ch1 -> image channels ---
        self.output_norm = nn.GroupNorm(num_groups=32, num_channels=ch1)
        self.output_conv = nn.Conv2d(ch1, image_channels, kernel_size=3, padding=1)


    def forward(self, x, t):
        # x: (batch, 3, 32, 32)
        # t: (batch,)

        # Time embedding
        time_emb = self.time_embed(t)     # (batch, time_dim)

        # Initial conv
        x = self.input_conv(x)            # (batch, ch1, 32, 32)

        # === DOWNSAMPLING ===

        # 32x32
        d1 = self.down1_block1(x, time_emb)
        d1 = self.down1_block2(d1, time_emb)
        # SAVE d1 for skip connection later — shape (batch, ch1, 32, 32)

        # 32x32 -> 16x16
        d2 = self.downsample1(d1)
        d2 = self.down2_block1(d2, time_emb)
        d2 = self.down2_block2(d2, time_emb)
        d2 = self.down2_attention(d2)
        # SAVE d2 for skip connection later — shape (batch, ch2, 16, 16)

        # 16x16 -> 8x8
        d3 = self.downsample2(d2)
        d3 = self.down3_block1(d3, time_emb)
        d3 = self.down3_block2(d3, time_emb)
        # SAVE d3 for skip connection later — shape (batch, ch3, 8, 8)

        # === BOTTLENECK ===

        # 8x8 -> 4x4
        h = self.downsample3(d3)          # (batch, ch3, 4, 4)
        h = self.bottleneck_block1(h, time_emb)
        h = self.bottleneck_attention(h)
        h = self.bottleneck_block2(h, time_emb)

        # === UPSAMPLING ===

        # 4x4 -> 8x8
        u3 = self.upsample3(h)             # (batch, ch3, 8, 8)
        u3 = torch.cat([u3, d3], dim=1)    # (batch, ch3*2, 8, 8) — skip connection!
        u3 = self.up3_block1(u3, time_emb)
        u3 = self.up3_block2(u3, time_emb)

        # 8x8 -> 16x16
        u2 = self.upsample2(u3)            # (batch, ch2, 16, 16)
        u2 = torch.cat([u2, d2], dim=1)    # (batch, ch2*2, 16, 16) — skip connection!
        u2 = self.up2_block1(u2, time_emb)
        u2 = self.up2_block2(u2, time_emb)
        u2 = self.up2_attention(u2)

        # 16x16 -> 32x32
        u1 = self.upsample1(u2)            # (batch, ch1, 32, 32)
        u1 = torch.cat([u1, d1], dim=1)    # (batch, ch1*2, 32, 32) — skip connection!
        u1 = self.up1_block1(u1, time_emb)
        u1 = self.up1_block2(u1, time_emb)

        # === OUTPUT ===

        out = self.output_norm(u1)
        out = F.silu(out)
        out = self.output_conv(out)        # (batch, 3, 32, 32)

        return out


# ============================================
# TESTING — verify shapes
# ============================================

if __name__ == "__main__":
    torch.manual_seed(0)

    print("Testing TimeEmbedding...")
    te = TimeEmbedding(512)
    t = torch.randint(0, 1000, (8,))
    assert te(t).shape == (8, 512)

    print("Testing ResidualBlock...")
    rb = ResidualBlock(128, 256, 512)
    x = torch.randn(8, 128, 32, 32)
    t_emb = torch.randn(8, 512)
    assert rb(x, t_emb).shape == (8, 256, 32, 32)

    print("Testing AttentionBlock...")
    ab = AttentionBlock(256)
    x = torch.randn(8, 256, 16, 16)
    assert ab(x).shape == (8, 256, 16, 16)

    print("Testing full UNet...")
    model = UNet()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    assert model(x, t).shape == (4, 3, 32, 32)
    print("All passed!")

    # Count parameters — should be around 35-40M for base_channels=128
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
