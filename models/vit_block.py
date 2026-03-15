import torch
import torch.nn as nn

class VisionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

if __name__ == "__main__":
    model = VisionTransformerBlock(dim=512)
    sample_input = torch.randn(1, 197, 512) # [B, N, D]
    output = model(sample_input)
    print(f"ViT Block Input: {sample_input.shape} -> Output: {output.shape}")