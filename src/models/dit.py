import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super(AdaptiveLayerNorm, self).__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x, cond, scale_transform: nn.Parameter, shift_transform: nn.Parameter):
        B, T, D = x.shape
        x = self.layer_norm(x)
        x = x * (1 + scale_transform.unsqueeze(1)) + shift_transform.unsqueeze(1)

        return x

class DitBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(DitBlock, self).__init__()

        self. adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 6)
        )

        self.mha_adaln = AdaptiveLayerNorm(embed_dim)
        self.mlp_adaln = AdaptiveLayerNorm(embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, cond):
        self.shift_mha, self.scale_mha, self.gate_mha, self.shift_mlp, self.scale_mlp, self.gate_mlp = self.adaln_modulation(cond).chunk(6, dim=1)

        norm_x = self.mha_adaln(x, cond, self.scale_mha, self.shift_mha)
        attn_out = self.mha(norm_x, norm_x, norm_x, need_weights=False)[0]
        x = x + self.gate_mha.unsqueeze(1) * attn_out

        norm_x = self.mlp_adaln(x, cond, self.scale_mlp, self.shift_mlp)
        mlp_out = self.mlp(norm_x)
        x = x + self.gate_mlp.unsqueeze(1) * mlp_out
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super(TimestepEmbedding, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        embedding = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(embedding)

class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(DiffusionTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.time_embedding = TimestepEmbedding(embed_dim)
        self.layers = nn.ModuleList([DitBlock(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.final_adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

        self.lm_head = nn.Sequential(
            nn.Linear(embed_dim, vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, t, cond = None):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.position_embedding(positions)
        t = self.time_embedding(t)
        print(x.shape, t.shape)
        if not cond:
            cond = torch.zeros_like(t)

        cond = t + cond

        for layer in self.layers:
            x = layer(x, cond)

        shift, scale = self.final_adaln_modulation(cond).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        logits = self.lm_head(x)

        return logits