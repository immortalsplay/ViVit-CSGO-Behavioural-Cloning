import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange
import math
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################################
#                                Transformer Model                              #
#################################################################################

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),   # dim=1024, hidden_dim=2048
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer_block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        """
        x: batch * horizon * dim
        """
        for attn, norm, ff in self.layers:
            x = attn(norm(x)) + x
            x = ff(x) + x
        return x

#################################################################################
#                                   CSGO Model                                  #
#################################################################################

class InvDynamic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class CSGO_model(nn.Module):
    def __init__(self, horizon, num_feature, depth, num_heads, head_dim, inverse_dynamic_dim):
        super().__init__()

        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, num_feature)

        self.transformer = Transformer_block(dim = num_feature, 
                                             depth = depth,
                                             heads = num_heads, 
                                             dim_head = head_dim,
                                             mlp_dim = 2048,
                                             dropout = 0.
                                             )
        
        self.fc_outputs = nn.ModuleList(
            [InvDynamic(num_feature * 2, 1, inverse_dynamic_dim) for _ in range(26)]
            )

    def forward(self, x):
        """
        x: [batch_size, horizon, channel, height, width]
        """
        batch_size, horizon, _, _, _ = x.shape

        batch_size = x.shape[0]

        x = rearrange(x, 'b h c x y -> (b h) c x y')
        x = self.encoder(x)
        x = rearrange(x, '(b h) d -> b h d', b=batch_size)

        x = self.transformer(x)  

        x_t = x[:, 0, :]
        x_t_1 = x[:, 1, :]
        x_comb = torch.cat([x_t, x_t_1], dim=1)

        outputs = torch.zeros([batch_size, 26, 1], device=x.device)

        for b in range(batch_size):
            i = 0
            for fc in self.fc_outputs:
                outputs[b, i] = fc(x_comb)
                i += 1

        return outputs
