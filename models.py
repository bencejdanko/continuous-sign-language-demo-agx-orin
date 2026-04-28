import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEncoder(nn.Module):
    """1D-CNN: [B, T=60, F=540] → [B, D=512, T'=15]"""
    def __init__(self, input_dim=540, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(512, latent_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):          # x: [B, T, F]
        x = x.transpose(1, 2)     # [B, F, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x                   # [B, D, T']


class DiffusionDecoder(nn.Module):
    """1D-UNet conditioned on Z; predicts noise."""
    def __init__(self, input_dim=540, latent_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.GELU(), nn.Linear(128, 128)
        )
        self.upsample_z = nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=4, stride=4)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim + latent_dim + 128, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x, z, t):
        x = x.transpose(1, 2)                                        # [B, F, T]
        t_emb = self.time_mlp(t.float()).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        z_up  = self.upsample_z(z)
        out   = self.net(torch.cat([x, z_up, t_emb], dim=1))
        return out.transpose(1, 2)                                    # [B, T, F]


class TranslationModel(nn.Module):
    """FLAN-T5-small fed with latent Z [B, 15, 512] as encoder embeddings."""
    def __init__(self):
        super().__init__()
        from transformers import T5ForConditionalGeneration
        self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    def forward(self, z, labels=None):
        return self.t5(inputs_embeds=z, labels=labels) if labels is not None                else self.t5.generate(inputs_embeds=z)
