import torch
import torch.nn as nn


class StyleModulator(nn.Module):
    def __init__(self, content_dim, speaker_dim, style_dim, hidden_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(content_dim + speaker_dim + style_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, content_dim),
        )

    def forward(self, content, speaker_embedding, style_embedding):
        speaker_embedding = speaker_embedding.unsqueeze(1).expand(
            -1, content.shape[1], -1
        )  # [B, T', speaker_dim]
        style_embedding = style_embedding.unsqueeze(1).expand(
            -1, content.shape[1], -1
        )  # [B, T', style_dim]
        x = torch.cat(
            [content, speaker_embedding, style_embedding], dim=-1
        )  # [B, T', content_dim + speaker_dim + style_dim]
        modulated_content = self.fusion(x)  # [B, T', content_dim]
        return modulated_content
