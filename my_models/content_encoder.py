import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class ContentEncoder(nn.Module):

    def __init__(self, output_dim, pretrained=True, freeze_base=False):
        """
        Args:
            output_dim (int): Dimension of the output content features.
            pretrained (bool): Whether to load pretrained Wav2Vec2.
            freeze_base (bool): Whether to freeze Wav2Vec2 backbone.
        """
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base") if pretrained else Wav2Vec2Model.from_config(
            Wav2Vec2Model.config_class()
        )
        self.projection = nn.Linear(768, output_dim)
        self.freeze_base = freeze_base

        if self.freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False  # Freeze backbone

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input waveform of shape [B, 1, T]

        Returns:
            torch.Tensor: Output content features of shape [B, T', output_dim]
        """
        print(f"Shape of input to ContentEncoder: {x.shape}")  # Debugging
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]

        if self.freeze_base:
            with torch.no_grad():
                wav2vec_output = self.wav2vec2(x).last_hidden_state
        else:
            wav2vec_output = self.wav2vec2(x).last_hidden_state

        content = self.projection(wav2vec_output)  # [B, T', output_dim]
        return content
