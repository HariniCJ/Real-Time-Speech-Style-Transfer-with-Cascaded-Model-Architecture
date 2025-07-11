
import torch
import torch.nn as nn


class ResidualBlockGN(nn.Module):

    def __init__(self, channels, kernel_size=3, stride=1, groups=8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding)
        self.gn2 = nn.GroupNorm(groups, channels)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class StyleEncoder(nn.Module):

    def __init__(self, input_dim=80, embed_dim=128):
        super().__init__()
        self.conv_in = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.gn_in = nn.GroupNorm(8, 128)
        self.relu = nn.ReLU()

        # Add residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlockGN(128),
            ResidualBlockGN(128),
        )

        self.downsample1 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.gn_down1 = nn.GroupNorm(8, 256)

        self.downsample2 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.gn_down2 = nn.GroupNorm(8, 256)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Linear(256, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv_in, self.downsample1, self.downsample2, self.embedding_layer]:
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mel_spectrogram):
        mel_spectrogram = torch.nan_to_num(mel_spectrogram, nan=0.0, posinf=1.0, neginf=-1.0)
        mel_spectrogram = torch.clamp(mel_spectrogram, min=-10.0, max=10.0)

        print("🔍 [StyleEncoder] Input mel stats — min:", mel_spectrogram.min().item(),
              "max:", mel_spectrogram.max().item(),
              "mean:", mel_spectrogram.mean().item())

        x = self.relu(self.gn_in(self.conv_in(mel_spectrogram)))
        x = self.res_blocks(x)

        x = self.relu(self.gn_down1(self.downsample1(x)))
        x = self.relu(self.gn_down2(self.downsample2(x)))

        x = self.global_avg_pool(x).squeeze(-1)
        style_embedding = self.embedding_layer(x)

        print("✅ Style embedding:", style_embedding.shape, "NaNs?", torch.isnan(style_embedding).any())
        return style_embedding

