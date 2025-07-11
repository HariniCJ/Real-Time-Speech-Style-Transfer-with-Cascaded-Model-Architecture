
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class SpeakerEncoder(nn.Module):

    def __init__(self, input_dim=80, embed_dim=128, num_residual_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(128, kernel_size=3, dilation=1) for _ in range(num_residual_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity='relu')
        if self.conv_in.bias is not None:
            nn.init.constant_(self.conv_in.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, mel):
        mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=-1.0)
        mel = torch.clamp(mel, min=-10.0, max=10.0)

        print("🔍 [SpeakerEncoder] Input mel:", mel.shape)
        print("NaNs in mel?", torch.isnan(mel).any())

        x = self.conv_in(mel)
        x = self.relu(x)

        x = self.res_blocks(x)
        print("After residual blocks:", x.shape, "NaNs?", torch.isnan(x).any())

        x = self.pool(x)
        print("After adaptive pooling:", x.shape, "NaNs?", torch.isnan(x).any())

        x = x.view(x.size(0), -1)
        print("After flatten:", x.shape, "NaNs?", torch.isnan(x).any())

        out = self.fc(x)
        print("✅ Speaker embedding:", out.shape, "NaNs?", torch.isnan(out).any())
        return out

