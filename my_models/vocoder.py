
import json
import os

import torch
import torch.nn as nn

from hifi_gan.inference import AttrDict
from hifi_gan.models import Generator


class HiFiGANVocoder(nn.Module):



    def __init__(self, checkpoint_path="hifi_gan/g_02500000", config_path="hifi_gan/config.json"):
        """
        Args:
            checkpoint_path (str): Path to the HiFi-GAN checkpoint.
            config_path (str): Path to the HiFi-GAN config file.
        """
        super().__init__()
        if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
            raise FileNotFoundError("HiFi-GAN checkpoint or config file not found.")

        with open(config_path) as f:
            config_data = json.load(f)
        h = AttrDict(config_data)

        self.hifigan = Generator(h).to("cpu")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.hifigan.load_state_dict(state_dict["generator"])
        self.hifigan.eval()
        for param in self.hifigan.parameters():
            param.requires_grad = False

        # New projection layer to map from 256 → 80 channels
        self.project_to_mel = nn.Conv1d(in_channels=256, out_channels=80, kernel_size=1)

    def forward(self, modulated_content):
        """
        Args:
            modulated_content (torch.Tensor): Feature tensor [B, 256, T']

        Returns:
            torch.Tensor: Generated audio waveform of shape [B, 1, T_audio]
        """
        mel_like = self.project_to_mel(modulated_content)  # [B, 80, T']
        audio = self.hifigan(mel_like)  # [B, 1, T_audio]
        return audio

