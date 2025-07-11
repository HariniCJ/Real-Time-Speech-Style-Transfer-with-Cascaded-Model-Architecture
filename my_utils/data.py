
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from my_utils import audio
from my_utils.audio import (extract_mel_spectrogram, normalize_amplitude,
                            random_crop)


class RAVDESSDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, audio_length, sr):
        self.root_dir = root_dir
        self.file_paths = self._get_file_paths()
        self.audio_length = audio_length
        self.sr = sr

    def _get_file_paths(self):
        file_paths = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.endswith(".wav"):
                    file_paths.append(os.path.join(dirpath, fname))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(audio_path)  # [C, T_orig], sr

        # Resample if needed
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            waveform = resampler(waveform)

        waveform = self._preprocess_audio(waveform)  # [1, T]

        mel_spectrogram = extract_mel_spectrogram(waveform, sr=self.sr)  # [1, n_mels, T']
        mel_spectrogram = mel_spectrogram.squeeze(0)  # remove batch dim -> [n_mels, T']

        return waveform, mel_spectrogram

    def _preprocess_audio(self, waveform):
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

        # Convert to numpy for external utils
        waveform_np = waveform.numpy()

        # Normalize amplitude safely
        waveform_np = normalize_amplitude(waveform_np)

        # Random crop or pad to target length
        waveform_np = random_crop(waveform_np, self.audio_length, self.sr)

        # Convert back to tensor
        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        return waveform

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """
    Args:
        dataset (Dataset): PyTorch Dataset object.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

