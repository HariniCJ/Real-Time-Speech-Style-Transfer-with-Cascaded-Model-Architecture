import librosa
import numpy as np
import torch
import torchaudio


def load_audio(path, sr=16000):
    """Loads and resamples audio to the target sample rate."""
    wav, _ = librosa.load(path, sr=sr)
    return wav


def extract_mel_spectrogram(
    audio,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000,
):
    """
    Args:
        audio (np.ndarray or torch.Tensor): shape [T] or [1, T] or [B, T]
    Returns:
        mel_spectrogram (torch.Tensor): shape [B, n_mels, T']
    """

    # Convert to torch.Tensor if it's a NumPy array
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    # Ensure float32 and 2D [B, T]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # [1, T]
    elif audio.ndim == 2 and audio.shape[0] == 1:
        pass  # already [1, T]
    elif audio.ndim == 2:
        pass  # [B, T]
    else:
        raise ValueError(f"Unexpected audio shape {audio.shape}")

    audio = audio.float()  # Ensure float32

    # Create MelSpectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
    )

    mel_spectrogram = mel_transform(audio)  # [B, n_mels, T']

    # Convert to decibel scale
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    mel_db = db_transform(mel_spectrogram)

    return mel_db.float()


def normalize_amplitude(audio):
    """Normalizes the audio signal to have a max absolute amplitude of 1."""
    max_val = np.max(np.abs(audio))
    if max_val < 1e-9:
        return audio
    return audio / max_val


def random_crop(audio, target_length, sr=16000):
    """Randomly crops or pads the audio to the target length.

    Args:
        audio (numpy.ndarray): Audio waveform of shape [1, T] or [T]
        target_length (int): Target length of the audio.
        sr (int): Sample rate.

    Returns:
        numpy.ndarray: Cropped or padded audio waveform of shape [1, target_length]
    """

    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)  # [1, T]

    audio_len = audio.shape[1]
    if audio_len < target_length:
        padding = np.zeros((1, target_length - audio_len), dtype=audio.dtype)
        audio = np.concatenate([audio, padding], axis=1)  # [1, target_length]
    elif audio_len > target_length:
        start = np.random.randint(0, audio_len - target_length)
        audio = audio[:, start : start + target_length]  # [1, target_length]
    return audio
